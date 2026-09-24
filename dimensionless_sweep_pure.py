"""
dimensionless_sweep_pure.py
===========================
Sweep driver for the pure-metal / eutectic remelting correlation.

This is the pure-metal counterpart of dimensionless_sweep_scheil.py.
The two scripts share the same:
  - operating-point grid (ste_cold_op × ste_hot_op from PARAMS)
  - sizing pipeline (size_run → build_grid)
  - inline refinement logic
  - checkpoint / resume protocol
  - output column conventions (where applicable)

They differ only in:
  - LHS axes: pure has 3 (E, C, r); Scheil has 5 (E, C, r, mushy_frac, k_p)
  - Solver model: 'pure_metal' vs 'scheil'
  - Output columns: pure omits mushy_frac, k_p, theta_solidus, Ste_m,
                    D_sol (= D_rem for pure), mushy_width_at_max_remelt

Group algebra (pure metal)
--------------------------
  B          = E / √r              [(ρcp)_l / (ρcp)_s]
  cp_ratio   = C
  rho_ratio  = B / C               = E / (C√r)
  k_ratio    = r · B               = E√r

  theta_melt = ste_cold / (ste_cold + ste_hot/C)
             = ste_cold · inv_Ste
  inv_Ste    = 1 / (ste_cold + ste_hot/C)   [= L / (cp_s · dT_ref)]

The pure kernel (_fd_solver_core_pure) takes:
  theta_melt, inv_Ste, k_ratio, rho_ratio, cp_ratio

All five are derived exactly from (E, C, r, ste_cold, ste_hot).

Relationship to existing results_refined.csv
---------------------------------------------
  results_refined.csv was produced by the pre-pipeline pure-metal sweep.
  This script produces output that is:
    - column-compatible with the analysis pipeline (analyse_sweep_pure.py)
    - comparable to results_refined.csv after renaming theta_melt ↔ theta_liquidus
  The existing data is valid and need not be regenerated, but this script
  can extend it or regenerate it reproducibly from a fixed seed.

Usage
-----
  python dimensionless_sweep_pure.py --n_lhs 1000 --workers 8
  python dimensionless_sweep_pure.py --n_lhs 1000 --workers 8 --resume results_pure/progress.csv
  python dimensionless_sweep_pure.py --n_lhs 1000 --dry_run
  python dimensionless_sweep_pure.py --n_lhs 1000 --workers 8 --neumann

--neumann flag
--------------
  When set, computes the Neumann similarity eigenvalue λ (solve_stefan_similarity)
  before each FD run and uses it to:
    1. Improve domain sizing: Z = max(1 + 2λ√(rt) + z_pen, Z_floor)
       instead of Z = max(1 + z_pen, Z_floor).  For deep-remelting cases this
       can save allocating an unnecessarily large domain.
    2. Store lam_neumann as an output column (physically meaningful feature:
       corr(log λ, log D_rem) ≈ 0.79 on the resolved population).
    3. Provide a per-case analytical cross-check (λ exists ↔ onset criterion met).
  Cost: ~0.05–0.2 s per case (1D root-find vs 2D grid scan for Scheil).
  The flag defaults to False to keep the sweep fast for initial runs.

Note on existing data
---------------------
  If you want to analyse results_refined.csv with analyse_sweep_pure.py,
  pass --legacy_csv results_refined.csv to that script instead of re-running
  this sweep.  The column mapping is documented in analyse_sweep_pure.py.

CHANGELOG
---------
  [fix] run_one()'s four return paths (guard-skip, sizing-error, solver-error,
        success) previously built their row dicts independently, with
        different key sets — the skip/error paths spread **adim (which
        includes 'theta_liquidus' and 'B', but never 'theta_melt' or
        'config_fingerprint'), while the success path used explicit keys
        (including 'theta_melt' and 'config_fingerprint', but never
        'theta_liquidus' or 'B'). If a checkpoint batch (P.checkpoint_every
        rows) happened to land entirely on one path — e.g. all guard-skips —
        pd.DataFrame(buf) built a narrower-than-header DataFrame for that
        batch, and writing it with header=False silently misaligned every
        column for the whole batch. Fixed by introducing ROW_SCHEMA + a
        _blank_row() helper: every return path now builds from the identical
        23-key schema, so column-count drift between batches is no longer
        possible. As a side effect, 'theta_contact' is now populated on
        skip/error rows too (previously only on success rows) — this is a
        strict improvement, not a behaviour change anyone should rely on
        the absence of.
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).parent))

from config_pipeline import CONVENTIONS as CV, PARAMS as P
from grid import build_grid, size_run
from solvers import fd_solve
from analytical import solve_stefan_similarity


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Sweep configuration
# ═══════════════════════════════════════════════════════════════════════════════

# LHS bounds — 3 material-class axes for pure metal
LHS_AXES = {
    'E': ('linear', 0.40, 3.00),
    'C': ('linear', 0.60, 2.00),
    'r': ('log',    0.20, 2.00),
}

STE_COLD_OP = P.ste_cold_op   # (0.10, 0.25, 0.50, 0.90, 1.40, 2.00)
STE_HOT_OP  = np.linspace(0.15, 6.00, 20)    # (0.05, 0.15, 0.30, 0.60, 1.00)

PHYS_AXES = {
    'k_ratio':   ('log',    0.28, 1.28),
    'rho_ratio': ('linear', 0.80, 1.10),
    'cp_ratio':  ('linear', 0.98, 1.45),
}

# Melt-biased operating grid (replaces STE_COLD_OP / STE_HOT_OP for this
# sweep only -- same 6x5=30 shape, shifted toward the melting region).
STE_COLD_OP_REAL = (0.05, 0.10, 0.20, 0.35, 0.55, 0.80)
STE_HOT_OP_REAL  = (0.20, 0.35, 0.55, 0.85, 1.30)

def generate_lhs_real(n_lhs, seed=None):
    """
    Generate n_lhs points on the physical manifold, sampling
    (k_ratio, rho_ratio, cp_ratio) directly and deriving (E, C, r).

    Returns np.ndarray shape (n_lhs, 3), columns [E, C, r] -- same column
    order as generate_lhs(), so build_job_list() needs no changes.
    """
    if seed is None:
        seed = P.rng_seed

    axes   = list(PHYS_AXES.keys())   # ['k_ratio', 'rho_ratio', 'cp_ratio']
    n_axes = len(axes)

    sampler = qmc.LatinHypercube(d=n_axes, rng=seed)
    unit    = sampler.random(n=n_lhs)

    phys = np.empty_like(unit)
    for j, ax in enumerate(axes):
        scale, lo, hi = PHYS_AXES[ax]
        if scale == 'log':
            phys[:, j] = np.exp(unit[:, j] * (np.log(hi) - np.log(lo)) + np.log(lo))
        else:
            phys[:, j] = unit[:, j] * (hi - lo) + lo

    k_ratio, rho_ratio, cp_ratio = phys[:, 0], phys[:, 1], phys[:, 2]

    E = np.sqrt(k_ratio * rho_ratio * cp_ratio)
    C = cp_ratio
    r = k_ratio / (rho_ratio * cp_ratio)

    out = np.column_stack([E, C, r])
    return out   # columns: E, C, r -- same as generate_lhs()

# ── after LHS_AXES, before STE_COLD_OP/STE_HOT_OP ──────────────────────────
# Targeted r-crossing validation sweep (linearity-criterion check).
# 3 (E, C) pairs spanning distinct C/E ratios (tests "E, C cancel" directly,
# rather than assuming it) crossed with a dense log-spaced r grid bracketing
# the r0(Ste_cold) crossing found empirically: FD 0.29-1.34, closure 1.16-2.87.
TARGETED_EC_PAIRS = [
    (0.50, 0.70),   # C/E = 1.40
    (1.50, 1.30),   # C/E = 0.867
    (2.80, 1.90),   # C/E = 0.679
]
TARGETED_R_RANGE = (0.15, 3.50)     # log-spaced, margin around both r0 estimates
TARGETED_N_R     = 18               # -> 3 * 18 = 54 materials

# Dense Ste_hot grid: log-spaced, replaces the 5-pt/2-pt grids so the
# curvature and its zero-crossing are resolved instead of bin-interpolated.
STE_HOT_OP_TARGETED = tuple(np.logspace(np.log10(0.02), np.log10(6.0), 18))

STE_COLD_OP = P.ste_cold_op   # unchanged — (0.10, 0.25, 0.50, 0.90, 1.40, 2.00)
STE_HOT_OP  = P.ste_hot_op    # unchanged default; targeted run overrides via flag

DEFAULT_OUTPUT = Path('results_pure1') / 'progress.csv'

_N_SNAPS = 8

# ═══════════════════════════════════════════════════════════════════════════════
# Canonical row schema
# ═══════════════════════════════════════════════════════════════════════════════
# BUG FIX (see CHANGELOG below): run_one() previously had four separate
# `return {...}` statements (guard-skip, sizing-error, solver-error, success),
# each with a DIFFERENT set of dict keys. pd.DataFrame(buf) infers columns
# from the union of keys actually present in a checkpoint batch — if one
# 500-row batch happened to contain only rows from paths missing some keys
# (e.g. all guard-skips, which never set 'config_fingerprint' or 'theta_melt'),
# that batch's DataFrame had FEWER columns than the file's header, and
# writing it with header=False silently misaligned every value in the batch.
#
# Fix: every return path now builds its row from this single canonical key
# list via `_blank_row()`, so every row dict — regardless of which branch
# produced it — has the exact same 23 keys, in the exact same order, with
# np.nan/False placeholders for whatever that branch doesn't compute.
# This makes column-count drift between checkpoint batches impossible.
ROW_SCHEMA = [
    'lhs_id', 'op_id', 'E', 'C', 'r', 'ste_cold', 'ste_hot',
    'theta_melt', 'inv_Ste', 'k_ratio', 'rho_ratio', 'cp_ratio', 'theta_contact',
    'lam_neumann', 't_total', 'Z_total', 'time_capped', 'domain_capped',
    'D_rem_coarse', 'D_rem', 't_at_max_remelt', 'refined', 'status', 'config_fingerprint',
]


def _blank_row() -> Dict[str, Any]:
    """
    A dict with every ROW_SCHEMA key present, defaulted to NaN (or False for
    the two boolean flags). Callers overwrite only the keys they actually
    compute — every return path ends up with the full, identical key set.

    Typed Dict[str, Any] (not Dict[str, float]) because 'status' is always
    a str by the time any caller reads it, even though it starts as NaN here.
    """
    row: Dict[str, Any] = {k: np.nan for k in ROW_SCHEMA}
    row['time_capped'] = False
    row['domain_capped'] = False
    row['refined'] = False
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Group algebra
# ═══════════════════════════════════════════════════════════════════════════════

def groups_pure(E, C, r, ste_cold, ste_hot):
    """
    Convert (E, C, r, ste_cold, ste_hot) to pure-metal kernel inputs.

    Returns dict with kernel inputs + audit quantities.
    """
    B         = E / np.sqrt(r)
    cp_ratio  = C
    rho_ratio = B / C
    k_ratio   = r * B          # = E√r

    inv_Ste    = 1.0 / (ste_cold + ste_hot / C)
    theta_liquidus = ste_cold * inv_Ste   # = (T_liq-T_sub)/(T_dep-T_sub)

    return {
        # ── Kernel inputs ─────────────────────────────────────────────────────
        'theta_liquidus': theta_liquidus,
        'inv_Ste':        inv_Ste,
        'k_ratio':        k_ratio,
        'rho_ratio':      rho_ratio,
        'cp_ratio':       cp_ratio,
        # ── Audit quantities ──────────────────────────────────────────────────
        'theta_contact':  E / (1.0 + E),
        'B':              B,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Guard-rail checks
# ═══════════════════════════════════════════════════════════════════════════════

def check_guards(adim):
    """Return a reason string if skipped, or None if OK."""
    tm  = adim['theta_liquidus']
    kr  = adim['k_ratio']
    rr  = adim['rho_ratio']

    if not (P.theta_melt_min < tm < P.theta_melt_max):
        return f'theta_melt={tm:.4f} outside [{P.theta_melt_min}, {P.theta_melt_max}]'
    if not (P.k_ratio_min <= kr <= P.k_ratio_max):
        return f'k_ratio={kr:.4f} outside [{P.k_ratio_min}, {P.k_ratio_max}]'
    if not (P.rho_ratio_min <= rr <= P.rho_ratio_max):
        return f'rho_ratio={rr:.4f} outside [{P.rho_ratio_min}, {P.rho_ratio_max}]'
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. Neumann similarity eigenvalue (optional --neumann flag)
# ═══════════════════════════════════════════════════════════════════════════════

def neumann_groups_pure(E, C, r, ste_cold, ste_hot):
    """
    Build the dict that solve_stefan_similarity() expects, from our sweep groups.

    Mapping (all aliases — no new physics):
      E_ratio          = E           (effusivity ratio eff_l/eff_s)
      cp_ratio         = C           (cp_l/cp_s)
      Ste_superheat    = ste_hot     (cp_l·(T_dep−T_liq)/L)
      Ste_undercooling = ste_cold    (cp_s·(T_liq−T_sub)/L)
      r                = r           (diffusivity ratio α_l/α_s)

    Returns nan if solve_stefan_similarity raises or returns non-finite.
    """
    groups = {
        'r'               : r,
        'E_ratio'         : E,
        'cp_ratio'        : C,
        'Ste_superheat'   : ste_hot,
        'Ste_undercooling': ste_cold,
    }
    try:
        lam = solve_stefan_similarity(groups)
        return float(lam) if (lam is not None and np.isfinite(lam)) else np.nan
    except Exception:
        return np.nan

def run_one(args) -> Dict[str, Any]:
    """
    Run one (LHS point × operating point) case.

    Parameters (packed tuple for multiprocessing)
    ---------------------------------------------
    lhs_id                    : int
    op_id                     : int   (0..29)
    E, C, r                   : float — LHS groups
    ste_cold, ste_hot         : float — operating point
    compute_neumann           : bool  — if True, solve for λ before sizing

    Returns
    -------
    dict — flat row for the output CSV
    """
    (lhs_id, op_id, E, C, r, ste_cold, ste_hot, compute_neumann) = args

    base = {
        'lhs_id': lhs_id, 'op_id': op_id,
        'E': E, 'C': C, 'r': r,
        'ste_cold': ste_cold, 'ste_hot': ste_hot,
    }

    # ── Group algebra ─────────────────────────────────────────────────────────
    adim = groups_pure(E, C, r, ste_cold, ste_hot)
    # adim's physics keys (theta_melt, inv_Ste, k_ratio, rho_ratio, cp_ratio,
    # theta_contact) are shared by every return path below; theta_contact and
    # B are audit-only and not part of ROW_SCHEMA (B is recomputable as
    # E/sqrt(r) from the row's own E, r if ever needed downstream).
    physics = {
        'theta_melt':    adim['theta_liquidus'],
        'inv_Ste':       adim['inv_Ste'],
        'k_ratio':       adim['k_ratio'],
        'rho_ratio':     adim['rho_ratio'],
        'cp_ratio':      adim['cp_ratio'],
        'theta_contact': adim['theta_contact'],
    }

    # ── Guard rails ───────────────────────────────────────────────────────────
    reason = check_guards(adim)
    if reason:
        row = _blank_row()
        row.update(base)
        row.update(physics)
        row['status'] = f'skipped_guard:{reason}'
        return row

    # ── Neumann eigenvalue (optional) ─────────────────────────────────────────
    lam_neumann = np.nan
    if compute_neumann:
        lam_neumann = neumann_groups_pure(E, C, r, ste_cold, ste_hot)

    # ── A-priori sizing ───────────────────────────────────────────────────────
    try:
        t_total, Z_total, flags = size_run(E, C, r, ste_cold, ste_hot,
                                           lam_neumann=lam_neumann)
    except Exception as exc:
        row = _blank_row()
        row.update(base)
        row.update(physics)
        row['lam_neumann'] = lam_neumann
        row['status'] = f'sizing_error:{exc}'
        return row

    # ── Coarse-grid solve ─────────────────────────────────────────────────────
    try:
        grid = build_grid(Z_total=Z_total, dz=P.dz_coarse,
                          t_total=t_total, r=r)
        snap_idx = list(np.linspace(0, grid.nt_max - 1, _N_SNAPS,
                                    dtype=np.int64))
        result = fd_solve(adim, grid.z, grid.interface_idx, grid.dt,
                          grid.nt_max, snap_idx, model='pure_metal')
    except Exception as exc:
        row = _blank_row()
        row.update(base)
        row.update(physics)
        row['lam_neumann'] = lam_neumann
        row['t_total'] = t_total
        row['Z_total'] = Z_total
        row['time_capped'] = flags.time_capped
        row['domain_capped'] = flags.domain_capped
        row['status'] = f'solver_error:{exc}'
        return row

    D_rem_coarse = result.D_rem

    # ── Inline refinement (shallow band) ─────────────────────────────────────
    refined = False
    D_rem   = D_rem_coarse
    t_max_remelt = result.time_to_max_remelt * grid.dt
    onset_missed = (D_rem_coarse == 0.0
                     and np.isfinite(lam_neumann) and lam_neumann > 0)
    if (P.shallow_lo < D_rem_coarse < P.shallow_hi) or onset_missed:
        try:
            grid_f = build_grid(Z_total=Z_total, dz=P.dz_fine,
                                t_total=t_total, r=r)
            snap_f = list(np.linspace(0, grid_f.nt_max - 1, _N_SNAPS,
                                      dtype=np.int64))
            result_f = fd_solve(adim, grid_f.z, grid_f.interface_idx,
                                grid_f.dt, grid_f.nt_max, snap_f,
                                model='pure_metal')
            D_rem   = result_f.D_rem
            t_max_remelt = result_f.time_to_max_remelt * grid_f.dt
            refined = True
        except Exception:
            pass   # fall back to coarse


    row = _blank_row()
    row.update(base)
    row.update(physics)
    row['lam_neumann']        = lam_neumann
    row['t_total']            = t_total
    row['Z_total']            = Z_total
    row['time_capped']        = flags.time_capped
    row['domain_capped']      = flags.domain_capped
    row['D_rem_coarse']       = D_rem_coarse
    row['D_rem']              = D_rem
    row['t_at_max_remelt']    = t_max_remelt
    row['refined']            = refined
    row['status']             = 'ok'
    row['config_fingerprint'] = P.fingerprint()
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LHS generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_lhs(n_lhs, seed=None):
    """
    Generate n_lhs points in 3D LHS space (E, C, r).

    Returns np.ndarray shape (n_lhs, 3).
    """
    if seed is None:
        seed = P.rng_seed

    axes   = list(LHS_AXES.keys())   # ['E', 'C', 'r']
    n_axes = len(axes)
    def _make_lhs_sampler(n_axes, seed):
        try:
            return qmc.LatinHypercube(d=n_axes, rng=seed)
        except TypeError:
            return qmc.LatinHypercube(d=n_axes, seed=seed)  # type: ignore[call-arg]
    sampler = _make_lhs_sampler(n_axes, seed)
    unit    = sampler.random(n=n_lhs)

    out = np.empty_like(unit)
    for j, ax in enumerate(axes):
        scale, lo, hi = LHS_AXES[ax]
        if scale == 'log':
            out[:, j] = np.exp(unit[:, j] * (np.log(hi) - np.log(lo)) + np.log(lo))
        else:
            out[:, j] = unit[:, j] * (hi - lo) + lo

    return out   # columns: E, C, r

# ── new function, next to generate_lhs() ───────────────────────────────────
def generate_targeted_grid():
    """
    Deterministic (E, C, r) grid for the linearity-criterion validation sweep.
    Not LHS: r is log-spaced within each of 3 fixed (E, C) pairs so the
    r0(Ste_cold) crossing can be resolved with real resolution, and the
    "E, C cancel" claim can be checked directly (3 distinct C/E ratios).
    Returns array shape (len(TARGETED_EC_PAIRS) * TARGETED_N_R, 3): E, C, r.
    """
    r_vals = np.logspace(np.log10(TARGETED_R_RANGE[0]),
                          np.log10(TARGETED_R_RANGE[1]), TARGETED_N_R)
    rows = [(E, C, r) for (E, C) in TARGETED_EC_PAIRS for r in r_vals]
    return np.array(rows, dtype=float)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Job list assembly
# ═══════════════════════════════════════════════════════════════════════════════

# ── build_job_list: accept an override grid (minimal signature change) ─────
def build_job_list(lhs_pts, ste_cold_op, ste_hot_op, compute_neumann=False):
    """
    ste_cold_op, ste_hot_op : the operating grids to use for this sweep
                 (e.g. STE_COLD_OP/STE_HOT_OP for the canonical sweep,
                 STE_COLD_OP_REAL/STE_HOT_OP_REAL for --real_materials,
                 or STE_COLD_OP/STE_HOT_OP_TARGETED for --targeted).
                 Always passed explicitly by run_sweep -- no module-level
                 fallback here, so there is only one place a sweep's op
                 grid is chosen.
    """
    op_points = [(float(sc), float(sh))
                 for sc in ste_cold_op for sh in ste_hot_op]
    jobs = []
    for i, (E, C, r) in enumerate(lhs_pts):
        E, C, r = float(E), float(C), float(r)
        for op_id, (sc, sh) in enumerate(op_points):
            jobs.append((i, op_id, E, C, r, sc, sh, compute_neumann))
    return jobs

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Resume logic
# ═══════════════════════════════════════════════════════════════════════════════

def load_done_keys(csv_path):
    """Return set of (lhs_id, op_id) pairs already in the checkpoint CSV."""
    p = Path(csv_path)
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p, usecols=['lhs_id', 'op_id'])
        return set(zip(df['lhs_id'].tolist(), df['op_id'].tolist()))
    except Exception:
        return set()


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Main sweep loop
# ═══════════════════════════════════════════════════════════════════════════════

# ── run_sweep: add a targeted switch ────────────────────────────────────────
def run_sweep(n_lhs, workers, output_csv, resume, dry_run=False, seed=None,
              compute_neumann=False, targeted=False, real_materials = False):
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ste_hot_op = STE_HOT_OP_TARGETED if targeted else STE_HOT_OP
    n_ops = len(STE_COLD_OP) * len(ste_hot_op)
    print(f"\n{'='*64}")
    print(f"  Pure-metal dimensionless sweep{'  [TARGETED r0 validation]' if targeted else ''}")
    print(f"  Config fingerprint: {P.fingerprint()}")
    print(f"  op-grid = {len(STE_COLD_OP)}x{len(ste_hot_op)} = {n_ops}")
    print(f"  Workers: {workers}   |  output: {out_path}")
    print(f"{'='*64}\n")

    if targeted:
        lhs_pts = generate_targeted_grid()
        print(f"  Targeted grid: {len(TARGETED_EC_PAIRS)} (E,C) pairs x "
              f"{TARGETED_N_R} r-values = {len(lhs_pts)} materials")
    else:
        if real_materials:
          lhs_pts   = generate_lhs_real(n_lhs, seed=seed)
          ste_cold_op, ste_hot_op = STE_COLD_OP_REAL, STE_HOT_OP_REAL
        else:
          lhs_pts   = generate_lhs(n_lhs, seed=seed)
          ste_cold_op, ste_hot_op = STE_COLD_OP, STE_HOT_OP

    print(f"  Total planned runs: {len(lhs_pts) * n_ops}")

    all_jobs = build_job_list(lhs_pts, ste_cold_op, ste_hot_op,
            compute_neumann=compute_neumann)
    print(f"  Generated {len(all_jobs)} jobs"
          + ("  [+neumann]" if compute_neumann else ""))


    done_keys = load_done_keys(out_path) if resume else set()
    if done_keys:
        print(f"  Resuming: {len(done_keys)} rows already in {out_path}")

    jobs = [j for j in all_jobs if (j[0], j[1]) not in done_keys]
    print(f"  Jobs to run: {len(jobs)}")

    if dry_run:
        print("  [DRY RUN] First 3 jobs:")
        for jb in jobs[:3]:
            print(f"    {jb}")
        return

    if not jobs:
        print("  Nothing to do — all cases already complete.")
        return

    first_write = not (resume and out_path.exists() and done_keys)
    buffer      = []
    n_done = n_skip = n_err = 0
    t0 = time.perf_counter()

    def _flush(buf, first):
        df_buf = pd.DataFrame(buf)
        df_buf.to_csv(out_path, index=False,
                      mode='w' if first else 'a',
                      header=first)
        return False

    if workers == 1:
        for job in jobs:
            row = run_one(job)
            buffer.append(row)
            n_done += 1
            status_str = str(row['status'])
            if 'skipped' in status_str:
                n_skip += 1
            elif status_str != 'ok':
                n_err += 1

            if n_done % P.checkpoint_every == 0 or n_done == len(jobs):
                first_write = _flush(buffer, first_write)
                buffer = []
                elapsed = time.perf_counter() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta  = (len(jobs) - n_done) / rate if rate > 0 else float('nan')
                print(f"  [{n_done:>6}/{len(jobs)}]  "
                      f"ok={n_done-n_skip-n_err}  skip={n_skip}  err={n_err}  "
                      f"{rate:.1f} cases/s  ETA {eta/60:.1f} min")
    else:
        chunksize = max(1, min(50, len(jobs) // (workers * 4)))
        with mp.Pool(processes=workers) as pool:
            for row in pool.imap_unordered(run_one, jobs, chunksize=chunksize):
                buffer.append(row)
                n_done += 1
                status_str = str(row['status'])
                if 'skipped' in status_str:
                    n_skip += 1
                elif status_str != 'ok':
                    n_err += 1

                if n_done % P.checkpoint_every == 0 or n_done == len(jobs):
                    first_write = _flush(buffer, first_write)
                    buffer = []
                    elapsed = time.perf_counter() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta  = (len(jobs) - n_done) / rate if rate > 0 else float('nan')
                    print(f"  [{n_done:>6}/{len(jobs)}]  "
                          f"ok={n_done-n_skip-n_err}  skip={n_skip}  err={n_err}  "
                          f"{rate:.1f} cases/s  ETA {eta/60:.1f} min")

    elapsed = time.perf_counter() - t0
    print(f"\n  Done. {n_done} cases in {elapsed:.1f}s  ({n_done/elapsed:.1f} cases/s)")
    print(f"  ok={n_done-n_skip-n_err}  skipped={n_skip}  errors={n_err}")
    print(f"  Output: {out_path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse():
    p = argparse.ArgumentParser(
        description='Pure-metal sweep driver: LHS × op-grid with inline refinement')
    p.add_argument('--n_lhs',   type=int, default=P.n_lhs,
                   help=f'LHS material points (default {P.n_lhs})')
    p.add_argument('--real_materials', action='store_true',
                     help='Sample on the real-material manifold '
                          '(k_ratio, rho_ratio, cp_ratio) with a '
                          'melt-biased operating grid, instead of the '
                          'default (E, C, r) box sweep.')
    p.add_argument('--workers', type=int, default=1,
                   help='Parallel workers (default 1)')
    p.add_argument('--output',  type=str, default=str(DEFAULT_OUTPUT),
                   help=f'Output CSV (default {DEFAULT_OUTPUT})')
    p.add_argument('--resume',  action='store_true',
                   help='Skip rows already in output CSV')
    p.add_argument('--dry_run', action='store_true',
                   help='Print job list only, do not run')
    p.add_argument('--seed',    type=int, default=None,
                   help=f'RNG seed (default PARAMS.rng_seed={P.rng_seed})')
    p.add_argument('--neumann', action='store_true',
                   help='Compute Neumann λ per case; improves domain sizing and adds lam_neumann column')
    # ── CLI: one new flag ────────────────────────────────────────────────────────
    p.add_argument('--targeted', action='store_true',
                help='Run the r0-crossing validation sweep (dense Ste_hot grid, '
                    '3 fixed (E,C) pairs x 18 log-spaced r) instead of the LHS sweep')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    run_sweep(
        n_lhs          = args.n_lhs,
        workers        = args.workers,
        output_csv     = args.output,
        resume         = args.resume,
        dry_run        = args.dry_run,
        seed           = args.seed,
        compute_neumann= args.neumann,
        targeted       = args.targeted,
        real_materials= args.real_materials,
    )