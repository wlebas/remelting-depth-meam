"""
dimensionless_sweep_scheil.py
=============================
Staged sweep driver for the Scheil remelting correlation.

Pipeline per case
-----------------
  1. groups_scheil()   — convert 7 dimensionless groups to kernel inputs
  2. size_run()        — predict t_total, Z_total from the groups
  3. build_grid()      — construct grid at dz_coarse
  4. fd_solve()        — run Scheil FD solver
  5. (optional)        — if D_rem in [no_melt_threshold, resolved_threshold),
                         re-run at dz_fine (inline refinement)
  6. checkpoint        — flush to CSV every PARAMS.checkpoint_every rows

Sweep design
------------
  LHS axes (N_LHS points):
    E            — effusivity ratio        ∈ [0.40, 3.00]  (linear)
    C            — cp ratio                ∈ [0.60, 2.00]  (linear)
    r            — diffusivity ratio       ∈ [0.20, 4.00]  (log)
    mushy_frac   — θ_sol/θ_liq            ∈ [0.50, 0.99]  (linear)
    k_p          — partition coefficient  ∈ [0.10, 0.97]  (linear)

  Operating-point grid (tiled over each LHS point):
    ste_cold × ste_hot   PARAMS.ste_cold_op × PARAMS.ste_hot_op  (6×5=30)

  Total planned runs:  N_LHS × 30

Guard rails (from PARAMS)
-------------------------
  theta_melt_min/max, k_ratio_min/max, rho_ratio_min/max
  Cases outside bounds are skipped (status='skipped_guard')

Output columns
--------------
  lhs_id, op_id                    — provenance indices
  E, C, r, mushy_frac, k_p        — LHS groups
  ste_cold, ste_hot                — operating point
  theta_liquidus, theta_solidus    — derived thermal positions
  Ste_m, inv_Ste                   — Stefan numbers
  k_ratio, rho_ratio, cp_ratio     — property ratios (derived, for audit)
  t_total, Z_total                 — sizing decisions
  time_capped, domain_capped       — sizing flags
  D_rem_coarse, D_sol_coarse       — coarse-grid results
  D_rem, D_sol                     — final results (refined if in shallow band)
  mushy_width_at_max_remelt        — |D_sol − D_rem| at peak
  refined                          — bool: was dz_fine re-run used?
  status                           — 'ok' | 'skipped_guard' | 'solver_error'
  config_fingerprint               — PARAMS.fingerprint() for traceability

Usage
-----
  python dimensionless_sweep_scheil.py --n_lhs 1200 --workers 8
  python dimensionless_sweep_scheil.py --n_lhs 1200 --workers 8 --resume results_scheil/progress.csv
  python dimensionless_sweep_scheil.py --n_lhs 1200 --workers 1 --dry_run
  python dimensionless_sweep_scheil.py --n_lhs 1200 --workers 8 --neumann

--neumann flag
--------------
  When set, solves the mushy-zone similarity problem (solve_mushy_similarity)
  before each FD run, returning eigenvalues (λ₁, λ₂):
    λ₁ — liquidus front eigenvalue (analogous to pure-metal λ)
    λ₂ — solidus front eigenvalue  (deeper than λ₁)
  These are used to:
    1. Improve domain sizing via D_est = 2λ₂√(rt) in size_run.
    2. Store lam_liq and lam_sol as output columns.
    3. Provide a per-case analytical cross-check against the FD result.
  Cost: ~0.5 s per case (2D 30×30 grid scan + fsolve polish).
  Defaults to False; enable for production runs where column completeness matters.

"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ── Local imports ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config_pipeline import CONVENTIONS as CV, PARAMS as P
from grid import build_grid, size_run
from solvers import fd_solve
from analytical import solve_mushy_similarity

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Sweep configuration
# ═══════════════════════════════════════════════════════════════════════════════

# LHS bounds for the 5 material-class axes
# (ste_cold and ste_hot are the operating-point grid, not LHS axes here)
LHS_AXES = {
    'E':          ('linear', 0.40, 3.00),
    'C':          ('linear', 0.60, 2.00),
    'r':          ('log',    0.20, 4.00),
    'mushy_frac': ('linear', 0.50, 0.99),
    'k_p':        ('linear', 0.10, 0.97),
}

# Operating-point grid (same as pure-metal sweep for consistency)
STE_COLD_OP = P.ste_cold_op
STE_HOT_OP  = P.ste_hot_op

# Output CSV path
DEFAULT_OUTPUT = Path('results_scheil') / 'progress.csv'

# ═══════════════════════════════════════════════════════════════════════════════
# Canonical row schema
# ═══════════════════════════════════════════════════════════════════════════════
# BUG FIX (see CHANGELOG below): run_one()'s four return paths (guard-skip,
# sizing-error, solver-error, success) previously built their row dicts
# independently, with different key sets — the skip/error paths spread
# **adim (which includes 'theta_liquidus', 'theta_solidus', 'Ste_m',
# 'inv_Ste', 'k_ratio', 'rho_ratio', 'cp_ratio', 'B', 'theta_contact', but
# never 'config_fingerprint' or several result columns), while the success
# path used explicit keys with a different set entirely. If a checkpoint
# batch (P.checkpoint_every rows) happened to land entirely on one path,
# pd.DataFrame(buf) built a narrower-than-header DataFrame for that batch,
# and writing it with header=False silently misaligned every column for the
# whole batch (this exact failure mode was confirmed in the pure-metal sweep's
# results_pure.csv; the Scheil sweep has the identical structural bug, even
# though it may not have triggered yet in any given run).
#
# Fix: every return path now builds from this single canonical key list via
# _blank_row(), so every row dict — regardless of which branch produced it —
# has the exact same keys, in the exact same order, with np.nan/False
# placeholders for whatever that branch doesn't compute.
ROW_SCHEMA = [
    'lhs_id', 'op_id', 'E', 'C', 'r', 'mushy_frac', 'k_p',
    'ste_cold', 'ste_hot',
    'theta_liquidus', 'theta_solidus', 'Ste_m', 'inv_Ste',
    'k_ratio', 'rho_ratio', 'cp_ratio', 'theta_contact',
    'lam_liq', 'lam_sol', 'mushy_status', 'theta0_surface',
    't_total', 'Z_total', 'time_capped', 'domain_capped',
    'D_rem_coarse', 'D_sol_coarse', 'D_rem', 'D_sol',
    'mushy_width_at_max_remelt', 't_at_max_remelt', 't_at_max_solidus',
    'refined', 'status', 'config_fingerprint',
]


def _blank_row() -> Dict[str, Any]:
    """
    A dict with every ROW_SCHEMA key present, defaulted to NaN (or False for
    the boolean flags). Callers overwrite only the keys they actually
    compute — every return path ends up with the full, identical key set.

    Typed Dict[str, Any] (not Dict[str, float]) because 'status' is always
    a str by the time any caller reads it, even though it starts as NaN here.
    """
    row: Dict[str, Any] = {k: np.nan for k in ROW_SCHEMA}
    row['time_capped'] = False
    row['domain_capped'] = False
    row['refined'] = False
    row['mushy_status'] = 'not_computed'
    return row

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Group algebra
# ═══════════════════════════════════════════════════════════════════════════════

def groups_scheil(E, C, r, ste_cold, ste_hot, k_p, mushy_frac):
    """
    Convert the 7 dimensionless groups to Scheil kernel inputs.

    Algebra
    -------
    B = (ρ cp)_l / (ρ cp)_s = E / √r          [from E² = k_ratio·B, r = k_ratio/B]
    cp_ratio    = C
    rho_ratio   = B / C
    k_ratio     = r · B = E√r
    theta_liq   = ste_cold / (ste_cold + ste_hot/C)   [operating condition]
    theta_sol   = theta_liq · mushy_frac
    Ste_m       = ½(1+C) · inv_Ste · (theta_liq − theta_sol)   [mean-cp convention]

    Parameters
    ----------
    E, C, r, ste_cold, ste_hot, k_p, mushy_frac : float

    Returns
    -------
    dict with all kernel inputs + audit quantities, or None if unphysical.
    """
    # ── Derived property ratios ───────────────────────────────────────────────
    B         = E / np.sqrt(r)
    cp_ratio  = C
    rho_ratio = B / C
    k_ratio   = r * B                 # = E√r

    # ── Operating condition ───────────────────────────────────────────────────
    inv_Ste        = ste_cold + ste_hot / C
    theta_liquidus = ste_cold / inv_Ste
    theta_solidus  = theta_liquidus * mushy_frac

    # ── Mushy Stefan number ───────────────────────────────────────────────────
    dtheta_f = theta_liquidus - theta_solidus   # = theta_liq · (1 - mushy_frac)
    Ste_m    = 0.5 * (1.0 + C) * inv_Ste * dtheta_f

    return {
        # ── Kernel inputs ─────────────────────────────────────────────────────
        'theta_liquidus': theta_liquidus,
        'theta_solidus':  theta_solidus,
        'Ste_m':          Ste_m,
        'k_p':            k_p,
        'k_ratio':        k_ratio,
        'rho_ratio':      rho_ratio,
        'cp_ratio':       cp_ratio,
        # ── Audit quantities ──────────────────────────────────────────────────
        'inv_Ste':        1.0 / (ste_cold + ste_hot / C),   # L/(cp_s·dT_ref), matches pure convention
        'B':              B,
        'theta_contact':  E / (1.0 + E),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Guard-rail checks
# ═══════════════════════════════════════════════════════════════════════════════

def check_guards(adim):
    """
    Return a reason string if the case should be skipped, or None if OK.
    """
    theta_liq = adim['theta_liquidus']
    theta_sol = adim['theta_solidus']
    k_rat     = adim['k_ratio']
    rho_rat   = adim['rho_ratio']

    # theta_melt is theta_liquidus for Scheil onset criterion
    theta_melt = theta_liq
    if not (P.theta_melt_min < theta_melt < P.theta_melt_max):
        return f'theta_melt={theta_melt:.4f} outside [{P.theta_melt_min}, {P.theta_melt_max}]'
    if not (P.k_ratio_min <= k_rat <= P.k_ratio_max):
        return f'k_ratio={k_rat:.4f} outside [{P.k_ratio_min}, {P.k_ratio_max}]'
    if not (P.rho_ratio_min <= rho_rat <= P.rho_ratio_max):
        return f'rho_ratio={rho_rat:.4f} outside [{P.rho_ratio_min}, {P.rho_ratio_max}]'
    if theta_sol <= 0.0 or theta_sol >= theta_liq:
        return f'theta_solidus={theta_sol:.4f} unphysical'
    if adim['Ste_m'] <= 0.0:
        return f'Ste_m={adim["Ste_m"]:.4f} unphysical'

    return None   # all clear


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. Neumann / mushy similarity eigenvalues (optional --neumann flag)
# ═══════════════════════════════════════════════════════════════════════════════

def neumann_groups_scheil(E, C, r, ste_cold, ste_hot, k_p, mushy_frac):
    """
    Build the dict that solve_mushy_similarity() expects, derived from sweep groups.

    Required fields: r, r_eff, Ste_m, theta_deposit(=1), theta_substrate(=0),
                     theta_liquidus, theta_solidus, eff_LM, eff_SM.

    Mushy mean properties are computed as arithmetic means of liquid and solid:
      k_m/k_s     = (1 + k_ratio) / 2
      rho_m/rho_s = (1 + rho_ratio) / 2
      cp_m/cp_s   = (1 + C) / 2
      alpha_m     = k_m / (rho_m · cp_m)  [relative to alpha_s = 1]
      r_eff       = alpha_m · Ste_m / (1 + Ste_m)   [apparent heat capacity]
      eff_m/eff_s = sqrt(rho_m_ratio · cp_m_ratio · k_m_ratio)
      eff_LM      = E / (eff_m/eff_s)    [eff_l / eff_m]
      eff_SM      = 1 / (eff_m/eff_s)    [eff_s / eff_m]

    Returns (lam_liq, lam_sol, mushy_status, theta0_surface).
    lam_liq may be 0.0 in Regime 2 (liquidus pinned, D_rem -> 0); lam_sol > 0.
    Either eigenvalue may be nan if no solution exists.
    """
    adim = groups_scheil(E, C, r, ste_cold, ste_hot, k_p, mushy_frac)
    B         = adim['B']
    rho_ratio = adim['rho_ratio']
    k_ratio   = adim['k_ratio']
    Ste_m     = adim['Ste_m']
    theta_liq = adim['theta_liquidus']
    theta_sol = adim['theta_solidus']

    # Arithmetic mean ratios (relative to solid)
    k_m_ratio   = 0.5 * (1.0 + k_ratio)
    rho_m_ratio = 0.5 * (1.0 + rho_ratio)
    cp_m_ratio  = 0.5 * (1.0 + C)

    alpha_m_ratio = k_m_ratio / (rho_m_ratio * cp_m_ratio)
    r_eff = alpha_m_ratio * Ste_m / (1.0 + Ste_m) if (1.0 + Ste_m) > 0 else np.nan

    eff_m_ratio = np.sqrt(rho_m_ratio * cp_m_ratio * k_m_ratio)
    eff_LM = E / eff_m_ratio if eff_m_ratio > 0 else np.nan
    eff_SM = 1.0 / eff_m_ratio if eff_m_ratio > 0 else np.nan

    groups = {
        'r'               : r,
        'r_eff'           : r_eff,
        'Ste_m'           : Ste_m,
        'theta_deposit'   : 1.0,
        'theta_substrate' : 0.0,
        'theta_liquidus'  : theta_liq,
        'theta_solidus'   : theta_sol,
        'eff_LM'          : eff_LM,
        'eff_SM'          : eff_SM,
    }

    if not all(np.isfinite(v) for v in groups.values()):
        return np.nan, np.nan, 'groups_non_finite', np.nan

    try:
        lam1, lam2 = solve_mushy_similarity(groups)
        lam1 = float(lam1) if np.isfinite(lam1) else np.nan
        lam2 = float(lam2) if np.isfinite(lam2) and lam2 > 0 else np.nan   # lam2 ordering still enforced
        return lam1, lam2, 'converged', np.nan
    except Exception:
        return np.nan, np.nan, 'solver_exception', np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Single-case runner
# ═══════════════════════════════════════════════════════════════════════════════

# Snapshot count (kept low — just for diagnostics; sweep doesn't use histories)
_N_SNAPS = 8

def run_one(args) -> Dict[str, Any]:
    """
    Run a single (LHS point × operating point) case.

    Parameters (packed as a single tuple for multiprocessing)
    ----------------------------------------------------------
    lhs_id       : int
    op_id        : int
    E, C, r, mushy_frac, k_p : float  — LHS groups
    ste_cold, ste_hot          : float  — operating point
    compute_neumann            : bool   — if True, solve for λ₁, λ₂

    Returns
    -------
    dict  — flat row ready for the output CSV
    """
    (lhs_id, op_id,
     E, C, r, mushy_frac, k_p,
     ste_cold, ste_hot, compute_neumann) = args

    base = {
        'lhs_id': lhs_id, 'op_id': op_id,
        'E': E, 'C': C, 'r': r, 'mushy_frac': mushy_frac, 'k_p': k_p,
        'ste_cold': ste_cold, 'ste_hot': ste_hot,
    }

    # ── Group algebra ─────────────────────────────────────────────────────────
    adim = groups_scheil(E, C, r, ste_cold, ste_hot, k_p, mushy_frac)
    # adim's physics keys are shared by every return path below.
    physics = {
        'theta_liquidus': adim['theta_liquidus'],
        'theta_solidus':  adim['theta_solidus'],
        'Ste_m':          adim['Ste_m'],
        'inv_Ste':        adim['inv_Ste'],
        'k_ratio':        adim['k_ratio'],
        'rho_ratio':      adim['rho_ratio'],
        'cp_ratio':       adim['cp_ratio'],
        'theta_contact':  adim['theta_contact'],
    }

    # ── Guard rails ───────────────────────────────────────────────────────────
    reason = check_guards(adim)
    if reason:
        row = _blank_row()
        row.update(base)
        row.update(physics)
        row['status'] = f'skipped_guard:{reason}'
        return row

    # ── Mushy similarity eigenvalues (optional) ───────────────────────────────
    lam_liq, lam_sol = np.nan, np.nan
    mushy_status, theta0_surface = 'not_computed', np.nan
    if compute_neumann:
        lam_liq, lam_sol, mushy_status, theta0_surface = neumann_groups_scheil(
            E, C, r, ste_cold, ste_hot, k_p, mushy_frac)

    # ── A-priori sizing (use lam_liq as depth estimate for domain) ───────────
    try:
        t_total, Z_total, flags = size_run(
            E, C, r, ste_cold, ste_hot,
            lam_neumann=lam_liq if (np.isfinite(lam_liq) and lam_liq > 0) else None
        )
    except Exception as exc:
        row = _blank_row()
        row.update(base)
        row.update(physics)
        row['lam_liq'] = lam_liq
        row['lam_sol'] = lam_sol
        row['mushy_status'] = mushy_status
        row['theta0_surface'] = theta0_surface
        row['status'] = f'sizing_error:{exc}'
        return row

    # ── Coarse-grid solve ─────────────────────────────────────────────────────
    try:
        grid = build_grid(Z_total=Z_total, dz=P.dz_coarse, t_total=t_total, r=r)
        snap_idx = list(np.linspace(0, grid.nt_max - 1, _N_SNAPS, dtype=np.int64))
        result = fd_solve(adim, grid.z, grid.interface_idx, grid.dt,
                          grid.nt_max, snap_idx, model='scheil')
    except Exception as exc:
        row = _blank_row()
        row.update(base)
        row.update(physics)
        row['lam_liq'] = lam_liq
        row['lam_sol'] = lam_sol
        row['mushy_status'] = mushy_status
        row['theta0_surface'] = theta0_surface
        row['t_total'] = t_total
        row['Z_total'] = Z_total
        row['time_capped'] = flags.time_capped
        row['domain_capped'] = flags.domain_capped
        row['status'] = f'solver_error:{exc}'
        return row

    D_rem_coarse = result.D_rem
    D_sol_coarse = result.D_sol

    # ── Inline refinement (shallow band) ─────────────────────────────────────
    refined  = False
    D_rem    = D_rem_coarse
    D_sol    = D_sol_coarse
    mushy_w  = result.mushy_width_at_max_remelt
    t_max_remelt  = result.time_to_max_remelt  * grid.dt
    t_max_solidus = result.time_to_max_solidus * grid.dt
    onset_missed = (D_rem_coarse == 0.0
                     and np.isfinite(lam_liq) and lam_liq > 0)
    if (P.shallow_lo < D_rem_coarse < P.shallow_hi) or onset_missed:
        try:
            grid_f = build_grid(Z_total=Z_total, dz=P.dz_fine, t_total=t_total, r=r)
            snap_f = list(np.linspace(0, grid_f.nt_max - 1, _N_SNAPS, dtype=np.int64))
            result_f = fd_solve(adim, grid_f.z, grid_f.interface_idx, grid_f.dt,
                                grid_f.nt_max, snap_f, model='scheil')
            D_rem   = result_f.D_rem
            D_sol   = result_f.D_sol
            mushy_w = result_f.mushy_width_at_max_remelt
            t_max_remelt  = result_f.time_to_max_remelt  * grid_f.dt
            t_max_solidus = result_f.time_to_max_solidus * grid_f.dt
            refined = True
        except Exception:
            pass   # fall back to coarse

    row = _blank_row()
    row.update(base)
    row.update(physics)
    row['lam_liq']                    = lam_liq
    row['lam_sol']                    = lam_sol
    row['mushy_status']               = mushy_status
    row['theta0_surface']             = theta0_surface
    row['t_total']                    = t_total
    row['Z_total']                    = Z_total
    row['time_capped']                = flags.time_capped
    row['domain_capped']              = flags.domain_capped
    row['D_rem_coarse']               = D_rem_coarse
    row['D_sol_coarse']               = D_sol_coarse
    row['D_rem']                      = D_rem
    row['D_sol']                      = D_sol
    row['mushy_width_at_max_remelt']  = mushy_w
    row['t_at_max_remelt']            = t_max_remelt
    row['t_at_max_solidus']           = t_max_solidus
    row['refined']                    = refined
    row['status']                     = 'ok'
    row['config_fingerprint']         = P.fingerprint()
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LHS generation
# ═══════════════════════════════════════════════════════════════════════════════

def _make_lhs_sampler(n_axes, seed):
    """
    Construct qmc.LatinHypercube across scipy versions.

    scipy >= 1.15 deprecated the `seed` kwarg in favour of `rng`; in 1.17.x
    `LatinHypercube.__init__` no longer accepts `seed` at all. Try `rng`
    first, fall back to `seed` so this runs on either generation.
    """
    try:
        return qmc.LatinHypercube(d=n_axes, rng=seed)
    except TypeError:
        return qmc.LatinHypercube(d=n_axes, seed=seed)  # type: ignore[call-arg]  # older scipy stub only


def generate_lhs(n_lhs, seed=None):
    """
    Generate N_LHS points in 5D LHS space (E, C, r, mushy_frac, k_p).

    Returns
    -------
    np.ndarray, shape (n_lhs, 5), columns ordered as in LHS_AXES.
    """
    if seed is None:
        seed = P.rng_seed

    axes   = list(LHS_AXES.keys())
    n_axes = len(axes)

    sampler = _make_lhs_sampler(n_axes, seed)
    unit    = sampler.random(n=n_lhs)  # ∈ [0, 1]^n_axes

    out = np.empty_like(unit)
    for j, ax in enumerate(axes):
        scale, lo, hi = LHS_AXES[ax]
        if scale == 'log':
            out[:, j] = np.exp(unit[:, j] * (np.log(hi) - np.log(lo)) + np.log(lo))
        else:
            out[:, j] = unit[:, j] * (hi - lo) + lo

    return out   # columns: E, C, r, mushy_frac, k_p


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Job list assembly
# ═══════════════════════════════════════════════════════════════════════════════

def build_job_list(lhs_pts, compute_neumann=False):
    """
    Build the full list of (lhs_id, op_id, E, C, r, mushy_frac, k_p,
    ste_cold, ste_hot, compute_neumann) tuples.

    op_id ∈ [0, 29] — position in the operating-point grid.
    (lhs_id, op_id) is the composite resume key.
    """
    jobs = []
    op_points = [(float(sc), float(sh))
                 for sc in STE_COLD_OP for sh in STE_HOT_OP]
    for i, (E, C, r, mf, kp) in enumerate(lhs_pts):
        E, C, r, mf, kp = float(E), float(C), float(r), float(mf), float(kp)
        for op_id, (sc, sh) in enumerate(op_points):
            jobs.append((i, op_id, E, C, r, mf, kp, sc, sh, compute_neumann))
    return jobs


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Resume logic
# ═══════════════════════════════════════════════════════════════════════════════

def load_done_keys(csv_path):
    """
    Return a set of (lhs_id, op_id) pairs already present in the checkpoint CSV.
    """
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

def run_sweep(n_lhs, workers, output_csv, resume, dry_run=False, seed=None,
              compute_neumann=False):
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Scheil dimensionless sweep")
    print(f"  Config fingerprint: {P.fingerprint()}")
    print(f"  N_LHS = {n_lhs}   |  op-grid = {len(STE_COLD_OP)}×{len(STE_HOT_OP)} = "
          f"{len(STE_COLD_OP)*len(STE_HOT_OP)}")
    print(f"  Total planned runs: {n_lhs * len(STE_COLD_OP) * len(STE_HOT_OP)}")
    print(f"  Workers: {workers}   |  output: {out_path}")
    print(f"{'='*64}\n")

    # ── Generate LHS and job list ─────────────────────────────────────────────
    lhs_pts  = generate_lhs(n_lhs, seed=seed)
    all_jobs = build_job_list(lhs_pts, compute_neumann=compute_neumann)
    print(f"  Generated {len(all_jobs)} jobs"
          + ("  [+neumann]" if compute_neumann else ""))

    # ── Filter already-done jobs (resume mode) ────────────────────────────────
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

    # ── Sweep ─────────────────────────────────────────────────────────────────
    first_write = not (resume and out_path.exists() and done_keys)
    buffer      = []
    n_done      = 0
    n_skip      = 0
    n_err       = 0
    t0          = time.perf_counter()

    def _flush(buf, first):
        df_buf = pd.DataFrame(buf)
        if first:
            df_buf.to_csv(out_path, index=False, mode='w')
        else:
            df_buf.to_csv(out_path, index=False, mode='a', header=False)
        return False   # next flush is not the first

    if workers == 1:
        # Serial path (easier debugging)
        for job in jobs:
            row = run_one(job)
            buffer.append(row)
            n_done += 1
            status_str = str(row['status'])
            if status_str == 'ok':
                pass
            elif status_str.startswith('skipped'):
                n_skip += 1
            else:
                n_err += 1

            if n_done % P.checkpoint_every == 0 or n_done == len(jobs):
                first_write = _flush(buffer, first_write)
                buffer = []
                elapsed = time.perf_counter() - t0
                rate    = n_done / elapsed if elapsed > 0 else 0
                eta     = (len(jobs) - n_done) / rate if rate > 0 else float('nan')
                print(f"  [{n_done:>6}/{len(jobs)}]  "
                      f"ok={n_done-n_skip-n_err}  skip={n_skip}  err={n_err}  "
                      f"{rate:.1f} cases/s  ETA {eta/60:.1f} min")
    else:
        # Parallel path
        chunksize = max(1, min(50, len(jobs) // (workers * 4)))
        with mp.Pool(processes=workers) as pool:
            for row in pool.imap_unordered(run_one, jobs, chunksize=chunksize):
                buffer.append(row)
                n_done += 1
                status_str = str(row['status'])
                if status_str != 'ok':
                    if 'skipped' in status_str:
                        n_skip += 1
                    else:
                        n_err += 1

                if n_done % P.checkpoint_every == 0 or n_done == len(jobs):
                    first_write = _flush(buffer, first_write)
                    buffer = []
                    elapsed = time.perf_counter() - t0
                    rate    = n_done / elapsed if elapsed > 0 else 0
                    eta     = (len(jobs) - n_done) / rate if rate > 0 else float('nan')
                    print(f"  [{n_done:>6}/{len(jobs)}]  "
                          f"ok={n_done-n_skip-n_err}  skip={n_skip}  err={n_err}  "
                          f"{rate:.1f} cases/s  ETA {eta/60:.1f} min")

    elapsed = time.perf_counter() - t0
    print(f"\n  Done. {n_done} cases in {elapsed:.1f}s  "
          f"({n_done/elapsed:.1f} cases/s)")
    print(f"  ok={n_done-n_skip-n_err}  skipped={n_skip}  errors={n_err}")
    print(f"  Output: {out_path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse():
    p = argparse.ArgumentParser(
        description='Scheil sweep driver: LHS × op-grid with inline refinement')
    p.add_argument('--n_lhs',   type=int,   default=P.n_lhs,
                   help=f'Number of LHS material points (default {P.n_lhs})')
    p.add_argument('--workers', type=int,   default=1,
                   help='Number of parallel workers (default 1)')
    p.add_argument('--output',  type=str,   default=str(DEFAULT_OUTPUT),
                   help=f'Output CSV path (default {DEFAULT_OUTPUT})')
    p.add_argument('--resume',  action='store_true',
                   help='Skip rows already present in output CSV')
    p.add_argument('--dry_run', action='store_true',
                   help='Generate job list but do not run any simulations')
    p.add_argument('--seed',    type=int,   default=None,
                   help=f'RNG seed for LHS (default PARAMS.rng_seed={P.rng_seed})')
    p.add_argument('--neumann', action='store_true',
                   help='Compute mushy similarity eigenvalues (λ_liq, λ_sol) per case; '
                        'improves domain sizing and adds lam_liq/lam_sol columns')
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
    )
