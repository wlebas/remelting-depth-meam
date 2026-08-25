"""
case_study_runner.py
=====================
Run a small, hand-picked set of "case study" simulations -- a few pure-metal
and a few Scheil cases -- and save FULL space- and time-resolved results for
each one, the same way the real-material validation runs are meant to be
saved (df_time, df_space, theta(z,t) snapshot history), rather than just the
single scalar summary row that dimensionless_sweep_pure.py / _scheil.py keep
in their big-sweep progress.csv files.

This is deliberately NOT a sweep. It exists to let you actually look at a
handful of representative cases in detail -- plot the melt-front trajectory,
the temperature profile at the moment of peak remelting, the full θ(z)
snapshot history as a function of time -- which the 30,000/36,000-row sweep
CSVs were never built to retain.

Pipeline per case (identical physics/sizing/refinement logic to the big
sweeps, just not thrown away afterwards):

  1. groups_pure() / groups_scheil()   -- dimensionless groups -> kernel inputs
  2. check_guards()                    -- same physical guard rails as the sweeps
  3. size_run()                        -- a-priori (t_total, Z_total)
  4. build_grid()                      -- coarse grid
  5. fd_solve()                        -- coarse-grid solve (model='pure_metal'/'scheil')
  6. inline refinement                 -- if coarse D_rem falls in the shallow
                                           band, re-run at dz_fine (same band
                                           used by the sweeps), and the FINE
                                           result's full fields are what gets
                                           saved
  7. save everything                   -- df_time, df_space, theta_hist (with
                                           its snapshot times) to per-case CSVs,
                                           plus one cross-case summary CSV

Output layout (under --outdir, default 'case_studies_out/'):

  case_studies_out/
    summary.csv                              <- one row per case (scalars + provenance)
    pure_case01_<label>/
      time_series.csv                        <- df_time  (timestep, t, interface_temp,
                                                            liquidus_front, solidus_front)
      space_profile.csv                      <- df_space (z, T_final, T_at_max_remelt)
      theta_snapshots.csv                    <- long-format: snapshot_idx, t, z, theta
      meta.json                              <- full input groups + sizing + provenance
    pure_case02_<label>/
      ...
    scheil_case01_<label>/
      ...

Usage
-----
    python case_study_runner.py
    python case_study_runner.py --outdir my_case_studies
    python case_study_runner.py --dz_fine 0.001   # override refinement resolution

Editing the case list
----------------------
Cases are defined explicitly in PURE_CASES and SCHEIL_CASES below as plain
dicts -- edit those lists directly to add/remove/change cases. Each entry
needs a short `label` (used in the output folder name) plus the relevant
dimensionless groups. ste_cold/ste_hot are required for both; pure cases
need (E, C, r); Scheil cases additionally need (k_p, mushy_frac).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config_pipeline import CONVENTIONS as CV, PARAMS as P
from grid import build_grid, size_run
from solvers import fd_solve

from dimensionless_sweep_pure import groups_pure, check_guards as check_guards_pure
from dimensionless_sweep_scheil import groups_scheil, check_guards as check_guards_scheil
from materials import AL_SI12, compute_adim_groups
from analytical import solve_stefan_similarity
import dataclasses
import dimensionless_sweep_pure as _dsp
_dsp.P = dataclasses.replace(_dsp.P, rho_ratio_max=10.0, t_cap=10000.0, z_cap=120.0)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Case definitions -- EDIT HERE
# ═══════════════════════════════════════════════════════════════════════════════
# Picked to span the design space the big sweeps cover (E in [0.40,3.00],
# C in [0.60,2.00], r in [0.20,4.00] log, ste_cold in [0.05,2.00] log,
# ste_hot in [0.02,1.50] log; Scheil additionally mushy_frac in [0.50,0.99],
# k_p in [0.10,0.97]) -- one "mid" case, then cases that probe the corners
# known to be physically interesting from the verification plan (deep
# low-r corner, near-eutectic-like high mushy_frac/high k_p, strongly
# partitioning low k_p).

PURE_CASES = [
    # ── critr extension: tests for saturation far beyond the FD-verified 30x onset ──
    dict(label='linear_critr_17', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=50.0),
    dict(label='linear_critr_18', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=70.0),
    dict(label='linear_critr_19', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=95.0),
]
"""
# ── NONLINEAR: r=0.02, Ste_cold=5.0, 1.05x -> 30x onset ──────────────────
    # REQUIRES cap/guard patch below. Far cases are expensive (see note).
    dict(label='nonlinear_01', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=5.2500),
    dict(label='nonlinear_02', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=6.5648),
    dict(label='nonlinear_03', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=8.2089),
    dict(label='nonlinear_04', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=10.2647),
    dict(label='nonlinear_05', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=12.8354),
    dict(label='nonlinear_06', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=16.0498),
    dict(label='nonlinear_07', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=20.0693),
    dict(label='nonlinear_08', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=25.0954),
    dict(label='nonlinear_09', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=31.3802),
    dict(label='nonlinear_10', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=39.2390),
    dict(label='nonlinear_11', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=49.0660),
    dict(label='nonlinear_12', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=61.3540),
    dict(label='nonlinear_13', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=76.7193),
    dict(label='nonlinear_14', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=95.9328),
    dict(label='nonlinear_15', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=119.9580),
    dict(label='nonlinear_16', E=1.0, C=1.0, r=0.02, ste_cold=5.0, ste_hot=150.0000),

    # ── Demonstrates CLEARLY LINEAR behaviour via the critical r* ────────────
    # r=r*=4/(pi-2)^2=3.069288 exactly, Ste_cold=1.0 (not small): m_on/m_inf=1.0000
    dict(label='linear_critr_01', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=1.0500),
    dict(label='linear_critr_02', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=1.3130),
    dict(label='linear_critr_03', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=1.6418),
    dict(label='linear_critr_04', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=2.0529),
    dict(label='linear_critr_05', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=2.5671),
    dict(label='linear_critr_06', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=3.2100),
    dict(label='linear_critr_07', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=4.0139),
    dict(label='linear_critr_08', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=5.0191),
    dict(label='linear_critr_09', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=6.2760),
    dict(label='linear_critr_10', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=7.8478),
    dict(label='linear_critr_11', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=9.8132),
    dict(label='linear_critr_12', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=12.2708),
    dict(label='linear_critr_13', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=15.3439),
    dict(label='linear_critr_14', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=19.1866),
    dict(label='linear_critr_15', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=23.9916),
    dict(label='linear_critr_16', E=1.0, C=1.0, r=3.069288, ste_cold=1.0, ste_hot=30.0000),

    # ── Bonus: CLEARLY LINEAR via small Ste_cold, real AlSi12 material ───────
    # matches the T_sub=515C plot discussed earlier; onset ~655C
    dict(label='linear_alsi12_Tdep700',  E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=0.3449),
    dict(label='linear_alsi12_Tdep800',  E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=0.6348),
    dict(label='linear_alsi12_Tdep900',  E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=0.9246),
    dict(label='linear_alsi12_Tdep1000', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=1.2145),
    dict(label='linear_alsi12_Tdep1200', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=1.7941),
    dict(label='linear_alsi12_Tdep1500', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=2.6637),
    dict(label='linear_alsi12_Tdep1750', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=3.3883),
    dict(label='linear_alsi12_Tdep2000', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=4.1129),
    dict(label='linear_alsi12_Tdep2500', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=5.5621),
    dict(label='linear_alsi12_Tdep3000', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=7.0114),
    dict(label='linear_alsi12_Tdep3500', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=8.4606),
    dict(label='linear_alsi12_Tdep4000', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=9.9098),
    dict(label='linear_alsi12_Tdep4500', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=11.3591),
    dict(label='linear_alsi12_Tdep5000', E=0.8903, C=1.1708, r=0.7262, ste_cold=0.1634, ste_hot=12.8083),
"""
#    dict(label='zarzalejo',         E=1.68, C=1.59, r=1.42, ste_cold=2.42, ste_hot=2.80),
#    dict(label='rangel7',         E=0.66, C=1.16, r=0.5, ste_cold=1.57, ste_hot=2.93),  
#    dict(label='rangel8',         E=0.66, C=1.16, r=0.5, ste_cold=1.57, ste_hot=3.08),  
#    dict(label='rangel9',         E=0.72, C=1.12, r=0.5, ste_cold=0.13, ste_hot=0.34),
#   dict(label='rangel10',         E=0.72, C=1.12, r=0.5, ste_cold=0.13, ste_hot=0.63),
#    dict(label='heugenhauser',         E=0.85, C=1.17, r=0.62, ste_cold=0.27, ste_hot=0.42),            
#    dict(label='shallow',         E=1.20, C=1.00, r=1.00, ste_cold=0.33, ste_hot=0.30),
#    dict(label='small_resolved',  E=1.20, C=1.00, r=1.00, ste_cold=0.25, ste_hot=0.30),
#    dict(label='mid_resolved',    E=1.50, C=1.00, r=1.00, ste_cold=0.20, ste_hot=0.50),
#    dict(label='deep_low_r',      E=1.80, C=1.00, r=0.21, ste_cold=0.10, ste_hot=0.50),
#    dict(label='deep_high_r',     E=1.80, C=1.00, r=3.90, ste_cold=0.10, ste_hot=0.50),
#    dict(label='deep_low_C',      E=1.80, C=0.61, r=1.00, ste_cold=0.10, ste_hot=0.50),
#    dict(label='deep_high_C',     E=1.80, C=1.95, r=1.00, ste_cold=0.10, ste_hot=0.50),
#    dict(label='deep_low_ste_hot', E=1.80, C=1.00, r=1.00, ste_cold=0.10, ste_hot=0.02),
#    dict(label='deep_high_ste_hot',E=1.80, C=1.00, r=1.00, ste_cold=0.10, ste_hot=1.50),
#    dict(label='deep_low_ste_cold',E=1.80, C=1.00, r=1.00, ste_cold=0.05, ste_hot=0.50),
#    dict(label='deep_high_ste_cold',E=1.80, C=1.00, r=1.00, ste_cold=2.00, ste_hot=0.50),


# ── AlSi12, real-temperature grid ───────────────────────────────────────────
# E, C, r are fixed by AlSi12's material properties alone (k, rho, cp) and do
# NOT depend on the process temperatures -- only ste_hot and ste_cold do:
#   ste_hot  = cp_l*(T_dep - T_liq)/L   <- set by deposition/melt-pool temp
#   ste_cold = cp_s*(T_liq - T_sub)/L   <- set by substrate/interlayer temp
# So rather than picking (ste_hot, ste_cold) numbers by hand (as the earlier
# alsi12_650/700/800 cases did, sweeping only T_dep at a fixed T_sub), we
# sweep the actual physical temperatures [deg C] and derive the five
# dimensionless groups via the project's own materials.compute_adim_groups --
# this guarantees they stay mutually consistent the way a hand-tuned guess
# would not. T_dep is the deposition/melt-pool temperature; T_sub stands in
# for the previously-deposited layer temperature (interlayer temp in
# multilayer MEX/WAAM, not necessarily a cold baseplate), per the project's
# framing. AL_SI12: T_liq=581 C, T_sol=577 C -- guards require T_sub < T_sol.
#
# Edit these two lists to change the grid -- every (T_dep, T_sub) pair below
# becomes one PURE_CASES entry automatically, labelled
# 'alsi12_Tdep<T_dep>_Tsub<T_sub>'. All 16 of the combinations below were
# checked against check_guards_pure() and pass (theta_liquidus spans
# ~0.09-0.93 across the grid, well inside the guard band).
ALSI12_T_DEP = []#600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 3000, 5000, 10000, 20000]     # deposition/melt-pool temperature [deg C]
ALSI12_T_SUB = []#200, 300, 400, 515]     # interlayer/substrate temperature [deg C]


def _alsi12_case(T_dep: float, T_sub: float) -> Dict[str, Any]:
    """Build one PURE_CASES-style dict for AlSi12 at the given real process
    temperatures [deg C], via the project's own adimensionalisation function
    (not hand-computed) so E/C/r/ste_cold/ste_hot are guaranteed consistent."""
    props = dict(AL_SI12)
    props['T_deposit'] = T_dep
    props['T_sub'] = T_sub
    g = compute_adim_groups(props)
    return dict(
        label=f'alsi12_Tdep{T_dep:g}_Tsub{T_sub:g}',
        E=g['E'], C=g['C'], r=g['r'],
        ste_cold=g['ste_cold'], ste_hot=g['ste_hot'],
    )


PURE_CASES += [
   _alsi12_case(T_dep, T_sub)
   for T_dep in ALSI12_T_DEP
   for T_sub in ALSI12_T_SUB
]

SCHEIL_CASES = [
 #   dict(label='mid_shallow',      E=1.20, C=1.00, r=1.00, ste_cold=0.50, ste_hot=0.30, k_p=0.60, mushy_frac=0.80),

]
"""
    dict(label='shallow_low_ste',  E=1.20, C=1.00, r=1.00, ste_cold=0.30, ste_hot=0.30,
         k_p=0.60, mushy_frac=0.80),
    dict(label='near_eutectic',    E=1.20, C=1.00, r=1.00, ste_cold=0.30, ste_hot=0.30,
         k_p=0.95, mushy_frac=0.97),
    dict(label='strong_partition', E=1.20, C=1.00, r=1.00, ste_cold=0.30, ste_hot=0.30,
         k_p=0.12, mushy_frac=0.60),
    dict(label='wide_mushy_deep',  E=1.20, C=1.00, r=1.00, ste_cold=0.30, ste_hot=0.60,
         k_p=0.50, mushy_frac=0.55),
"""
_N_SNAPS = 100   # more snapshots than the big sweeps (8) since these are few in number


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Per-case runner (shared by pure and Scheil)
# ═══════════════════════════════════════════════════════════════════════════════

def _json_safe(obj):
    """Make a dict JSON-serialisable (numpy scalars, bools, etc.)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

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

def _refine_cost_estimate(grid_f) -> float:
    """Rough node-update count for a fine-grid solve (nz * nt_max). Used only
    to print a cost warning -- not a physical quantity."""
    return float(len(grid_f.z)) * float(grid_f.nt_max)


def run_pure_case(case: Dict[str, Any], dz_coarse: float, dz_fine: float,
                  allow_refine: bool) -> Dict[str, Any]:
    """Run one pure-metal case end-to-end. Returns a dict with everything
    needed to save full space/time output plus a summary row."""
    label = case['label']
    E, C, r = case['E'], case['C'], case['r']
    ste_cold, ste_hot = case['ste_cold'], case['ste_hot']
    lam_neumann = neumann_groups_pure(E, C, r, ste_cold, ste_hot)
    adim = groups_pure(E, C, r, ste_cold, ste_hot)

    reason = check_guards_pure(adim)
    if reason:
        return dict(model='pure_metal', label=label, status=f'skipped_guard:{reason}',
                     inputs=case, adim=adim)

    t_total, Z_total, flags = size_run(E, C, r, ste_cold, ste_hot, lam_neumann=lam_neumann)

    grid_c = build_grid(Z_total=Z_total, dz=dz_coarse, t_total=t_total, r=r)
    snap_idx_c = list(np.linspace(0, grid_c.nt_max - 1, _N_SNAPS, dtype=np.int64))
    result_c = fd_solve(adim, grid_c.z, grid_c.interface_idx, grid_c.dt,
                        grid_c.nt_max, snap_idx_c, model='pure_metal')

    D_rem_coarse = result_c.D_rem
    refined = False
    refine_skipped = False
    result = result_c
    grid_used = grid_c
    snap_idx_used = snap_idx_c
    dz_used = dz_coarse

    wants_refine = P.shallow_lo < D_rem_coarse < P.shallow_hi
    if wants_refine:
        grid_f = build_grid(Z_total=Z_total, dz=dz_fine, t_total=t_total, r=r)
        cost = _refine_cost_estimate(grid_f)
        if allow_refine:
            snap_idx_f = list(np.linspace(0, grid_f.nt_max - 1, _N_SNAPS, dtype=np.int64))
            result_f = fd_solve(adim, grid_f.z, grid_f.interface_idx, grid_f.dt,
                                grid_f.nt_max, snap_idx_f, model='pure_metal')
            result = result_f
            grid_used = grid_f
            snap_idx_used = snap_idx_f
            dz_used = dz_fine
            refined = True
        else:
            refine_skipped = True
            print(f"\n      [refine skipped] D_rem_coarse={D_rem_coarse:.4f}h is in the "
                  f"shallow band; refining at dz_fine={dz_fine} needs nz={len(grid_f.z)}, "
                  f"nt_max={grid_f.nt_max} (~{cost:.2e} node-updates -- likely several "
                  f"minutes). Keeping the coarse result. Pass --refine to run it anyway.",
                  end='')

    return dict(
        model='pure_metal', label=label, status='ok',
        inputs=case, adim=adim,
        t_total=t_total, Z_total=Z_total,
        time_capped=flags.time_capped, domain_capped=flags.domain_capped,
        D_rem_coarse=D_rem_coarse, refined=refined, refine_skipped=refine_skipped,
        dz_used=dz_used,
        snap_idx=snap_idx_used, grid=grid_used, result=result, lam_neumann=lam_neumann,
    )



def run_scheil_case(case: Dict[str, Any], dz_coarse: float, dz_fine: float,
                    allow_refine: bool) -> Dict[str, Any]:
    """Run one Scheil case end-to-end. Returns a dict with everything needed
    to save full space/time output plus a summary row."""
    label = case['label']
    E, C, r = case['E'], case['C'], case['r']
    ste_cold, ste_hot = case['ste_cold'], case['ste_hot']
    k_p, mushy_frac = case['k_p'], case['mushy_frac']

    adim = groups_scheil(E, C, r, ste_cold, ste_hot, k_p, mushy_frac)

    reason = check_guards_scheil(adim)
    if reason:
        return dict(model='scheil', label=label, status=f'skipped_guard:{reason}',
                     inputs=case, adim=adim)

    t_total, Z_total, flags = size_run(E, C, r, ste_cold, ste_hot, lam_neumann=None)

    grid_c = build_grid(Z_total=Z_total, dz=dz_coarse, t_total=t_total, r=r)
    snap_idx_c = list(np.linspace(0, grid_c.nt_max - 1, _N_SNAPS, dtype=np.int64))
    result_c = fd_solve(adim, grid_c.z, grid_c.interface_idx, grid_c.dt,
                        grid_c.nt_max, snap_idx_c, model='scheil')

    D_rem_coarse = result_c.D_rem
    refined = False
    refine_skipped = False
    result = result_c
    grid_used = grid_c
    snap_idx_used = snap_idx_c
    dz_used = dz_coarse

    wants_refine = P.shallow_lo < D_rem_coarse < P.shallow_hi
    if wants_refine:
        grid_f = build_grid(Z_total=Z_total, dz=dz_fine, t_total=t_total, r=r)
        cost = _refine_cost_estimate(grid_f)
        if allow_refine:
            snap_idx_f = list(np.linspace(0, grid_f.nt_max - 1, _N_SNAPS, dtype=np.int64))
            result_f = fd_solve(adim, grid_f.z, grid_f.interface_idx, grid_f.dt,
                                grid_f.nt_max, snap_idx_f, model='scheil')
            result = result_f
            grid_used = grid_f
            snap_idx_used = snap_idx_f
            dz_used = dz_fine
            refined = True
        else:
            refine_skipped = True
            print(f"\n      [refine skipped] D_rem_coarse={D_rem_coarse:.4f}h is in the "
                  f"shallow band; refining at dz_fine={dz_fine} needs nz={len(grid_f.z)}, "
                  f"nt_max={grid_f.nt_max} (~{cost:.2e} node-updates -- likely several "
                  f"minutes). Keeping the coarse result. Pass --refine to run it anyway.",
                  end='')

    return dict(
        model='scheil', label=label, status='ok',
        inputs=case, adim=adim,
        t_total=t_total, Z_total=Z_total,
        time_capped=flags.time_capped, domain_capped=flags.domain_capped,
        D_rem_coarse=D_rem_coarse, refined=refined, refine_skipped=refine_skipped,
        dz_used=dz_used,
        snap_idx=snap_idx_used, grid=grid_used, result=result,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Saving full space/time output for one case
# ═══════════════════════════════════════════════════════════════════════════════

def save_case(case_out: Dict[str, Any], case_dir: Path) -> Dict[str, Any]:
    """
    Write time_series.csv, space_profile.csv, theta_snapshots.csv, meta.json
    for one case. Returns the summary row dict for this case (for the
    cross-case summary.csv).
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    label = case_out['label']
    model = case_out['model']

    if case_out['status'] != 'ok':
        meta = {
            'model': model, 'label': label, 'status': case_out['status'],
            'inputs': case_out['inputs'], 'adim': case_out.get('adim', {}),
        }
        with open(case_dir / 'meta.json', 'w') as f:
            json.dump(_json_safe(meta), f, indent=2)
        return {
            'model': model, 'label': label, 'status': case_out['status'],
            **case_out['inputs'],
        }

    result = case_out['result']
    grid = case_out['grid']
    snap_idx = case_out['snap_idx']
    dt = grid.dt
    refine_skipped = case_out.get('refine_skipped', False)

    # ── Time series (front positions, interface temp vs real time) ──────────
    df_time = result.df_time.copy()
    df_time.insert(1, 't', df_time['timestep'] * dt)
    df_time.to_csv(case_dir / 'time_series.csv', index=False)

    # ── Final + peak-remelt spatial profile ──────────────────────────────────
    df_space = result.df_space.copy()
    df_space.to_csv(case_dir / 'space_profile.csv', index=False)

    # ── Full theta(z,t) snapshot history, long format ────────────────────────
    n_snap_actual = result.snap_ctr
    rows = []
    for s in range(n_snap_actual):
        t_snap = snap_idx[s] * dt
        for zi, theta in zip(grid.z, result.theta_hist[:, s]):
            rows.append({'snapshot_idx': s, 'timestep': int(snap_idx[s]),
                        't': t_snap, 'z': zi, 'theta': theta})
    pd.DataFrame(rows).to_csv(case_dir / 'theta_snapshots.csv', index=False)

    # ── Metadata: full provenance for this case ───────────────────────────────
    meta = {
        'model': model, 'label': label, 'status': 'ok',
        'inputs': case_out['inputs'],
        'adim_groups': case_out['adim'],
        't_total': case_out['t_total'], 'Z_total': case_out['Z_total'],
        'time_capped': case_out['time_capped'], 'domain_capped': case_out['domain_capped'],
        'D_rem_coarse': case_out['D_rem_coarse'], 'refined': case_out['refined'],
        'refine_skipped': refine_skipped,
        'dz_used': case_out['dz_used'], 'dt_used': dt, 'nt_max': grid.nt_max,
        'nz': len(grid.z), 'interface_idx': grid.interface_idx,
        'n_snapshots_requested': _N_SNAPS, 'n_snapshots_captured': n_snap_actual,
        'D_rem': result.D_rem, 'D_sol': result.D_sol,
        'mushy_width_at_max_remelt': result.mushy_width_at_max_remelt,
        'time_to_max_remelt_step': result.time_to_max_remelt,
        'time_to_max_remelt_t': result.time_to_max_remelt * dt,
        'time_to_max_solidus_step': result.time_to_max_solidus,
        'time_to_max_solidus_t': result.time_to_max_solidus * dt,
        'final_interface_temp': result.final_interface_temp,
        'config_fingerprint': P.fingerprint(),
        'lam_neumann': case_out.get('lam_neumann', None),
    }
    with open(case_dir / 'meta.json', 'w') as f:
        json.dump(_json_safe(meta), f, indent=2)

    return {
        'model': model, 'label': label, 'status': 'ok',
        **case_out['inputs'],
        'D_rem': result.D_rem, 'D_sol': result.D_sol,
        'D_rem_coarse': case_out['D_rem_coarse'],
        'refined': case_out['refined'], 'refine_skipped': refine_skipped,
        'dz_used': case_out['dz_used'],
        't_total': case_out['t_total'], 'Z_total': case_out['Z_total'],
        'time_capped': case_out['time_capped'], 'domain_capped': case_out['domain_capped'],
        'time_to_max_remelt_t': result.time_to_max_remelt * dt,
        'time_to_max_solidus_t': result.time_to_max_solidus * dt,
        'mushy_width_at_max_remelt': result.mushy_width_at_max_remelt,
        'final_interface_temp': result.final_interface_temp,
        'lam_neumann': case_out.get('lam_neumann', None),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(outdir: str, dz_coarse: float, dz_fine: float, allow_refine: bool):
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"{'='*64}")
    print(f"  Case-study runner  (config fingerprint {P.fingerprint()})")
    print(f"  {len(PURE_CASES)} pure cases, {len(SCHEIL_CASES)} Scheil cases")
    print(f"  dz_coarse={dz_coarse}  dz_fine={dz_fine}  refine={'ON' if allow_refine else 'OFF (default)'}")
    print(f"  Output: {out_path.resolve()}")
    print(f"{'='*64}\n")

    summary_rows = []

    for i, case in enumerate(PURE_CASES, start=1):
        tag = f"pure_case{i:02d}_{case['label']}"
        print(f"  [{tag}]  E={case['E']} C={case['C']} r={case['r']} "
              f"ste_cold={case['ste_cold']} ste_hot={case['ste_hot']}  ...", end=' ')
        out = run_pure_case(case, dz_coarse, dz_fine, allow_refine)
        row = save_case(out, out_path / tag)
        row['case_tag'] = tag
        summary_rows.append(row)
        if out['status'] == 'ok':
            print(f"D_rem={out['result'].D_rem:.4f}h  refined={out['refined']}")
        else:
            print(out['status'])

    for i, case in enumerate(SCHEIL_CASES, start=1):
        tag = f"scheil_case{i:02d}_{case['label']}"
        print(f"  [{tag}]  E={case['E']} C={case['C']} r={case['r']} "
              f"ste_cold={case['ste_cold']} ste_hot={case['ste_hot']} "
              f"k_p={case['k_p']} mushy_frac={case['mushy_frac']}  ...", end=' ')
        out = run_scheil_case(case, dz_coarse, dz_fine, allow_refine)
        row = save_case(out, out_path / tag)
        row['case_tag'] = tag
        summary_rows.append(row)
        if out['status'] == 'ok':
            print(f"D_rem={out['result'].D_rem:.4f}h  D_sol={out['result'].D_sol:.4f}h  "
                  f"refined={out['refined']}")
        else:
            print(out['status'])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path / 'summary.csv', index=False)

    print(f"\n  Done. Summary written to {(out_path / 'summary.csv').resolve()}")


def _parse():
    p = argparse.ArgumentParser(
        description='Run a small hand-picked set of pure/Scheil cases and save '
                    'full space- and time-resolved results per case.')
    p.add_argument('--outdir', type=str, default='case_studies_out',
                   help='Output directory (default: case_studies_out)')
    p.add_argument('--dz_coarse', type=float, default=P.dz_coarse,
                   help=f'Coarse grid spacing [h] (default {P.dz_coarse})')
    p.add_argument('--dz_fine', type=float, default=P.dz_fine,
                   help=f'Fine (refinement) grid spacing [h] (default {P.dz_fine})')
    p.add_argument('--refine', action='store_true',
                   help='Actually run dz_fine inline refinement for cases whose coarse '
                        'D_rem falls in the shallow band [%.3f, %.3f). This matches the '
                        'big-sweep logic exactly, but a single fine-grid solve can take '
                        'several minutes (nz and nt_max both scale up sharply at dz_fine). '
                        'Default OFF: shallow-band cases keep their (faster) coarse result '
                        'and are flagged refine_skipped=True in the summary.'
                        % (P.shallow_lo, P.shallow_hi))
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    main(outdir=args.outdir, dz_coarse=args.dz_coarse, dz_fine=args.dz_fine,
         allow_refine=args.refine)