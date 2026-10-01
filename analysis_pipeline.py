"""
analyse_pipeline.py
====================
First-principles analysis pipeline for the dimensionless remelting/solidification
sweep data (pure-metal and Scheil tracks).

Usage
-----
    python analyse_pipeline.py --csv results_refined.csv --model pure --legacy
    python analyse_pipeline.py --csv results_pure/results.csv --model pure
    python analyse_pipeline.py --csv results_scheil/results.csv --model scheil
    python analyse_pipeline.py --csv results_refined.csv --model pure --legacy --phases 0
    python analyse_pipeline.py --csv ... --model pure --outdir figs_pure --no-plots

Requires: numpy, pandas, scipy, scikit-learn, matplotlib. Tries to import
conventions.py / config_pipeline.py / grid.py from the same directory for the
identity/censoring checks; degrades gracefully (prints a warning, skips that
sub-check) if they're not importable from the current working directory.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Optional same-directory project imports (graceful degradation) ─────────
# Guard with direct `is not None` checks on _P / _tau_dev themselves (not a
# separate _HAVE_SPINE boolean) so the type checker can actually narrow these
# from Optional[...] to their real type inside each guarded block, instead
# of flagging every _P.attr / _tau_dev(...) call as "not a known attribute
# of None".
try:
    from grid import tau_dev as _tau_dev
    from config_pipeline import PARAMS as _P
except ImportError:
    _tau_dev = None
    _P = None


# ═══════════════════════════════════════════════════════════════════════════
# 0.  Column harmonisation
# ═══════════════════════════════════════════════════════════════════════════

# Legacy results_refined.csv predates the pipeline's column names in a few
# places. Map them onto the new-pipeline names so every downstream function
# only ever has to know one schema.
_LEGACY_ALIASES = {
    'theta_melt': 'theta_liquidus',   # legacy name -> canonical name
}


_NUMERIC_COLUMNS = [
    'lhs_id', 'op_id', 'E', 'C', 'r', 'ste_cold', 'ste_hot',
    'theta_liquidus', 'theta_solidus', 'theta_contact',
    'inv_Ste', 'k_ratio', 'rho_ratio', 'cp_ratio',
    'mushy_frac', 'k_p', 'Ste_m',
    'lam_neumann', 'lam_liq', 'lam_sol',
    't_total', 'Z_total',
    'D_rem_coarse', 'D_rem', 'D_sol_coarse', 'D_sol', 'mushy_width_at_max_remelt',
    't_at_max_remelt', 't_at_max_solidus',
    'Effusivity condition', 'Remelting Prediction', 'Remelting FD',
]
# If any of these are corrupted on a status=='ok' row, that row can't be
# trusted for melt classification or depth regression even though it's
# labelled 'ok'. This is exactly the set of raw columns _build_features /
# _build_features_margin actually read for each model -- if a NaN is hiding
# in any of these, it surfaces as "PolynomialFeatures does not accept NaN"
# three phases downstream rather than here, where it's actually traceable.
_CRITICAL_NUMERIC_COLUMNS_BY_MODEL = {
    'pure': ['E', 'C', 'r', 'ste_cold', 'ste_hot', 'theta_liquidus', 'D_rem', 'D_rem_coarse'],
    'scheil': ['E', 'C', 'r', 'ste_cold', 'ste_hot', 'theta_liquidus', 'theta_solidus',
              'mushy_frac', 'k_p', 'D_rem', 'D_rem_coarse', 'D_sol', 'D_sol_coarse'],
}


def load_data(csv_path: str, model: str, legacy: bool) -> pd.DataFrame:
    """Load a sweep CSV and harmonise columns to the canonical schema.

    Returns the FULL dataframe (including non-'ok' rows if a status column
    exists) — callers that want only successful runs should filter on
    df['status'] == 'ok' themselves (legacy data has no status column and is
    treated as if status == 'ok' for all rows, since guard-skipped rows were
    never written to results_refined.csv in the first place).

    Also coerces every physics/measurement column to numeric (a real sweep
    CSV had a stray non-numeric value in D_rem_coarse on rows labelled 'ok',
    which made the WHOLE column object-dtype and crashed the very first
    comparison downstream with "'>=' not supported between instances of
    'str' and 'float'" three call-frames away from the actual bad data --
    fixing it here, once, protects every phase rather than patching whichever
    comparison happened to be first in line).
    """
    df = pd.read_csv(csv_path)

    # theta_melt -> theta_liquidus: NOT gated behind --legacy. The pure-metal
    # sweep's canonical ROW_SCHEMA uses 'theta_melt' (a deliberate naming
    # choice in the sweep script's own bugfix, matching legacy convention),
    # while Scheil's ROW_SCHEMA uses 'theta_liquidus' -- so this rename is
    # needed for current, non-legacy pure-sweep output too, not just genuinely
    # old results_refined.csv. Only rename if theta_liquidus isn't already
    # present, so this is a no-op on Scheil data and harmless either way.
    if 'theta_melt' in df.columns and 'theta_liquidus' not in df.columns:
        df = df.rename(columns={'theta_melt': 'theta_liquidus'})

    if legacy:
        df = df.rename(columns={k: v for k, v in _LEGACY_ALIASES.items() if k in df.columns})
        if 'status' not in df.columns:
            df['status'] = 'ok'
        if 'time_capped' not in df.columns:
            df['time_capped'] = False
        if 'domain_capped' not in df.columns:
            df['domain_capped'] = False
        if 'config_fingerprint' not in df.columns:
            df['config_fingerprint'] = np.nan

    if model == 'scheil' and 'theta_liq' in df.columns and 'theta_liquidus' not in df.columns:
        df = df.rename(columns={'theta_liq': 'theta_liquidus', 'theta_sol': 'theta_solidus'})

    # ── Numeric sanitisation ─────────────────────────────────────────────────
    corrupted_by_col = {}
    for col in _NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        was_present = df[col].notna()
        coerced = pd.to_numeric(df[col], errors='coerce')
        newly_nan = was_present & coerced.isna()
        if newly_nan.any():
            corrupted_by_col[col] = int(newly_nan.sum())
        df[col] = coerced

    if corrupted_by_col:
        print(f"  ⚠ load_data: non-numeric value(s) found in supposedly-numeric columns, "
              f"coerced to NaN: {corrupted_by_col}. This means a sweep-script bug wrote a "
              f"non-numeric placeholder into one of these columns for some rows -- inspect "
              f"the raw CSV for the affected rows before trusting them.")

    # ── Derived-column recovery (pure-metal) ────────────────────────────────
    # theta_liquidus, inv_Ste, k_ratio, rho_ratio are all simple, deterministic
    # functions of E/C/r/ste_cold/ste_hot (the same formulas the identity
    # audit in Phase 0.3 already checks them against). If one of these came
    # out NaN on disk -- e.g. a CSV-writing bug in the sweep script dropped a
    # column's value even though the case ran fine and every OTHER column is
    # intact -- it's recoverable rather than a reason to discard the row.
    # Runs AFTER numeric sanitisation above, so E/C/r/ste_cold/ste_hot are
    # guaranteed clean float dtype before doing arithmetic on them.
    # Real-world motivating case: a full results_pure.csv run had
    # theta_liquidus == NaN for every single 'ok' row while ste_cold/ste_hot/C
    # (its own inputs) were all fine -- recomputing it directly here recovers
    # the entire dataset instead of quarantining 28,860/28,860 rows over a
    # column the sweep script never actually needed to round-trip through disk.
    if model == 'pure' and all(c in df.columns for c in ('C', 'ste_cold', 'ste_hot')):
        recovered = {}
        have_C_sc_sh = df[['C', 'ste_cold', 'ste_hot']].notna().all(axis=1)
        if 'theta_liquidus' in df.columns:
            missing = df['theta_liquidus'].isna() & have_C_sc_sh
            if missing.any():
                df.loc[missing, 'theta_liquidus'] = (
                    df.loc[missing, 'ste_cold'] /
                    (df.loc[missing, 'ste_cold'] + df.loc[missing, 'ste_hot'] / df.loc[missing, 'C']))
                recovered['theta_liquidus'] = int(missing.sum())
        if 'inv_Ste' in df.columns:
            missing = df['inv_Ste'].isna() & have_C_sc_sh
            if missing.any():
                df.loc[missing, 'inv_Ste'] = 1.0 / (
                    df.loc[missing, 'ste_cold'] + df.loc[missing, 'ste_hot'] / df.loc[missing, 'C'])
                recovered['inv_Ste'] = int(missing.sum())
        if all(c in df.columns for c in ('E', 'r')):
            have_E_r = df[['E', 'r']].notna().all(axis=1)
            if 'k_ratio' in df.columns:
                missing = df['k_ratio'].isna() & have_E_r
                if missing.any():
                    df.loc[missing, 'k_ratio'] = df.loc[missing, 'E'] * np.sqrt(df.loc[missing, 'r'])
                    recovered['k_ratio'] = int(missing.sum())
            if 'rho_ratio' in df.columns:
                have_E_r_C = have_E_r & df['C'].notna()
                missing = df['rho_ratio'].isna() & have_E_r_C
                if missing.any():
                    df.loc[missing, 'rho_ratio'] = (
                        df.loc[missing, 'E'] / (df.loc[missing, 'C'] * np.sqrt(df.loc[missing, 'r'])))
                    recovered['rho_ratio'] = int(missing.sum())
        if recovered:
            print(f"  ⚠ load_data: recovered {recovered} by recomputing from intact raw inputs "
                  f"(E/C/r/ste_cold/ste_hot) -- the stored value was missing on disk but every "
                  f"input needed to reconstruct it was fine. This is a real bug worth tracking "
                  f"down in the sweep script's CSV-writing path, but it isn't a reason to lose "
                  f"these rows here.")

    # ── Boolean sanitisation ─────────────────────────────────────────────────
    # 'refined' (and friends) get used directly as boolean masks throughout
    # the pipeline (df_ok[df_ok['refined']]). pandas raises rather than
    # silently coercing if the column has ANY NaN in it ("Cannot mask with
    # non-boolean array containing NA / NaN values") -- a missing flag here
    # almost always means the sweep script just didn't set it for that row
    # (e.g. a case that was never run through whatever check the flag
    # tracks), which is exactly what the flag being False should mean.
    # Treating missing-as-False is the conservative reading: it doesn't
    # inflate refined/capped counts, it just stops a NaN from being load-
    # bearing in a boolean context three phases downstream.
    for col in ('time_capped', 'domain_capped', 'refined', 'censored', 'became_zero'):
        if col not in df.columns:
            continue
        n_nan = int(df[col].isna().sum())
        if n_nan:
            print(f"  ⚠ load_data: {n_nan} NaN/missing value(s) in boolean column '{col}', "
                  f"treated as False -- inspect the raw CSV if this count looks larger than "
                  f"a handful of edge cases.")
        df[col] = df[col].fillna(False).astype(bool)

    # Quarantine 'ok' rows with NaN in any column the rest of the pipeline
    # actually depends on -- runs UNCONDITIONALLY, not just when the loop
    # above found string corruption. A column can be NaN on an 'ok' row for
    # reasons that have nothing to do with the CSV-text corruption above
    # (e.g. a derived quantity like theta_liquidus that came out NaN from an
    # edge-case computation in the sweep script itself) -- the result is the
    # same either way: PolynomialFeatures three phases downstream gets a NaN
    # it can't handle, with no trace back to which row caused it. Catching
    # it here, once, by column rather than waiting for whichever fit happens
    # to run first is the whole point.
    if 'status' in df.columns:
        critical_cols = _CRITICAL_NUMERIC_COLUMNS_BY_MODEL.get(model, [])
        present_critical = [c for c in critical_cols if c in df.columns]
        if present_critical:
            corrupted_critical = df[present_critical].isna().any(axis=1) & (df['status'] == 'ok')
            n_relabelled = int(corrupted_critical.sum())
            if n_relabelled:
                nan_counts = {c: int(df.loc[corrupted_critical, c].isna().sum())
                             for c in present_critical if df.loc[corrupted_critical, c].isna().any()}
                df.loc[corrupted_critical, 'status'] = 'corrupted_numeric_value'
                print(f"  ⚠ load_data: {n_relabelled} row(s) labelled 'ok' had NaN in a column "
                      f"the pipeline depends on directly ({nan_counts}) -- relabelled status to "
                      f"'corrupted_numeric_value' so they're excluded from df_ok like any other "
                      f"failure status, instead of crashing or silently corrupting results "
                      f"three phases downstream.")

    return df


def classify_melt(D_rem: pd.Series, no_melt_thr=0.000) -> pd.Series:
    """The ONLY population split used for modelling: did this case melt at
    all, by the FD ground truth. Replaces the old three-way classify_population
    — see module docstring for why the transient/resolved distinction was
    dropped (no detectable regime change at the old D_rem=0.10 boundary)."""
    return D_rem > no_melt_thr


def shallow_band_mask(D_rem_coarse: pd.Series, no_melt_thr=0.005, refine_hi=0.10) -> pd.Series:
    """Cases whose COARSE D_rem falls in the inline-refinement trigger band.
    This is bookkeeping for the dz_fine refinement step (Phase 0.6 / 2.1) —
    i.e. a numerical-resolution concern (D_rem here is only 2-10 grid cells at
    dz_coarse=0.01h) — NOT a modelling category. Use classify_melt for that."""
    return (D_rem_coarse >= no_melt_thr) & (D_rem_coarse < refine_hi)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0 — Integrity
# ═══════════════════════════════════════════════════════════════════════════

def phase0_integrity(df: pd.DataFrame, model: str, legacy: bool,
                      no_melt_thr=0.005, resolved_thr=0.10) -> dict:
    print("\n" + "═" * 78)
    print("  PHASE 0 — INTEGRITY")
    print("═" * 78)

    report = {}

    # ── Status distribution ─────────────────────────────────────────────────
    print("\n[0.1] Status distribution")
    status_counts = df['status'].value_counts()
    print(status_counts.to_string())
    n_ok = int((df['status'] == 'ok').sum())
    n_total = len(df)
    print(f"  ok: {n_ok}/{n_total} ({100*n_ok/n_total:.1f}%)")
    non_trivial = [s for s in status_counts.index
                   if s != 'ok' and not str(s).startswith('skipped_guard')]
    if non_trivial:
        print(f"  ⚠ non-trivial failure statuses present: {non_trivial} "
              f"— inspect before trusting this dataset.")
    report['status_counts'] = status_counts.to_dict()

    df_ok = df[df['status'] == 'ok'].copy()

    # ── Config fingerprint ──────────────────────────────────────────────────
    print("\n[0.2] Config fingerprint")
    fps = df_ok['config_fingerprint'].dropna().unique() if 'config_fingerprint' in df_ok else []
    if legacy:
        print("  legacy data — no fingerprint stamped, skipped.")
    elif len(fps) == 0:
        print("  no fingerprint found in data.")
    else:
        if len(fps) > 1:
            print(f"  ⚠ MULTIPLE fingerprints in one file: {list(fps)} "
                  f"— this file mixes runs from different configs.")
        else:
            print(f"  data fingerprint: {fps[0]}")
        if _P is not None:
            current = _P.fingerprint()
            print(f"  current config_pipeline.py fingerprint: {current}")
            if len(fps) == 1 and fps[0] != current:
                print("  ⚠ data fingerprint != current config — PARAMS changed "
                      "since this data was generated. Confirm that's expected "
                      "(e.g. you intentionally bumped a threshold) before reusing it.")
        else:
            print("  (config_pipeline.py not importable from cwd — can't compare)")
    report['fingerprints'] = list(map(str, fps))

    # ── Algebraic identity audit ────────────────────────────────────────────
    print("\n[0.3] Algebraic identity audit (recomputed from raw inputs)")
    if model == 'pure':
        E, C, r = df_ok['E'], df_ok['C'], df_ok['r']
        sc, sh = df_ok['ste_cold'], df_ok['ste_hot']
        checks = {
            'k_ratio = E*sqrt(r)':        (df_ok.get('k_ratio'), E * np.sqrt(r)),
            'rho_ratio = E/(C*sqrt(r))':  (df_ok.get('rho_ratio'), E / (C * np.sqrt(r))),
            'theta_liquidus = sc/(sc+sh/C)': (df_ok.get('theta_liquidus'), sc / (sc + sh / C)),
        }
        if 'inv_Ste' in df_ok.columns:
            checks['inv_Ste = 1/(sc+sh/C)'] = (df_ok['inv_Ste'], 1.0 / (sc + sh / C))
    else:
        E, r = df_ok['E'], df_ok['r']
        mf = df_ok['mushy_frac']
        tl = df_ok['theta_liquidus']
        checks = {
            'theta_solidus = mushy_frac*theta_liquidus': (df_ok.get('theta_solidus'), mf * tl),
            'k_ratio = E*sqrt(r)': (df_ok.get('k_ratio'), E * np.sqrt(r)),
        }

    report['identity_checks'] = {}
    for name, (lhs, rhs) in checks.items():
        if lhs is None:
            print(f"  [SKIP] {name} — column not present")
            continue
        n_nan = int(lhs.isna().sum())
        relerr = (lhs - rhs).abs() / rhs.abs().clip(lower=1e-12)
        # relerr is NaN wherever lhs is NaN, and "NaN > 1e-6" is False (not an
        # error) -- silently undercounting failures by exactly the NaN count.
        # pandas' .max() also defaults to skipna=True, hiding NaN from
        # max_relerr too. Both are fixed explicitly here: a NaN stored value
        # IS a failure (it can't be verified against anything), and max() is
        # computed with skipna=False so a hidden NaN shows up as exactly that
        # rather than a falsely-clean small number.
        n_fail = int(((relerr > 1e-6) | lhs.isna()).sum())
        max_relerr = relerr.max(skipna=False)
        tag = "PASS" if n_fail == 0 else "FAIL"
        nan_note = f"  ⚠ {n_nan} NaN in stored value (counted as failures)" if n_nan else ""
        print(f"  [{tag}] {name}  max_relerr={max_relerr:.2e}  n_fail={n_fail}/{len(lhs)}{nan_note}")
        report['identity_checks'][name] = {'max_relerr': float(max_relerr) if pd.notna(max_relerr) else None,
                                           'n_fail': n_fail, 'n_nan': n_nan}

    # ── Duplicate LHS points ────────────────────────────────────────────────
    print("\n[0.4] Duplicate LHS-point detection")
    group_keys = ['E', 'C', 'r'] if model == 'pure' else ['E', 'C', 'r', 'mushy_frac', 'k_p']
    group_keys = [k for k in group_keys if k in df_ok.columns]
    rounded = df_ok[group_keys].round(6)
    dup_mask = rounded.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_groups = int(rounded[dup_mask].drop_duplicates().shape[0])
    print(f"  {n_dup_rows} rows belong to {n_dup_groups} duplicated material points "
          f"({n_dup_rows / len(df_ok) * 100:.2f}% of ok rows)")
    report['n_duplicate_rows'] = n_dup_rows

    # ── Melt / no-melt split ─────────────────────────────────────────────────
    print("\n[0.5] Melt / no-melt split (the only population split used downstream)")
    melted = classify_melt(df_ok['D_rem'], no_melt_thr)
    n_melt = int(melted.sum())
    print(f"  no_melt: {len(df_ok)-n_melt:6d}  ({100*(len(df_ok)-n_melt)/len(df_ok):.1f}%)")
    print(f"  melted : {n_melt:6d}  ({100*n_melt/len(df_ok):.1f}%)")
    if 'D_rem_coarse' in df_ok.columns:
        shallow = shallow_band_mask(df_ok['D_rem_coarse'], no_melt_thr, resolved_thr)
        print(f"  (of which {int(shallow.sum())} fell in the refinement-trigger band "
              f"[{no_melt_thr}, {resolved_thr})h on the coarse grid — bookkeeping only, "
              f"see Phase 0.6; not a separate modelling population)")
    report['population_counts'] = {'no_melt': len(df_ok) - n_melt, 'melted': n_melt}

    # ── Refinement audit ─────────────────────────────────────────────────────
    if 'refined' in df_ok.columns:
        print("\n[0.6] Inline-refinement audit (numerical bookkeeping, not a population)")
        n_refined = int(df_ok['refined'].sum())
        print(f"  {n_refined} rows refined at fine grid "
              f"({n_refined/len(df_ok)*100:.2f}% of ok rows)")
        if n_refined > 0:
            ref = df_ok[df_ok['refined']]
            became_no_melt = int((ref['D_rem'] < no_melt_thr).sum())
            exceeded_band = int((ref['D_rem'] >= resolved_thr).sum())
            print(f"    -> reclassified to no_melt after refinement: {became_no_melt}")
            print(f"    -> exceeded the refinement band (D_rem >= {resolved_thr}) after refinement: "
                  f"{exceeded_band}")
            rel_change = (ref['D_rem'] - ref['D_rem_coarse']).abs() / ref['D_rem_coarse'].clip(lower=1e-6)
            print(f"    -> median |ΔD_rem|/D_rem_coarse in shallow band: {rel_change.median()*100:.1f}%  "
                  f"(90th pct: {rel_change.quantile(0.9)*100:.1f}%)")
        report['n_refined'] = n_refined

    # ── Time-censoring audit (the from-first-principles re-derivation) ──────
    print("\n[0.7] Time-censoring audit, all melted cases")
    melted_df = df_ok[melted].copy()
    if _P is not None and _tau_dev is not None and 'r' in melted_df.columns and 'theta_liquidus' in melted_df.columns:
        melted_df['tau_dev'] = [
            _tau_dev(r, tm) for r, tm in zip(melted_df['r'], melted_df['theta_liquidus'])
        ]
        finite = melted_df['tau_dev'].notna()
        n_finite = int(finite.sum())
        crit_simple = finite & (melted_df['tau_dev'] > melted_df['t_total'])
        # Small relative tolerance on criterion B specifically: size_run() sets
        # t_total = tau_dev_mult*tau_dev EXACTLY for any non-time-capped case,
        # so recomputing tau_dev here from CSV-round-tripped r/theta_liquidus
        # reproduces that boundary to within floating-point noise, not
        # exactly -- without the tolerance, cases sitting precisely AT the
        # adequacy boundary (genuinely fine, not under-resourced) flip to
        # "censored" by ~1e-15. Confirmed directly while testing
        # rerun_censored.py against real Scheil output.
        safety_ratio = (_P.tau_dev_mult * melted_df['tau_dev']) / melted_df['t_total'].clip(lower=1e-12)
        crit_safety = finite & (safety_ratio > 1.0 + 1e-9)
        n_simple = int(crit_simple.sum())
        n_safety = int(crit_safety.sum())
        print(f"  melted cases with finite tau_dev (theta_melt<0.5): {n_finite}/{len(melted_df)}")
        print(f"  criterion A  tau_dev > t_total            (\"haven't reached even 1 tau_dev\"): "
              f"{n_simple} ({100*n_simple/len(melted_df):.1f}%)")
        print(f"  criterion B  {_P.tau_dev_mult}*tau_dev > t_total  (pipeline's OWN size_run() "
              f"adequacy bar): {n_safety} ({100*n_safety/len(melted_df):.1f}%)")
        if n_simple != n_safety:
            print(f"  ⚠ these differ by {n_safety - n_simple} cases. Criterion A is the looser "
                  f"bar (used to report '836' in earlier write-ups, restricted to the old "
                  f"'resolved' population only); criterion B is the bar the pipeline itself "
                  f"uses to decide t_total is *adequate* (1% peak-capture calibration), evaluated "
                  f"here on ALL melted cases. Quote B, not A, when claiming a dataset is non-censored.")
        report['censoring'] = {
            'n_melted': len(melted_df), 'n_finite_tau_dev': n_finite,
            'n_criterion_A_tau_dev_gt_t_total': n_simple,
            'n_criterion_B_safety_margin_violated': n_safety,
        }
    else:
        print("  grid.py/config_pipeline.py not importable, or required columns missing — skipped.")

    # ── Sizing-cap audit (only meaningful for new-pipeline data) ────────────
    if not legacy and 'time_capped' in df_ok.columns and 'domain_capped' in df_ok.columns:
        print("\n[0.8] Sizing-cap audit")
        n_tc = int(df_ok['time_capped'].sum())
        n_dc = int(df_ok['domain_capped'].sum())
        print(f"  time_capped:   {n_tc}  ({100*n_tc/len(df_ok):.2f}%) — these hit t_cap and may "
              f"STILL be time-censored even under adaptive sizing")
        print(f"  domain_capped: {n_dc}  ({100*n_dc/len(df_ok):.2f}%)")
        report['n_time_capped'] = n_tc
        report['n_domain_capped'] = n_dc
    elif not legacy:
        print("\n[0.8] Sizing-cap audit -- time_capped/domain_capped columns missing, skipped.")

    # ── Headline descriptive stats ───────────────────────────────────────────
    print("\n[0.9] D_rem descriptive stats (ok rows)")
    print(df_ok['D_rem'].describe().to_string())
    if model == 'scheil' and 'D_sol' in df_ok.columns:
        print("\n  D_sol descriptive stats")
        print(df_ok['D_sol'].describe().to_string())

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Visualize
# ═══════════════════════════════════════════════════════════════════════════

def phase1_visualize(df_ok: pd.DataFrame, model: str, outdir: Path,
                      no_melt_thr=0.005, resolved_thr=0.10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    melted = classify_melt(df_ok['D_rem'], no_melt_thr)
    pop = melted.map({True: 'melted', False: 'no_melt'})
    colors = {'no_melt': '#9aa5b1', 'melted': '#cc3b3b'}
    print("\n" + "═" * 78)
    print("  PHASE 1 — VISUALIZE")
    print("═" * 78)

    # [1.1] LHS coverage ------------------------------------------------------
    groups = ['E', 'C', 'r'] if model == 'pure' else ['E', 'C', 'r', 'mushy_frac', 'k_p']
    groups = [g for g in groups if g in df_ok.columns]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3))
    for ax, g in zip(np.atleast_1d(axes), groups):
        ax.hist(df_ok[g], bins=40, color='#4a6fa5')
        ax.set_title(g)
    fig.suptitle('LHS input coverage')
    fig.tight_layout()
    fig.savefig(outdir / '01_lhs_coverage.png', dpi=130)
    plt.close(fig)

    # [1.2] Melt / no-melt bar chart ------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 3.5))
    counts = pop.value_counts().reindex(['no_melt', 'melted']).fillna(0)
    count_vals = counts.to_numpy(dtype=float)
    ax.bar(list(counts.index), count_vals, color=[colors[c] for c in counts.index])
    for i, v in enumerate(count_vals):
        ax.text(i, v, f"{int(v)}", ha='center', va='bottom')
    ax.set_title('Melt / no-melt split')
    fig.tight_layout()
    fig.savefig(outdir / '02_population_split.png', dpi=130)
    plt.close(fig)

    # [1.3] Onset boundary: E vs theta_liquidus -------------------------------
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for cat in ['no_melt', 'melted']:
        m = pop == cat
        ax.scatter(df_ok.loc[m, 'theta_liquidus'], df_ok.loc[m, 'E'],
                   s=4, alpha=0.4, label=cat, color=colors[cat])
    th = np.linspace(0.005, 0.995, 400)
    ax.plot(th, th / (1 - th), 'k--', lw=1.5, label='E = θ/(1-θ)  [contact criterion]')
    ax.set_xlabel('theta_liquidus (theta_melt)')
    ax.set_ylabel('E')
    ax.set_ylim(0, df_ok['E'].quantile(0.99) * 1.1)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title('Onset boundary vs analytic contact-temperature criterion', fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / '03_onset_boundary.png', dpi=130)
    plt.close(fig)

    # [1.4] D_rem vs each input group, coloured by population -----------------
    feat_cols = ['E', 'C', 'r', 'ste_cold', 'ste_hot'] if model == 'pure' \
        else ['E', 'r', 'mushy_frac', 'k_p', 'ste_cold', 'ste_hot']
    feat_cols = [c for c in feat_cols if c in df_ok.columns]
    fig, axes = plt.subplots(2, (len(feat_cols) + 1) // 2,
                              figsize=(3.4 * ((len(feat_cols) + 1) // 2), 6))
    axes = np.ravel(axes)
    for ax, col in zip(axes, feat_cols):
        for cat in ['no_melt', 'melted']:
            m = pop == cat
            ax.scatter(df_ok.loc[m, col], df_ok.loc[m, 'D_rem'],
                       s=3, alpha=0.3, color=colors[cat])
        ax.set_xlabel(col)
        ax.set_ylabel('D_rem')
        if col in ('r', 'ste_cold', 'ste_hot'):
            ax.set_xscale('log')
    for ax in axes[len(feat_cols):]:
        ax.axis('off')
    fig.suptitle('D_rem vs each dimensionless group')
    fig.tight_layout()
    fig.savefig(outdir / '04_D_rem_vs_groups.png', dpi=130)
    plt.close(fig)

    # [1.5] Refinement parity (coarse vs fine, shallow band) ------------------
    if 'refined' in df_ok.columns and df_ok['refined'].any():
        ref = df_ok[df_ok['refined']]
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.scatter(ref['D_rem_coarse'], ref['D_rem'], s=6, alpha=0.5, color='#4a6fa5')
        lims = [0, max(ref['D_rem_coarse'].max(), ref['D_rem'].max()) * 1.05]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlabel('D_rem_coarse (dz=0.01)')
        ax.set_ylabel('D_rem fine (dz=0.0025)')
        ax.set_title('Refinement parity, shallow band')
        fig.tight_layout()
        fig.savefig(outdir / '05_refinement_parity.png', dpi=130)
        plt.close(fig)

    # [1.6] Neumann eigenvalue cross-check (melted pop, pure only) ----------
    lam_col = 'lam_neumann' if model == 'pure' else 'lam_liq'
    if lam_col in df_ok.columns:
        res = df_ok[(pop == 'melted') & df_ok[lam_col].notna() & (df_ok[lam_col] > 0)]
        if len(res) > 5:
            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            logl, logd = np.log(res[lam_col]), np.log(res['D_rem'])
            ax.scatter(logl, logd, s=4, alpha=0.4, color='#cc3b3b')
            corr = np.corrcoef(logl, logd)[0, 1]
            ax.set_xlabel(f'log({lam_col})')
            ax.set_ylabel('log(D_rem)')
            ax.set_title(f'Neumann eigenvalue vs depth (melted), corr={corr:.3f}')
            fig.tight_layout()
            fig.savefig(outdir / '06_lam_neumann_crosscheck.png', dpi=130)
            plt.close(fig)
            print(f"  [1.6] corr(log {lam_col}, log D_rem) on melted population: {corr:.4f}")

    # [1.7] Scheil-only: D_sol vs D_rem and mushy width ------------------------
    if model == 'scheil' and 'D_sol' in df_ok.columns:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
        m = pop == 'melted'
        axes[0].scatter(df_ok.loc[m, 'D_rem'], df_ok.loc[m, 'D_sol'],
                        s=4, alpha=0.4, color=colors['melted'])
        lims = [0, df_ok['D_sol'].quantile(0.995)]
        axes[0].plot(lims, lims, 'k--', lw=1, label='D_sol = D_rem')
        axes[0].set_xlabel('D_rem'); axes[0].set_ylabel('D_sol'); axes[0].legend(fontsize=7)
        axes[0].set_title('Solidus vs liquidus depth')

        mushy_w = df_ok['D_sol'] - df_ok['D_rem']
        sc = axes[1].scatter(df_ok['mushy_frac'], mushy_w, c=df_ok['k_p'],
                             s=4, alpha=0.5, cmap='viridis')
        axes[1].set_xlabel('mushy_frac'); axes[1].set_ylabel('D_sol - D_rem (mushy width)')
        fig.colorbar(sc, ax=axes[1], label='k_p')
        axes[1].set_title('Mushy width structure')
        fig.tight_layout()
        fig.savefig(outdir / '07_scheil_mushy_structure.png', dpi=130)
        plt.close(fig)

    print(f"\n  figures written to {outdir.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Convergence
# ═══════════════════════════════════════════════════════════════════════════

def phase2_convergence(df_ok: pd.DataFrame, model: str,
                        no_melt_thr=0.005, resolved_thr=0.10) -> dict:
    print("\n" + "═" * 78)
    print("  PHASE 2 — CONVERGENCE")
    print("═" * 78)
    report = {}

    # ── Refinement effect size, by refinement-band outcome ───────────────────
    if 'refined' in df_ok.columns:
        print("\n[2.1] Coarse -> fine refinement effect (shallow band, dz 0.01 -> 0.0025)")
        ref = df_ok[df_ok['refined']]
        if len(ref) == 0:
            print("  no refined rows in this dataset.")
        else:
            rel = (ref['D_rem'] - ref['D_rem_coarse']).abs() / ref['D_rem_coarse'].clip(lower=1e-6)
            print(f"  n={len(ref)}  median |Δ|/coarse = {rel.median()*100:.1f}%  "
                  f"90th pct = {rel.quantile(0.9)*100:.1f}%  max = {rel.max()*100:.1f}%")
            flips_to_no_melt = int((ref['D_rem'] < no_melt_thr).sum())
            flips_above_band = int((ref['D_rem'] >= resolved_thr).sum())
            print(f"  reclassified coarse-shallow -> no_melt:        {flips_to_no_melt} "
                  f"({100*flips_to_no_melt/len(ref):.1f}% of refined)")
            print(f"  moved above the refinement band (D_rem >= {resolved_thr}): {flips_above_band} "
                  f"({100*flips_above_band/len(ref):.1f}% of refined)")
            print("  Interpretation: a high reclassification rate here means dz_coarse=0.01 "
                  "is NOT resolving the shallow band on its own — the inline refinement step "
                  "is load-bearing, not a formality. A near-zero rate would mean the band "
                  "could be widened or refinement skipped for speed. Note this boundary is "
                  "purely about which D_rem values need a finer grid to trust — see 2.3 for "
                  "whether it also marks any change in the underlying D_rem(groups) function "
                  "(it doesn't).")
            report['refinement'] = {
                'n_refined': len(ref), 'median_rel_change': float(rel.median()),
                'p90_rel_change': float(rel.quantile(0.9)),
                'n_flip_to_no_melt': flips_to_no_melt, 'n_flip_above_band': flips_above_band,
            }
    else:
        print("\n[2.1] no 'refined' column — skipped.")

    # ── Sizing-adequacy risk map, by (r, theta) bin ───────────────────────────
    if _P is not None and _tau_dev is not None:
        print("\n[2.2] Sizing-adequacy risk by (r, theta_liquidus) bin, all melted cases")
        melted = classify_melt(df_ok['D_rem'], no_melt_thr)
        resolved = df_ok[melted].copy()
        if len(resolved) and 'r' in resolved and 'theta_liquidus' in resolved:
            resolved['tau_dev'] = [
                _tau_dev(r, tm) for r, tm in zip(resolved['r'], resolved['theta_liquidus'])
            ]
            finite = resolved['tau_dev'].notna()
            resolved.loc[finite, 'margin'] = (
                resolved.loc[finite, 't_total'] / (_P.tau_dev_mult * resolved.loc[finite, 'tau_dev'])
            )
            r_bins = pd.cut(resolved.loc[finite, 'r'], bins=[0, 0.5, 1, 1.5, 2.5, 10])
            # Same 1e-9 tolerance as Phase 0.7's criterion B and rerun_censored.py's
            # find_censored() -- margin recomputes to exactly 1.0 by construction for
            # any non-time-capped case, so floating-point noise alone can flip
            # "< 1.0" on a genuinely-adequate row sitting right at the boundary.
            risk = resolved.loc[finite].groupby(r_bins)['margin'].agg(
                ['count', lambda s: float((s < 1.0 - 1e-9).mean())])
            risk.columns = ['n', 'frac_below_safety_margin']
            print(risk.to_string())
            report['sizing_risk_by_r_bin'] = risk.reset_index().astype(str).to_dict('records')
        else:
            print("  required columns missing — skipped.")
    else:
        print("\n[2.2] grid.py not importable — skipped.")

    # ── Boundary-continuity check (the test that justified dropping the ─────
    #    three-population split — kept as a standing regression test) ───────
    print("\n[2.3] Boundary-continuity check at the old refinement-band edge "
          f"(D_rem={resolved_thr})")
    melted_mask = classify_melt(df_ok['D_rem'], no_melt_thr)
    melted_df = df_ok[melted_mask].copy()
    required_cols = {'theta_liquidus', 'E', 'C', 'r', 'ste_cold', 'ste_hot'} if model == 'pure' \
        else {'theta_liquidus', 'mushy_frac', 'k_p', 'E', 'C', 'r', 'ste_cold', 'ste_hot'}
    if len(melted_df) > 200 and required_cols.issubset(melted_df.columns):
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import KFold, cross_val_predict
        from scipy import stats as _stats

        X = _build_features(melted_df, model)
        y = np.log(melted_df['D_rem'].clip(lower=1e-4))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        pipe = make_pipeline(PolynomialFeatures(degree=4, include_bias=False),
                             StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 30)))
        pred = cross_val_predict(pipe, X, y, cv=kf)
        melted_df['resid'] = np.asarray(y) - pred

        # Symmetric window in log-ratio terms (0.8x / 1.25x are reciprocal),
        # NOT the whole shallow band vs a sliver above — comparing the full
        # band to a sliver is the wrong test (mismatched widths, and sensitive
        # to exactly where D_rem==resolved_thr cases that piled up on the
        # coarse grid get assigned). A proper Welch t-test on matched windows
        # is the real check.
        below = melted_df[(melted_df['D_rem'] >= resolved_thr * 0.8) & (melted_df['D_rem'] < resolved_thr)]
        above = melted_df[(melted_df['D_rem'] >= resolved_thr) & (melted_df['D_rem'] < resolved_thr * 1.25)]
        if len(below) <= 20 or len(above) <= 20:
            print(f"  too few cases near D_rem={resolved_thr} (n_below={len(below)}, n_above={len(above)}, "
                  f"need >20 each) — skipped. Re-run once more sweep data lands near this boundary.")
        if len(below) > 20 and len(above) > 20:
            ttest_result = _stats.ttest_ind(below['resid'], above['resid'], equal_var=False)
            t, p = float(ttest_result.statistic), float(ttest_result.pvalue)
            print(f"  matched-window Welch t-test: below n={len(below)} mean={below['resid'].mean():+.4f}  "
                  f"above n={len(above)} mean={above['resid'].mean():+.4f}   t={t:.2f}  p={p:.2e}")
            report['boundary_continuity'] = {
                'below_mean': float(below['resid'].mean()), 'above_mean': float(above['resid'].mean()),
                't': float(t), 'p': float(p),
            }
            if p < 0.01:
                # A real jump exists -- but is it a regime change in the
                # D_rem(groups) MAPPING, or a measurement-resolution artefact
                # of refinement correcting the shallow band downward while
                # the never-refined 'resolved' population stays uncorrected?
                # Isolate this by re-fitting the SAME model against the
                # ALWAYS-COARSE target (D_rem_coarse for every row, including
                # shallow-band rows pre-refinement) and re-measuring the jump
                # in the same windows. If refinement-asymmetry is the driver,
                # this should shrink (or flip sign on) the jump.
                if 'D_rem_coarse' in melted_df.columns:
                    yc = np.log(melted_df['D_rem_coarse'].clip(lower=1e-4))
                    predc = cross_val_predict(pipe, X, yc, cv=kf)
                    melted_df['resid_coarse'] = np.asarray(yc) - predc
                    belowc = melted_df[(melted_df['D_rem_coarse'] >= resolved_thr * 0.8) &
                                       (melted_df['D_rem_coarse'] < resolved_thr)]['resid_coarse']
                    abovec = melted_df[(melted_df['D_rem_coarse'] >= resolved_thr) &
                                       (melted_df['D_rem_coarse'] < resolved_thr * 1.25)]['resid_coarse']
                    jump_corrected = above['resid'].mean() - below['resid'].mean()
                    jump_coarse = abovec.mean() - belowc.mean() if len(belowc) > 20 and len(abovec) > 20 else None
                    print(f"  jump is significant (p<0.01). Attribution test: re-fit against the "
                          f"ALWAYS-COARSE target (D_rem_coarse, i.e. as if refinement never ran):")
                    if jump_coarse is not None:
                        print(f"    jump on refined-corrected target: {jump_corrected:+.4f}   "
                              f"jump on always-coarse target: {jump_coarse:+.4f}")
                        shrink = 1 - abs(jump_coarse) / abs(jump_corrected) if jump_corrected != 0 else 0
                        if shrink > 0.4 or np.sign(jump_coarse) != np.sign(jump_corrected):
                            ref_rows = melted_df[melted_df['refined'] & (melted_df['D_rem_coarse'] >= resolved_thr * 0.8)] \
                                if 'refined' in melted_df.columns else melted_df.iloc[0:0]
                            bias_str = (f"~{-((ref_rows['D_rem'] - ref_rows['D_rem_coarse']) / ref_rows['D_rem_coarse']).mean()*100:.1f}%"
                                       if len(ref_rows) > 5 else "an unquantified amount")
                            print(f"    -> the jump shrinks by {shrink*100:.0f}% (or reverses sign) once the "
                                  f"refinement correction is removed from BOTH sides. This is substantially "
                                  f"a COARSE-VS-FINE GRID MEASUREMENT BIAS, not a change in the underlying "
                                  f"D_rem(groups) function: near the trigger edge the coarse grid (dz=0.01h) "
                                  f"over-estimates D_rem by {bias_str} on average, which refinement corrects "
                                  f"downward for the shallow band but NOT for the never-refined 'resolved' "
                                  f"population just above it. A smaller residual jump persists even on the "
                                  f"always-coarse target, so the explanation isn't 100% clean — but this does "
                                  f"NOT reinstate the population split for modelling. It's a data-quality "
                                  f"argument for extending dz_fine refinement coverage above resolved_thr (or "
                                  f"a spot dz-convergence check there), independent of the fitting question.")
                        else:
                            print(f"    -> the jump does NOT shrink when the refinement correction is removed "
                                  f"— refinement asymmetry does not explain it. This is a more concerning "
                                  f"signal and is worth a closer look before trusting the combined fit "
                                  f"blindly in this region.")
                    else:
                        print("    too few always-coarse rows in the matched windows to compare — skipped.")
                else:
                    print("  ⚠ no 'D_rem_coarse' column to test attribution — can't rule out a genuine "
                          "regime change. Treat the combined fit's accuracy near this boundary with "
                          "caution until checked.")
            else:
                print("  no significant jump (p>=0.01) — consistent with one continuous function "
                      "across the old transient/resolved boundary.")
    else:
        print("  too few melted rows or missing columns — skipped.")

    print("\n[2.4] NOTE: this phase characterises the inline-refinement behaviour already "
          "present in the data. It is NOT a substitute for a true dz-sweep grid-convergence "
          "study (dz_convergence = (0.01, 0.005, 0.0025) in PARAMS) or the t_total/T-horizon "
          "study (PARAMS.horizons) — both are still open items (convergence_study.py rewrite, "
          "Q3 from the validation-questions list) and need their own dedicated runs.")

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Correlate
# ═══════════════════════════════════════════════════════════════════════════

def _build_features(df: pd.DataFrame, model: str) -> tuple:
    """Standard feature set: raw theta(+mushy_frac, k_p) + log material-property
    groups. This is the convention already locked in verification_plan_v4
    ('degree-4 polynomial on mixed inputs... outperforms raw cubic by ~0.054 R2'),
    reproduced here rather than re-invented."""
    feats: dict[str, Any]
    if model == 'pure':
        feats = {
            'theta_liquidus': df['theta_liquidus'],
            'logE': np.log(df['E']),
            'logC': np.log(df['C']),
            'logr': np.log(df['r']),
            'log_ste_cold': np.log(df['ste_cold']),
            'log_ste_hot': np.log(df['ste_hot']),
        }
    else:
        feats = {
            'theta_liquidus': df['theta_liquidus'],
            'mushy_frac': df['mushy_frac'],
            'k_p': df['k_p'],
            'logE': np.log(df['E']),
            'logC': np.log(df['C']),
            'logr': np.log(df['r']),
            'log_ste_cold': np.log(df['ste_cold']),
            'log_ste_hot': np.log(df['ste_hot']),
        }
    X = pd.DataFrame(feats)
    return X


def _theta_liquidus_from_groups(C, ste_cold, ste_hot):
    """theta_liquidus = ste_cold / (ste_cold + ste_hot/C) -- the same identity
    checked in Phase 0's algebraic-identity audit. Needed for partial-dependence
    queries where we vary C/ste_cold/ste_hot and must recompute theta_liquidus
    (and therefore margin) consistently rather than holding it fixed."""
    return ste_cold / (ste_cold + ste_hot / C)


def _compute_margin(df: pd.DataFrame) -> pd.Series:
    """theta_contact - theta_liquidus: the analytic E/(1+E) contact-temperature
    margin past the melt onset threshold. Single source of truth, shared by
    Phase 3 (classifier) and Phase 4 (equation extraction) -- same quantity,
    same formula, wherever it's used."""
    theta_contact = df['theta_contact'] if 'theta_contact' in df.columns else df['E'] / (1.0 + df['E'])
    return theta_contact - df['theta_liquidus']


def _build_features_margin(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Like _build_features, but replaces raw theta_liquidus with
    log(margin) = log(theta_contact - theta_liquidus) -- the SAME variable
    that governs the Part-1 onset classifier.

    This isn't just a cosmetic swap: on the legacy pure-metal data, the
    margin-based feature set gets a HIGHER R2_CV with FEWER sparse-regression
    terms than the raw-theta_liquidus feature set at every sparsity level
    tested (e.g. 9 terms / R2=0.992 vs 11 terms / R2=0.958). Margin isn't just
    the onset variable -- it's the leading-order depth variable too, which is
    the headline physical finding of Phase 4. See phase4_equation docstring.
    """
    margin = _compute_margin(df)
    base: dict[str, Any] = {
        'log_margin': np.log(margin),
        'logE': np.log(df['E']),
        'logC': np.log(df['C']),
        'logr': np.log(df['r']),
        'log_ste_cold': np.log(df['ste_cold']),
        'log_ste_hot': np.log(df['ste_hot']),
    }
    if model == 'scheil':
        base['mushy_frac'] = df['mushy_frac']
        base['k_p'] = df['k_p']
    return pd.DataFrame(base, index=df.index)


def _select_then_refit_cv(Xraw: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray],
                          alpha: float, n_splits: int = 5) -> dict:
    """Lasso for TERM SELECTION (which of the degree-4 polynomial terms
    survive at this sparsity budget), then an unregularised OLS refit on just
    the survivors (removes Lasso's shrinkage bias from the final coefficients
    -- this is the standard 'relaxed Lasso' trick and is what makes the
    final equation's coefficients honest rather than systematically shrunk).
    Cross-validated under ordinary KFold always, and GroupKFold-by-lhs_id
    when real grouping is available (selection is redone inside every fold,
    so there's no leakage from the term-selection step itself)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.model_selection import KFold, GroupKFold

    def _run(splitter, groups_arr=None):
        r2s, nsel = [], []
        for tr, te in splitter.split(Xraw, y, groups=groups_arr):
            sc = StandardScaler().fit(Xraw[tr])
            lasso = Lasso(alpha=alpha, max_iter=50000)
            lasso.fit(sc.transform(Xraw[tr]), y[tr])
            sel = np.abs(lasso.coef_) > 1e-8
            if sel.sum() == 0:
                sel[np.argmax(np.abs(lasso.coef_))] = True
            ols = LinearRegression().fit(Xraw[tr][:, sel], y[tr])
            pred = ols.predict(Xraw[te][:, sel])
            ss_res = np.sum((y[te] - pred) ** 2)
            ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
            r2s.append(1 - ss_res / ss_tot)
            nsel.append(int(sel.sum()))
        return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(nsel))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_kf_mean, r2_kf_std, n_terms = _run(kf)

    has_groups = _groupkfold_feasible(groups, n_splits=n_splits)
    if has_groups:
        gkf = GroupKFold(n_splits=n_splits)
        r2_gkf_mean, r2_gkf_std, _ = _run(gkf, groups_arr=groups)
    else:
        r2_gkf_mean, r2_gkf_std = float('nan'), float('nan')

    return dict(alpha=alpha, n_terms=n_terms,
                r2_kfold_mean=r2_kf_mean, r2_kfold_std=r2_kf_std,
                r2_groupkfold_mean=r2_gkf_mean, r2_groupkfold_std=r2_gkf_std,
                has_groups=has_groups)


def _final_equation(Xraw: np.ndarray, y: np.ndarray, alpha: float):
    """Fit the select-then-refit equation on the FULL dataset (point estimate
    for presentation -- the honest generalisation estimate is the CV result
    from _select_then_refit_cv at the same alpha, not this in-sample number)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso, LinearRegression

    sc = StandardScaler().fit(Xraw)
    lasso = Lasso(alpha=alpha, max_iter=50000)
    lasso.fit(sc.transform(Xraw), y)
    sel = np.abs(lasso.coef_) > 1e-8
    ols = LinearRegression().fit(Xraw[:, sel], y)
    pred = ols.predict(Xraw[:, sel])
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_insample = 1 - ss_res / ss_tot
    return sel, ols, float(r2_insample)


def _print_equation(names: np.ndarray, sel: np.ndarray, ols, target: str, r2_insample: float):
    print(f"  {int(sel.sum())} terms, in-sample R2={r2_insample:.4f} (fit on full dataset -- "
          f"see the CV row above for the honest generalisation estimate)")
    print(f"  log({target}) = {ols.intercept_:+.5f}")
    for nm, co in sorted(zip(names[sel], ols.coef_), key=lambda t: -abs(t[1])):
        print(f"    {co:+.5f} * {nm}")


def _partial_dependence_plot(df_melt: pd.DataFrame, model: str, poly, sel: np.ndarray, ols,
                             target: str, outdir: Path, fname: str):
    """Pick a real, comfortably-melted data point as the baseline operating
    point, then sweep each RAW dimensionless group one at a time (holding the
    others at the baseline), recomputing theta_liquidus/margin consistently
    as E/C/ste_cold/ste_hot change. This is the TOTAL physical effect of
    changing one input (both the 'via margin' and 'beyond margin' channels
    combined) -- the most assumption-light way to read physics out of the
    fitted equation, no coefficient-parsing required."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw_vars = ['E', 'C', 'r', 'ste_cold', 'ste_hot'] + (['mushy_frac', 'k_p'] if model == 'scheil' else [])

    med_val = df_melt[target].median()
    base_row = df_melt.iloc[(df_melt[target] - med_val).abs().argsort()[:1]].iloc[0]
    base = {v: float(base_row[v]) for v in raw_vars}

    ranges = {v: (df_melt[v].quantile(.02), df_melt[v].quantile(.98)) for v in raw_vars}

    fig, axes = plt.subplots(1, len(raw_vars), figsize=(3.4 * len(raw_vars), 3.3))
    axes = np.atleast_1d(axes)
    for ax, var in zip(axes, raw_vars):
        lo, hi = ranges[var]
        vals = np.linspace(lo, hi, 150)
        preds = []
        for v in vals:
            row = dict(base)
            row[var] = v
            theta_liq = _theta_liquidus_from_groups(row['C'], row['ste_cold'], row['ste_hot'])
            margin = row['E'] / (1.0 + row['E']) - theta_liq
            if margin <= 0:
                preds.append(np.nan)
                continue
            qdf = pd.DataFrame([{**row, 'theta_liquidus': theta_liq}])
            Xq = _build_features_margin(qdf, model)
            Xq_raw = poly.transform(Xq)
            preds.append(float(np.exp(ols.predict(Xq_raw[:, sel])[0])))
        vals_arr, preds_arr = np.array(vals), np.array(preds)
        ok = np.isfinite(preds_arr)
        ax.plot(vals_arr[ok], preds_arr[ok], color='#cc3b3b')
        ax.axvline(base[var], color='gray', lw=0.7, ls=':')
        ax.set_xlabel(var)
        if var == raw_vars[0]:
            ax.set_ylabel(f'predicted {target}')
        ax.set_title(var, fontsize=10)
    fig.suptitle(f'{target}: partial dependence (others at a representative melted point)', y=1.04)
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=140, bbox_inches='tight')
    plt.close(fig)


def phase4_equation(df_ok: pd.DataFrame, model: str, outdir: Path,
                    no_melt_thr=0.005, make_plots=True,
                    alphas=(0.1, 0.06, 0.03, 0.01, 0.006, 0.003),
                    target_terms=15) -> dict:
    """Extract a genuinely USEABLE equation for D_rem (and D_sol for Scheil)
    as a function of the dimensionless groups -- not just a high CV score.

    The locked degree-4 RidgeCV approach (Phase 3) scores R2~0.999 but keeps
    ~150+ of 209 polynomial terms (Ridge shrinks, it doesn't select) -- not
    something you'd write in a paper. This phase does Lasso-for-selection +
    OLS-refit ('relaxed Lasso') to get a short, honest-coefficient equation,
    swept across a sparsity budget so the terms-vs-R2 trade-off is visible
    rather than picked by fiat.

    It also uses log(margin) = log(theta_contact - theta_liquidus) in place of
    raw theta_liquidus -- empirically this gets a HIGHER R2 with FEWER terms
    at every sparsity level tested, because margin (the Part-1 onset variable)
    turns out to also be the leading-order Part-2 depth variable: D_rem scales
    close to linearly with margin, modulated by power laws in E/C/r/ste_hot.
    That's the headline physical reading this phase is built to produce.
    """
    print("\n" + "═" * 78)
    print("  PHASE 4 — EQUATION (useable D_rem / D_sol correlations)")
    print("═" * 78)
    report = {}

    melted = classify_melt(df_ok['D_rem'], no_melt_thr)
    df_melt = df_ok[melted].copy()

    # The analytic margin criterion isn't a perfect classifier (AUC~0.999, not
    # 1.0 -- see Phase 3) -- a small number of FD-melted cases can have
    # margin<=0, where log(margin) is undefined. Drop them here rather than
    # letting them crash the polynomial expansion; report how many, since
    # that count is itself informative (it's the same disagreement Phase 3's
    # classifier AUC is already quantifying, seen from the other side).
    margin_melt = _compute_margin(df_melt)
    bad_margin = ~(margin_melt > 0)   # catches negative, zero, AND NaN (NaN<=0 is False, would slip through)
    if bad_margin.any():
        print(f"  ⚠ dropping {int(bad_margin.sum())}/{len(df_melt)} melted cases with "
              f"analytic margin<=0 (log(margin) undefined there) -- these are the same "
              f"FD-vs-analytic disagreements the Phase 3 classifier AUC gap reflects.")
        df_melt = df_melt[~bad_margin].copy()

    if 'lhs_id' in df_melt.columns:
        groups = df_melt['lhs_id'].values
        has_groups = _groupkfold_feasible(groups)
    else:
        groups, has_groups = None, False
    if not has_groups:
        print("  ⚠ no real lhs_id grouping in this data -- GroupKFold columns below will be "
              "NaN. Re-run on real sweep output (which has lhs_id) before trusting these "
              "equations to generalise to unseen materials, not just unseen operating points.")

    targets = ['D_rem'] if model == 'pure' else ['D_rem', 'D_sol']
    X = _build_features_margin(df_melt, model)
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=4, include_bias=False)
    Xraw = poly.fit_transform(X)
    names = poly.get_feature_names_out(X.columns)
    print(f"\n  feature basis: {list(X.columns)}  ->  degree-4 expansion = {Xraw.shape[1]} candidate terms")

    for target in targets:
        print(f"\n[4.{target}] sparsity sweep, n={len(df_melt)}")
        y = np.asarray(np.log(df_melt[target].clip(lower=1e-4)))
        sweep = []
        for alpha in alphas:
            res = _select_then_refit_cv(Xraw, y, groups, alpha)
            sweep.append(res)
            gkf_str = f"{res['r2_groupkfold_mean']:.4f}" if has_groups else "n/a"
            print(f"  alpha={alpha:<6} n_terms~{res['n_terms']:4.1f}   "
                  f"R2_CV(KFold)={res['r2_kfold_mean']:.4f}   R2_CV(GroupKFold)={gkf_str}")

        chosen = min(sweep, key=lambda r: abs(r['n_terms'] - target_terms))
        sel, ols, r2_insample = _final_equation(Xraw, y, chosen['alpha'])
        print(f"\n  -> selected alpha={chosen['alpha']} (closest to target_terms={target_terms})")
        _print_equation(names, sel, ols, target, r2_insample)

        if make_plots:
            fname = f"07_{target}_partial_dependence.png"
            _partial_dependence_plot(df_melt, model, poly, sel, ols, target, outdir, fname)
            print(f"  partial-dependence plot -> {outdir / fname}")

        report[target] = {
            'sparsity_sweep': sweep,
            'chosen_alpha': chosen['alpha'],
            'chosen_n_terms': int(sel.sum()),
            'r2_insample_full_data': r2_insample,
            'r2_cv_at_chosen_alpha': chosen['r2_groupkfold_mean'] if has_groups else chosen['r2_kfold_mean'],
            'intercept': float(ols.intercept_),
            'equation': {str(nm): float(co) for nm, co in zip(names[sel], ols.coef_)},
        }

    return report


def _groupkfold_feasible(groups, n_splits: int = 5) -> bool:
    """Whether GroupKFold(n_splits) can be CONSTRUCTED on these groups without
    raising -- needs >= n_splits distinct groups. This is a different
    question from whether the grouping is MEANINGFUL (every row its own
    group also satisfies this if there are >= n_splits rows, but that's a
    degenerate split, not a real material holdout -- that case is warned
    about separately at each call site, not handled here)."""
    if groups is None:
        return False
    n_unique = len(np.unique(np.asarray(groups)))
    return n_unique >= n_splits


def _fit_and_score(X, y, groups, label):
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold, GroupKFold, cross_val_score

    if groups.nunique() == len(groups):
        print(f"  [{label}] ⚠ no real lhs_id grouping in this data (every row is its own "
              f"group) — GroupKFold below degenerates to an unshuffled row split and is "
              f"NOT a meaningful material-holdout test. Re-run on pipeline output with a "
              f"real lhs_id column to get that check.")

    can_group = _groupkfold_feasible(groups.values)
    if not can_group:
        print(f"  [{label}] ⚠ only {groups.nunique()} unique groups -- GroupKFold needs >= 5. "
              f"Skipping GroupKFold, KFold-only below.")

    results = {}
    alphas = np.logspace(-4, 3, 30)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5) if can_group else None

    def _gkf_score(estimator):
        if not can_group:
            return np.array([np.nan])
        return cross_val_score(estimator, X, y, cv=gkf, groups=groups, scoring='r2')

    for degree in (2, 4):
        pipe = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),
                             StandardScaler(), RidgeCV(alphas=alphas))
        r2_kf = cross_val_score(pipe, X, y, cv=kf, scoring='r2')
        r2_gkf = _gkf_score(pipe)
        gkf_str = f"{r2_gkf.mean():.4f} (±{r2_gkf.std():.4f})" if can_group else "n/a"
        print(f"  [{label}] degree-{degree} RidgeCV   "
              f"R2_CV(KFold)={r2_kf.mean():.4f} (±{r2_kf.std():.4f})   "
              f"R2_CV(GroupKFold by lhs_id)={gkf_str}")
        results[f'degree{degree}'] = {
            'r2_kfold_mean': float(r2_kf.mean()), 'r2_kfold_std': float(r2_kf.std()),
            'r2_groupkfold_mean': float(r2_gkf.mean()), 'r2_groupkfold_std': float(r2_gkf.std()),
        }

    rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    r2_kf = cross_val_score(rf, X, y, cv=kf, scoring='r2')
    r2_gkf = _gkf_score(rf)
    gkf_str = f"{r2_gkf.mean():.4f} (±{r2_gkf.std():.4f})" if can_group else "n/a"
    print(f"  [{label}] RandomForest        "
          f"R2_CV(KFold)={r2_kf.mean():.4f} (±{r2_kf.std():.4f})   "
          f"R2_CV(GroupKFold by lhs_id)={gkf_str}")
    rf.fit(X, y)
    importances = dict(sorted(zip(X.columns, rf.feature_importances_),
                              key=lambda kv: -kv[1]))
    print(f"  [{label}] RF importances: " +
          ", ".join(f"{k}={v:.3f}" for k, v in importances.items()))
    results['random_forest'] = {
        'r2_kfold_mean': float(r2_kf.mean()), 'r2_kfold_std': float(r2_kf.std()),
        'r2_groupkfold_mean': float(r2_gkf.mean()), 'r2_groupkfold_std': float(r2_gkf.std()),
        'importances': {k: float(v) for k, v in importances.items()},
    }
    if can_group and abs(results['degree4']['r2_kfold_mean'] - results['degree4']['r2_groupkfold_mean']) > 0.02:
        print(f"  ⚠ [{label}] KFold and GroupKFold R2 differ by "
              f">{abs(results['degree4']['r2_kfold_mean']-results['degree4']['r2_groupkfold_mean']):.3f} "
              f"— ordinary K-fold may be optimistic about generalising to unseen materials.")
    return results


def _fit_classifier(X, y_bool, groups, margin, label):
    """Melt/no-melt classifier, benchmarked against the analytic contact-
    criterion margin (theta_contact - theta_liquidus), which needs no fitting
    at all — it's the thing being tested against."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold, GroupKFold, cross_val_score
    from sklearn.metrics import roc_auc_score

    results = {}
    auc_analytic = roc_auc_score(y_bool, margin)
    print(f"  [{label}] analytic baseline  AUC(E/(1+E) - theta_liquidus) = {auc_analytic:.4f}  "
          f"(no fitting — this is the physics-only criterion)")
    results['analytic_auc'] = float(auc_analytic)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    can_group = _groupkfold_feasible(groups)
    gkf = GroupKFold(n_splits=5) if can_group else None
    if not can_group:
        n_unique = len(np.unique(np.asarray(groups))) if groups is not None else 0
        print(f"  [{label}] ⚠ only {n_unique} unique groups -- GroupKFold needs >= 5. "
              f"Skipping GroupKFold, KFold-only below.")

    def _gkf_score(estimator, X_, y_):
        if not can_group:
            return np.array([np.nan])
        return cross_val_score(estimator, X_, y_, cv=gkf, groups=groups, scoring='roc_auc')

    logit = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(),
                          LogisticRegressionCV(Cs=np.logspace(-3, 3, 15), max_iter=5000, cv=3))
    auc_kf = cross_val_score(logit, X, y_bool, cv=kf, scoring='roc_auc')
    auc_gkf = _gkf_score(logit, X, y_bool)
    gkf_str = f"{auc_gkf.mean():.4f} (±{auc_gkf.std():.4f})" if can_group else "n/a"
    print(f"  [{label}] logistic (deg-2)   AUC(KFold)={auc_kf.mean():.4f} (±{auc_kf.std():.4f})   "
          f"AUC(GroupKFold)={gkf_str}")
    results['logistic'] = {'auc_kfold_mean': float(auc_kf.mean()), 'auc_kfold_std': float(auc_kf.std()),
                           'auc_groupkfold_mean': float(auc_gkf.mean()), 'auc_groupkfold_std': float(auc_gkf.std())}

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    auc_kf = cross_val_score(rf, X, y_bool, cv=kf, scoring='roc_auc')
    auc_gkf = _gkf_score(rf, X, y_bool)
    gkf_str = f"{auc_gkf.mean():.4f} (±{auc_gkf.std():.4f})" if can_group else "n/a"
    print(f"  [{label}] RandomForest        AUC(KFold)={auc_kf.mean():.4f} (±{auc_kf.std():.4f})   "
          f"AUC(GroupKFold)={gkf_str}")
    rf.fit(X, y_bool)
    importances = dict(sorted(zip(X.columns, rf.feature_importances_), key=lambda kv: -kv[1]))
    print(f"  [{label}] RF importances: " + ", ".join(f"{k}={v:.3f}" for k, v in importances.items()))
    results['random_forest'] = {'auc_kfold_mean': float(auc_kf.mean()), 'auc_kfold_std': float(auc_kf.std()),
                                'auc_groupkfold_mean': float(auc_gkf.mean()), 'auc_groupkfold_std': float(auc_gkf.std()),
                                'importances': {k: float(v) for k, v in importances.items()}}

    best_fitted = max(results['logistic']['auc_kfold_mean'], results['random_forest']['auc_kfold_mean'])
    if best_fitted - auc_analytic < 0.005:
        print(f"  -> a fitted classifier adds essentially nothing over the analytic criterion "
              f"alone (Δ={best_fitted-auc_analytic:+.4f}). The contact-temperature criterion is "
              f"already a near-complete description of onset; lead with it in the paper, not a "
              f"black-box classifier.")
    return results


def phase3_correlate(df_ok: pd.DataFrame, model: str,
                      no_melt_thr=0.005, resolved_thr=0.10) -> dict:
    print("\n" + "═" * 78)
    print("  PHASE 3 — CORRELATE")
    print("═" * 78)
    report = {}

    melted = classify_melt(df_ok['D_rem'], no_melt_thr)
    groups_all = df_ok['lhs_id'] if 'lhs_id' in df_ok.columns else pd.Series(np.arange(len(df_ok)), index=df_ok.index)

    # ── 3a. Melt / no-melt classifier ────────────────────────────────────────
    print(f"\n[3a] Melt / no-melt classifier   n={len(df_ok)}  "
          f"({int(melted.sum())} melted / {int((~melted).sum())} no-melt)")
    X_cls = _build_features(df_ok, model)
    if 'theta_contact' in df_ok.columns:
        margin = df_ok['theta_contact'] - df_ok['theta_liquidus']
    else:
        margin = df_ok['E'] / (1.0 + df_ok['E']) - df_ok['theta_liquidus']
    report['classifier'] = _fit_classifier(X_cls, melted, groups_all, margin, label='melt_classifier')

    # ── 3b. Single combined regression, all melted cases, no further split ──
    targets = ['D_rem'] if model == 'pure' else ['D_rem', 'D_sol']
    df_melt = df_ok[melted].copy()
    groups_melt = df_melt['lhs_id'] if 'lhs_id' in df_melt.columns else pd.Series(np.arange(len(df_melt)), index=df_melt.index)

    for target in targets:
        print(f"\n[3b] {target}, single combined fit across all melted cases  n={len(df_melt)}")
        X = _build_features(df_melt, model)
        y = np.log(df_melt[target].clip(lower=1e-4))   # log-target: depth spans orders of magnitude
        res = _fit_and_score(X, y, groups_melt, label=target)
        report[f'{target}_combined'] = res

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — epsilon (Poirier & Geiger block-decay validity horizon)
# ═══════════════════════════════════════════════════════════════════════════

def _bisect_epsilon(lam: float, r: float, s2fd: float):
    """Find eps in (0,1) such that lam / (2*sqrt(r)*erfinv(eps)) == s2fd.

    Physical content (derived from Poirier & Geiger, Transport Phenomena in
    Materials Processing, eq. 9.51 -- the temperature at the centre of a slab
    of thickness h, initially at uniform T_i, decaying by 1-D diffusion):

        theta_centre(t) = T_i * erf(h / (4*sqrt(alpha*t)))

    With T_i = theta_deposit = 1 (confirmed -- the deposit's own initial
    temperature, not theta_contact) and alpha = the deposit's own diffusivity
    (so alpha/alpha_s = r), nondimensionalising by the project's standard
    t_tilde = alpha_s*t/h^2 gives:

        theta_centre(t_tilde) = erf(1 / (4*sqrt(r*t_tilde)))                  (*)

    Separately, lam (lam_neumann for pure / lam_liq, lam_sol for Scheil) is
    the Stefan-similarity eigenvalue, and the front sits at the SUBSTRATE-
    referenced similarity variable eta=x_tilde/(2*sqrt(t_tilde)) = lam, i.e.
    the front position is x_tilde_front = 2*lam*sqrt(t_tilde) -- this was
    derived (not assumed) from solve_stefan_similarity's own zone definitions
    in analytical.py (liquid zone uses eta/sqrt(r), solid zone uses eta
    directly, which is only consistent if eta is substrate-referenced).

    Substituting t_tilde = (s2fd/(2*lam))^2 (i.e. "the time at which the
    front, moving per the Neumann formula, would have reached the FD's
    actual depth s2fd") into (*) gives EXACTLY this function's equation,
    verified symbolically (difference == 0). So eps_opt returned here is
    theta_centre at the time implied by FD's own result -- the natural
    comparison is against theta_liquidus (theta_solidus for the Scheil
    solidus front): does the block's own heat reservoir, by the time FD
    says remelting stopped advancing, sit AT the threshold where melting
    should stall? On legacy pure-metal data this gives corr=0.80 against
    theta_liquidus (vs -0.14 against theta_contact, which is NOT the right
    comparison) with a depth-dependent residual (eps_opt undershoots
    theta_liquidus for shallow D_rem, overshoots for deep D_rem) -- plausibly
    because the symmetric two-sided block-decay picture loses heat in both
    directions while the real deposit only loses heat downward (adiabatic
    top), but that mechanism is a hypothesis, not yet verified independently.

    Returns (eps_opt, status_string).
    """
    from scipy.special import erfinv as _erfinv
    from scipy.optimize import brentq

    if not (np.isfinite(lam) and lam > 0 and np.isfinite(s2fd) and s2fd > 0):
        return np.nan, 'invalid_input'

    def _f(eps):
        ex = float(_erfinv(eps))
        if ex < 1e-15:
            return 1e8
        return lam / (2.0 * np.sqrt(r) * ex) - s2fd

    eps_lo, eps_hi = 1e-9, 1.0 - 1e-9
    f_lo, f_hi = _f(eps_lo), _f(eps_hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return np.nan, 'f_not_finite'
    if f_lo * f_hi > 0:
        return np.nan, 's2fd_below_range' if f_hi > 0 else 's2fd_above_range'
    try:
        eps_opt = float(brentq(_f, eps_lo, eps_hi, xtol=1e-10, rtol=1e-10))
        return eps_opt, 'converged'
    except Exception as exc:
        return np.nan, f'brentq_failed:{exc}'


def _epsilon_plot(sub: pd.DataFrame, target: str, theta_col: str, outdir: Path, fname: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    conv = sub[sub['status'] == 'converged'].copy()
    conv['_gap'] = conv['eps_opt'] - conv[theta_col]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

    axes[0].scatter(conv[theta_col], conv['eps_opt'], s=3, alpha=0.2, color='#cc3b3b')
    lims = [0, max(conv[theta_col].max(), conv['eps_opt'].max())]
    axes[0].plot(lims, lims, 'k--', lw=1)
    axes[0].set_xlabel(theta_col)
    axes[0].set_ylabel('eps_opt')
    axes[0].set_title(f'{target}: eps_opt vs {theta_col}')

    bins = np.unique(np.quantile(conv[target], np.linspace(0, 1, 9)))
    if len(bins) > 2:
        conv['bin'] = pd.cut(conv[target], bins=bins)
        # Aggregate a single precomputed column directly -- avoids
        # .groupby().apply(lambda d: ..., include_groups=False), which only
        # exists on pandas>=2.2 and is unnecessary complexity for a plain
        # per-group median anyway.
        resid_by_bin = conv.groupby('bin', observed=True)['_gap'].median()
        resid_by_bin.plot(kind='bar', ax=axes[1], color='#1f6fb4')
    axes[1].set_ylabel(f'median(eps_opt - {theta_col})')
    axes[1].set_title(f'{target}: residual vs depth bin')
    axes[1].axhline(0, color='k', lw=0.7)
    plt.setp(axes[1].get_xticklabels(), rotation=40, ha='right')
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=140)
    plt.close(fig)


def _eps_true_plot(sub_true: pd.DataFrame, target: str, theta_col: str, t_col: str,
                   outdir: Path, fname: str, overlap_eps_opt: Optional[pd.Series] = None):
    """Same style as _epsilon_plot, but for eps_true (computed directly from the
    real recorded critical time, not the implied time eps_opt used). If
    overlap_eps_opt is given (eps_opt values on the same index as a subset of
    sub_true), adds a third panel showing the two methods head-to-head on the
    exact same rows -- the direct visual answer to "how much did the old
    implied-time confound actually move the number"."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    has_overlap = overlap_eps_opt is not None and len(overlap_eps_opt) >= 5
    n_panels = 3 if has_overlap else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.3))

    sub_true = sub_true.copy()
    sub_true['_gap'] = sub_true['eps_true'] - sub_true[theta_col]

    axes[0].scatter(sub_true[theta_col], sub_true['eps_true'], s=6, alpha=0.35, color='#2c8a3d')
    lims = [0, max(sub_true[theta_col].max(), sub_true['eps_true'].max())]
    axes[0].plot(lims, lims, 'k--', lw=1)
    axes[0].set_xlabel(theta_col)
    axes[0].set_ylabel('eps_true')
    axes[0].set_title(f'{target}: eps_true (real t) vs {theta_col}')

    bins = np.unique(np.quantile(sub_true[target], np.linspace(0, 1, 9)))
    if len(bins) > 2:
        sub_true['bin'] = pd.cut(sub_true[target], bins=bins)
        resid_by_bin = sub_true.groupby('bin', observed=True)['_gap'].median()
        resid_by_bin.plot(kind='bar', ax=axes[1], color='#2c8a3d')
    axes[1].set_ylabel(f'median(eps_true - {theta_col})')
    axes[1].set_title(f'{target}: residual vs depth bin (real t)')
    axes[1].axhline(0, color='k', lw=0.7)
    plt.setp(axes[1].get_xticklabels(), rotation=40, ha='right')

    if has_overlap:
        eps_true_ov = sub_true.loc[overlap_eps_opt.index, 'eps_true']
        axes[2].scatter(eps_true_ov, overlap_eps_opt, s=10, alpha=0.5, color='#b4581f')
        lims2 = [0, max(eps_true_ov.max(), overlap_eps_opt.max())]
        axes[2].plot(lims2, lims2, 'k--', lw=1)
        axes[2].set_xlabel('eps_true (real t, direct)')
        axes[2].set_ylabel('eps_opt (implied t, old method)')
        axes[2].set_title(f'{target}: old vs new, same {len(overlap_eps_opt)} rows')

    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=140)
    plt.close(fig)


def phase5_epsilon(df_ok: pd.DataFrame, model: str, outdir: Path,
                   no_melt_thr=0.005, make_plots=True) -> dict:
    """Re-incorporates the eps_opt / 'P&G validity horizon' diagnostic
    (originally in analytical.py's _bisect_epsilon, used standalone in past
    sessions, never wired into this pipeline). See _bisect_epsilon's
    docstring for the full derivation and the corrected comparison target
    (theta_liquidus / theta_solidus, NOT theta_contact).

    ALSO computes eps_true directly from t_at_max_remelt / t_at_max_solidus
    where available -- the real recorded time the FD solver reached its
    peak depth, added to run_one()'s ROW_SCHEMA specifically to remove a
    confound in eps_opt: eps_opt is theta_centre evaluated at an IMPLIED
    time backed out by inverting the front-position formula x=2*lam*sqrt(t)
    against the FD's achieved depth, not the real elapsed time. Checked
    directly against a handful of real cases: the ratio of implied-to-real
    time ranged from ~0.3 (low theta_melt, deep cases) to ~1.4 (high
    theta_melt, shallow cases) -- a sign flip that lines up disturbingly
    well with the depth-dependent residual eps_opt has always shown
    (undershoots theta_liquidus shallow, overshoots deep). eps_true needs
    no eigenvalue at all (no lam, no bisection) -- just r and the real
    recorded time, so it's both simpler and not subject to the lam_liq
    degeneracy bug either.

    eps_true will only be available on rows that have gone through the
    patched sweep scripts (freshly re-run rows) until a full fresh sweep
    happens -- this function reports whatever n that actually is rather
    than assuming full coverage, and where the old (eps_opt) and new
    (eps_true) values overlap on the same rows, reports the two head-to-
    head so you can see directly how much of the old residual pattern
    survives once the confound is removed.
    """
    from scipy.special import erf
    print("\n" + "═" * 78)
    print("  PHASE 5 — EPSILON (Poirier & Geiger block-decay validity horizon)")
    print("═" * 78)
    report = {}

    melted = classify_melt(df_ok['D_rem'], no_melt_thr)
    df_melt = df_ok[melted].copy()

    if model == 'pure':
        targets = [('D_rem', 'lam_neumann', 'theta_liquidus', 't_at_max_remelt')]
    else:
        targets = [('D_rem', 'lam_liq', 'theta_liquidus', 't_at_max_remelt'),
                  ('D_sol', 'lam_sol', 'theta_solidus', 't_at_max_solidus')]

    for target, lam_col, theta_col, t_col in targets:
        print(f"\n[5.{target}] using {lam_col} as the Stefan-similarity eigenvalue")
        report[target] = {}
        conv = pd.DataFrame()

        if lam_col not in df_melt.columns or theta_col not in df_melt.columns:
            print(f"  ⚠ missing column ({lam_col} or {theta_col}) -- skipped old method.")
        else:
            sub = df_melt[[target, lam_col, 'r', theta_col]].dropna().copy()
            out = [_bisect_epsilon(row[lam_col], row['r'], row[target]) for _, row in sub.iterrows()]
            sub['eps_opt'] = [o[0] for o in out]
            sub['status'] = [o[1] for o in out]

            status_counts = sub['status'].value_counts()
            print(f"  n={len(sub)}  " + ", ".join(f"{k}={v}" for k, v in status_counts.items()))

            conv = sub[sub['status'] == 'converged']
            if len(conv) < 10:
                print("  too few converged rows -- skipped old method.")
            else:
                err = conv['eps_opt'] - conv[theta_col]
                corr = float(np.corrcoef(conv['eps_opt'], conv[theta_col])[0, 1])
                print(f"  [old, implied-time] corr(eps_opt, {theta_col}) = {corr:.4f}")
                print(f"  median(eps_opt - {theta_col}) = {err.median():+.4f}   "
                      f"median|eps_opt - {theta_col}| = {err.abs().median():.4f}   "
                      f"median relative = {(err.abs()/conv[theta_col]).median()*100:.1f}%")

                if make_plots:
                    fname = f"08_{target}_epsilon.png"
                    _epsilon_plot(sub, target, theta_col, outdir, fname)
                    print(f"  plot -> {outdir / fname}")

                report[target]['eps_opt'] = {
                    'n_total': int(len(sub)),
                    'status_counts': {str(k): int(v) for k, v in status_counts.items()},
                    'corr': corr,
                    'median_signed_error': float(err.median()),
                    'median_abs_error': float(err.abs().median()),
                    'median_relative_error': float((err.abs() / conv[theta_col]).median()),
                }

        # ── eps_true: direct, real-time computation ─────────────────────────
        if t_col not in df_melt.columns:
            print(f"\n  ⚠ {t_col} not in this CSV -- added in the run_one()/ROW_SCHEMA "
                  f"patch for recording the real critical time; only freshly re-run "
                  f"rows will have it. Skipping the corrected (real-time) comparison "
                  f"until more rows have been re-run.")
            continue

        sub_true = df_melt[[target, 'r', theta_col, t_col]].dropna().copy()
        n_true = len(sub_true)
        if n_true < 10:
            print(f"\n  ⚠ only {n_true} row(s) have {t_col} recorded -- too few for a "
                  f"meaningful comparison yet (grows as more rows get re-run through "
                  f"the patched sweep scripts).")
            continue

        sub_true['eps_true'] = erf(1.0 / (4.0 * np.sqrt(sub_true['r'] * sub_true[t_col])))
        err_true = sub_true['eps_true'] - sub_true[theta_col]
        corr_true = float(np.corrcoef(sub_true['eps_true'], sub_true[theta_col])[0, 1])
        print(f"\n  [new, real time -- no eigenvalue needed] n={n_true}")
        print(f"  corr(eps_true, {theta_col}) = {corr_true:.4f}")
        print(f"  median(eps_true - {theta_col}) = {err_true.median():+.4f}   "
              f"median|eps_true - {theta_col}| = {err_true.abs().median():.4f}   "
              f"median relative = {(err_true.abs()/sub_true[theta_col]).median()*100:.1f}%")

        report[target]['eps_true'] = {
            'n_total': n_true,
            'corr': corr_true,
            'median_signed_error': float(err_true.median()),
            'median_abs_error': float(err_true.abs().median()),
            'median_relative_error': float((err_true.abs() / sub_true[theta_col]).median()),
        }

        # Head-to-head on whatever rows have BOTH -- the direct answer to how
        # much of the old residual pattern was the implied-time confound.
        overlap_idx = sub_true.index.intersection(conv.index) if len(conv) else pd.Index([])
        overlap_eps_opt = None
        if len(overlap_idx) >= 5:
            overlap_eps_opt = conv.loc[overlap_idx, 'eps_opt']
            eps_true_ov = sub_true.loc[overlap_idx, 'eps_true']
            diff = overlap_eps_opt - eps_true_ov
            corr_methods = float(np.corrcoef(overlap_eps_opt, eps_true_ov)[0, 1])
            print(f"\n  [old vs new, same {len(overlap_idx)} rows]  "
                  f"corr(eps_opt, eps_true) = {corr_methods:.4f}   "
                  f"median(eps_opt - eps_true) = {diff.median():+.4f}   "
                  f"median|eps_opt - eps_true| = {diff.abs().median():.4f}")
            report[target]['old_vs_new'] = {
                'n_overlap': int(len(overlap_idx)),
                'corr': corr_methods,
                'median_diff': float(diff.median()),
                'median_abs_diff': float(diff.abs().median()),
            }
        else:
            print(f"\n  (only {len(overlap_idx)} row(s) have both eps_opt and eps_true -- "
                  f"too few yet to compare the two methods directly)")

        if make_plots:
            fname = f"09_{target}_epsilon_true.png"
            _eps_true_plot(sub_true, target, theta_col, t_col, outdir, fname,
                           overlap_eps_opt=overlap_eps_opt)
            print(f"  plot -> {outdir / fname}")

    return report




def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model', choices=['pure', 'scheil'], required=True)
    ap.add_argument('--legacy', action='store_true',
                    help='Input is the pre-pipeline results_refined.csv schema')
    ap.add_argument('--outdir', default='analysis_out')
    ap.add_argument('--phases', default='0,1,2,3,4,5',
                    help='Comma-separated phase numbers to run, e.g. "0,1"')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--no-melt-thr', type=float, default=0.005,
                    help='D_rem below this = no melt (the only classification boundary used for modelling)')
    ap.add_argument('--resolved-thr', type=float, default=0.10,
                    help='upper edge of the inline dz_fine refinement-trigger band (numerical '
                         'bookkeeping only, not a modelling population boundary)')
    ap.add_argument('--eq-target-terms', type=int, default=15,
                    help='Phase 4: pick the sparsity level whose term count is closest to this, '
                         'for the headline equation + partial-dependence plot (the full terms-vs-R2 '
                         'sweep is always printed regardless, so this only picks which one gets the '
                         'detailed printout and plot)')
    args = ap.parse_args()

    phases = set(int(p) for p in args.phases.split(','))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv, args.model, args.legacy)
    df_ok = df[df['status'] == 'ok'].copy()

    full_report = {'csv': args.csv, 'model': args.model, 'legacy': args.legacy}

    if 0 in phases:
        full_report['phase0'] = phase0_integrity(
            df, args.model, args.legacy, args.no_melt_thr, args.resolved_thr)
    if 1 in phases and not args.no_plots:
        phase1_visualize(df_ok, args.model, outdir, args.no_melt_thr, args.resolved_thr)
    if 2 in phases:
        full_report['phase2'] = phase2_convergence(
            df_ok, args.model, args.no_melt_thr, args.resolved_thr)
    if 3 in phases:
        full_report['phase3'] = phase3_correlate(
            df_ok, args.model, args.no_melt_thr, args.resolved_thr)
    if 4 in phases:
        full_report['phase4'] = phase4_equation(
            df_ok, args.model, outdir, args.no_melt_thr,
            make_plots=not args.no_plots, target_terms=args.eq_target_terms)
    if 5 in phases:
        full_report['phase5'] = phase5_epsilon(
            df_ok, args.model, outdir, args.no_melt_thr, make_plots=not args.no_plots)

    report_path = outdir / 'report.json'
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nFull report written to {report_path.resolve()}")


if __name__ == '__main__':
    main()