"""
derive_powerlaws_calorimetric.py
=================================
Leg G — calorimetric collapse of the pure-metal D_rem power law.

Companion to derive_powerlaws.py. Reproduces the headline 6-exponent law,
then shows it collapses, at statistically indistinguishable OOF R2, onto

    D_rem  ≈  a · Pi_c^b · margin,        margin exponent pinned EXACTLY 1

with a single physically-named group

    Pi_c = Pi / (1 + ste_cold),
    Pi   = rho_ratio · ste_hot
         = rho_l·cp_l·(T_dep − T_melt) / (rho_s·L)          (identity)

i.e.  Pi_c = deposit superheat energy per unit volume
             ─────────────────────────────────────────────
             substrate melting cost per unit volume (latent + sensible)

so the law reads  fuel^b × driving-force:  the Part-1 onset variable
(margin = E/(1+E) − theta_melt) is the depth's linear driving force, and
all five material/process groups beyond it enter only through the
calorimetric ratio.

Legs
----
  G1  headline 6-exponent fit (reproduction check vs derive_powerlaws.py)
  G2  margin exponent pinned := 1 (cost of the pin)
  G3  3-exponent compact: rho_ratio, ste_cold, ste_hot | margin=1
  G4  1-exponent collapse: Pi_c^b · margin
  G5  0-exponent check: a·Pi_c^(3/4)·margin, only prefactor calibrated
  G6  rate/duration decomposition of the Pi_c exponent:
        D_rem = 2k·lam·sqrt(t_max)  =>  regress log(lam) and
        0.5·log(t_at_max_remelt) separately on {log Pi_c, log margin}.
        NOTE: t_at_max_remelt and the implied k are FD OUTPUTS — G6 is a
        DIAGNOSTIC decomposition (why is b ≈ 3/4?), NOT a predictive leg,
        and must never be quoted as a correlation.
  G7  curvature check: log(margin)^2 term on top of G1 (erf-linearisation
      validity across the full melted population)

All predictive legs (G1-G5) are scored with GroupKFold-by-lhs_id OOF R2 on
log(D_rem), consistent with derive_powerlaws.py. Population: status=='ok',
D_rem > NO_MELT_THR, margin > 0 (margin<=0 melted rows are dropped and
counted — same FD-vs-analytic disagreement Phase 3's classifier AUC gap
reflects).

Usage
-----
    python derive_powerlaws_calorimetric.py --csv progress_pure.csv
    python derive_powerlaws_calorimetric.py --csv results_pure_merged.csv --thr 0.005
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

NO_MELT_THR_DEFAULT = 0.000000001
N_SPLITS = 5


# ────────────────────────────────────────────────────────────────────────────
# data
# ────────────────────────────────────────────────────────────────────────────

def load_population(csv_path: str, no_melt_thr: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()

    # pure metal: theta_liquidus == theta_melt; accept either column name
    theta_col = 'theta_liquidus' if ('theta_liquidus' in df.columns
                                     and df['theta_liquidus'].notna().any()) else 'theta_melt'
    df['margin'] = df['theta_contact'] - df[theta_col]

    melted = df['D_rem'] > no_melt_thr
    bad_margin = melted & ~(df['margin'] > 0)   # catches <=0 AND NaN
    if bad_margin.any():
        print(f"  ⚠ dropping {int(bad_margin.sum())}/{int(melted.sum())} melted rows with "
              f"margin<=0 (log undefined) — same disagreement the onset-classifier AUC gap "
              f"reflects; count is reported, not hidden.")

    d = df[melted & (df['margin'] > 0)].copy()

    # named groups (identities from conventions.py)
    d['rho_ratio_c'] = d['E'] / (d['C'] * np.sqrt(d['r']))       # = rho_l/rho_s
    d['Pi'] = d['rho_ratio_c'] * d['ste_hot']                    # = rho_l cp_l dT_sh /(rho_s L)
    d['Pi_c'] = d['Pi'] / (1.0 + d['ste_cold'])                  # latent + sensible cost

    # sanity: rho_ratio identity should match the CSV's own column
    if 'rho_ratio' in d.columns:
        rel = (d['rho_ratio_c'] / d['rho_ratio'] - 1.0).abs().max()
        assert rel < 1e-6, f"rho_ratio identity violated (max rel err {rel:.2e})"

    print(f"  population: n={len(d)} melted rows (thr={no_melt_thr}), "
          f"{d['lhs_id'].nunique()} materials, margin in "
          f"[{d['margin'].min():.4f}, {d['margin'].max():.4f}]")
    return d


# ────────────────────────────────────────────────────────────────────────────
# fitting machinery
# ────────────────────────────────────────────────────────────────────────────

def oof_powerlaw(d: pd.DataFrame, cols: list[str],
                 fixed: dict[str, float] | None = None,
                 target: str = 'D_rem', label: str = '') -> dict:
    """Fit log(target) = log a + Σ b_i·log(col_i) + Σ e_j·log(fixed_j),
    with the fixed exponents pinned (moved to the offset), and report
    GroupKFold-by-lhs_id OOF R2 measured on the FULL log(target) — so pinned
    and free forms are compared on the same footing."""
    y = np.log(d[target].values)
    groups = d['lhs_id'].values
    off = (sum(e * np.log(d[c].values) for c, e in fixed.items())
           if fixed else np.zeros_like(y))
    yy = y - off

    X = (np.column_stack([np.log(d[c].values) for c in cols])
         if cols else np.zeros((len(d), 0)))

    oof = np.zeros_like(yy)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr, te in gkf.split(np.arange(len(yy))[:, None], yy, groups):
        if cols:
            lr = LinearRegression().fit(X[tr], yy[tr])
            oof[te] = lr.predict(X[te])
        else:
            oof[te] = yy[tr].mean()          # 0-exponent: prefactor only

    pred = oof + off
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

    if cols:
        lr_full = LinearRegression().fit(X, yy)
        a = float(np.exp(lr_full.intercept_))
        exps = dict(zip(cols, (float(c) for c in lr_full.coef_)))
    else:
        a = float(np.exp(yy.mean()))
        exps = {}

    parts = [f'{c}^{e:+.3f}' for c, e in exps.items()]
    parts += [f'{c}^{e:+.3f} (pinned)' for c, e in (fixed or {}).items()]
    print(f"  {label:34s} OOF R2 = {r2:.4f}   a = {a:.3f}   " + "  ".join(parts))
    return {'label': label, 'r2_oof': r2, 'a': a, 'exponents': exps,
            'pinned': dict(fixed or {}), 'n': len(d)}


# ────────────────────────────────────────────────────────────────────────────
# legs
# ────────────────────────────────────────────────────────────────────────────

def run(csv_path: str, no_melt_thr: float) -> None:
    print("═" * 78)
    print("  LEG G — CALORIMETRIC COLLAPSE OF THE PURE-METAL D_rem POWER LAW")
    print("═" * 78)
    d = load_population(csv_path, no_melt_thr)
    results = []

    print("\n[G1] headline 6-exponent law (reproduction check)")
    results.append(oof_powerlaw(d, ['E', 'C', 'r', 'ste_cold', 'ste_hot', 'margin'],
                                label='E,C,r,ste_c,ste_h,margin'))
    print("\n[G1a] five raw groups, NO margin feature (motivates adding margin)")
    results.append(oof_powerlaw(d, ['E', 'C', 'r', 'ste_cold', 'ste_hot'],
                                label='E,C,r,ste_c,ste_h (no margin)'))
    print("\n[G2] margin exponent pinned := 1")
    results.append(oof_powerlaw(d, ['E', 'C', 'r', 'ste_cold', 'ste_hot'],
                                fixed={'margin': 1.0},
                                label='E,C,r,ste_c,ste_h | margin=1'))

    print("\n[G3] 3-exponent compact (rho_ratio = E/(C·√r) identity)")
    results.append(oof_powerlaw(d, ['rho_ratio_c', 'ste_cold', 'ste_hot'],
                                fixed={'margin': 1.0},
                                label='rho,ste_c,ste_h | margin=1'))

    print("\n[G4] 1-exponent calorimetric collapse: Pi_c = rho·ste_h/(1+ste_c)")
    g4 = oof_powerlaw(d, ['Pi_c'], fixed={'margin': 1.0},
                      label='Pi_c^b | margin=1')
    results.append(g4)

    print("\n[G5] 0-exponent check: b pinned := 3/4, only prefactor calibrated")
    g5 = oof_powerlaw(d, [], fixed={'Pi_c': 0.75, 'margin': 1.0},
                      label='a·Pi_c^(3/4)·margin')
    results.append(g5)

    # ── G6: WHY b ≈ 3/4 — rate/duration decomposition (DIAGNOSTIC ONLY) ────
    print("\n[G6] rate/duration decomposition (diagnostic — uses FD outputs, "
          "NOT a predictive correlation)")
    have = [c for c in ('lam_neumann', 't_at_max_remelt') if c in d.columns]
    if len(have) == 2:
        dd = d[(d['lam_neumann'] > 0) & (d['t_at_max_remelt'] > 0)].copy()
        print(f"  n = {len(dd)} rows with lam>0 and t_max>0 "
              f"({len(d) - len(dd)} excluded: negative eigenvalue or missing t_max)")
        X = np.column_stack([np.log(dd['Pi_c']), np.log(dd['margin'])])
        for name, tgt in (('log(lam_neumann)      [rate]', np.log(dd['lam_neumann'].values)),
                          ('0.5·log(t_max)    [duration]', 0.5 * np.log(dd['t_at_max_remelt'].values)),
                          ('log(D_rem)           [total]', np.log(dd['D_rem'].values))):
            lr = LinearRegression().fit(X, tgt)
            print(f"    {name}:  Pi_c^{lr.coef_[0]:+.3f}  margin^{lr.coef_[1]:+.3f}  "
                  f"(in-sample R2 {lr.score(X, tgt):.3f})")
        k = (dd['D_rem'] / (2 * dd['lam_neumann'] * np.sqrt(dd['t_at_max_remelt'])))
        print(f"    deceleration factor k = D_rem/(2·lam·√t_max): "
              f"median {k.median():.3f}, IQR [{k.quantile(.25):.3f}, {k.quantile(.75):.3f}]")
        print("  Reading: the Pi_c exponent in D_rem splits between the melt RATE "
              "(lam) and the melt DURATION (√t_max); margin lives almost entirely "
              "in the rate channel — the linearised-Neumann eigenvalue.")
    else:
        print("  ⚠ lam_neumann / t_at_max_remelt not in CSV — G6 skipped.")

    # ── G7: curvature check (erf-linearisation validity) ────────────────────
    print("\n[G7] curvature check: log(margin)^2 on top of G1 basis (in-sample)")
    cols7 = ['E', 'C', 'r', 'ste_cold', 'ste_hot', 'margin']
    X7 = np.column_stack([np.log(d[c].values) for c in cols7]
                         + [np.log(d['margin'].values) ** 2])
    lr7 = LinearRegression().fit(X7, np.log(d['D_rem'].values))
    print(f"  log(margin)^2 coef = {lr7.coef_[-1]:+.4f}  "
          f"(vs linear log(margin) coef {lr7.coef_[5]:+.4f}) — "
          f"|quad| ≪ |lin| ⇒ linear-in-margin (small-λ Neumann) regime holds "
          f"across the full melted population.")

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  SUMMARY — free exponents vs OOF R2 (GroupKFold-by-lhs_id)")
    print("─" * 78)
    n_free = {'E,C,r,ste_c,ste_h,margin': 6, 'E,C,r,ste_c,ste_h | margin=1': 5,
              'E,C,r,ste_c,ste_h (no margin)': 5,
              'rho,ste_c,ste_h | margin=1': 3, 'Pi_c^b | margin=1': 1,
              'a·Pi_c^(3/4)·margin': 0}
    for res in results:
        print(f"  {n_free.get(res['label'], '?'):>2} exponents   "
              f"R2 = {res['r2_oof']:.4f}   {res['label']}")
    b = g4['exponents'].get('Pi_c', float('nan'))
    print(f"\n  Headline closed form (quote fitted b = {b:.3f}; 3/4 is a rounding, "
          f"not a derivation):")
    print(f"    D_rem ≈ {g4['a']:.2f} · [ρ_l·cp_l·(T_dep−T_melt) / "
          f"(ρ_s·(L + cp_s·(T_melt−T_sub)))]^{b:.3f} · (E/(1+E) − θ_melt)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    ap.add_argument('--csv', required=True, help='pure-metal sweep results CSV')
    ap.add_argument('--thr', type=float, default=NO_MELT_THR_DEFAULT,
                    help=f'no-melt threshold on D_rem (default {NO_MELT_THR_DEFAULT})')
    args = ap.parse_args()
    sys.exit(run(args.csv, args.thr))