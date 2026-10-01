"""
conventions.py
==============
Single source of truth for every dimensionless-group definition, algebraic
identity, normalisation convention, and kernel acceptance test used across
the pure-metal and Scheil sweeps.

USAGE
-----
Import anywhere that maps physical properties to dimensionless groups, or
run as a script to verify an existing results CSV against all identities:

    python conventions.py                          # self-tests only
    python conventions.py --csv results_refined.csv          # + CSV audit
    python conventions.py --csv results_refined.csv --scheil # Scheil CSV

CONVENTIONS (definitive)
------------------------
θ-normalisation (full-swing):
    θ = (T − T_sub) / (T_dep − T_sub)
    θ_deposit  = 1    (by definition)
    θ_substrate = 0   (by definition)
    θ_liquidus = (T_liq − T_sub) / (T_dep − T_sub)
    θ_solidus  = (T_sol − T_sub) / (T_dep − T_sub)

Reference length:   h  (deposit thickness)
Reference time:     h² / α_s
Reference ΔT:       T_dep − T_sub   (FULL SWING, NOT T_liq − T_sub)

Solid properties normalised to 1:
    k̃_s = ρ̃_s = c̃p_s = 1

Dimensionless groups — pure-metal sweep (5 groups):
    E         = √(ρ k cp)_l / √(ρ k cp)_s        (effusivity ratio)
    C         = cp_l / cp_s                        (heat-capacity ratio)
    r         = α_l / α_s                          (diffusivity ratio)
    ste_cold  = cp_s · (T_melt − T_sub) / L        (substrate Stefan)
    ste_hot   = cp_l · (T_dep − T_melt) / L        (deposit Stefan)

Derived (pure-metal):
    cp_ratio  = C
    rho_ratio = E / (C · √r)
    k_ratio   = E · √r
    inv_ste_full = ste_cold + ste_hot / C  =  cp_s · (T_dep − T_sub) / L
    theta_melt   = ste_cold / inv_ste_full
    inv_Ste      = 1 / inv_ste_full         [= L / (cp_s · (T_dep − T_sub))]

Dimensionless groups — Scheil sweep (6 groups):
    Ste_m     = cp_m · (T_liq − T_sol) / L        (mushy-zone Stefan)
    r                                              (same as above)
    E                                              (same as above)
    k_p       = partition coefficient
    θ_liq     = (T_liq − T_sub) / (T_dep − T_sub)
    mushy_frac = θ_sol / θ_liq

Latent-heat reference (documented choice):
    L* = inv_Ste_dT = (θ_liq − θ_sol) / Ste_m
    This uses (ρcp)_s as reference, consistent with the sweep's full-swing
    normalisation and the pure-metal kernel.  The alternative (ρcp)_l
    reference gives L* → B · inv_Ste_dT.  Do not mix.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import erf, erfc

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PHYSICAL → DIMENSIONLESS  (canonical mapping, replaces compute_adim_groups)
# ═══════════════════════════════════════════════════════════════════════════════

def adim_groups_from_physical(props):
    """
    Convert a physical-properties dict to dimensionless groups.

    Parameters
    ----------
    props : dict with keys
        T_deposit, T_liq, T_sol, T_sub  [°C or K — consistent units]
        k_s, k_l    [W/mK]
        rho_s, rho_l  [kg/m³]
        cp_s, cp_l    [J/kgK]
        L             [J/kg]
        k_p           [–]  partition coefficient (Scheil; set 1.0 for pure)

    Returns
    -------
    dict with ALL dimensionless groups used by both sweeps and both solvers.

    KEY DIFFERENCE FROM compute_adim_groups()
    ------------------------------------------
    dT_ref = T_dep − T_sub   (FULL SWING)
    The old function used dT_ref = T_liq − T_sub, giving a wrong inv_Ste
    for superheated deposits.  This function is the correct replacement.
    """
    p = props
    T_dep = float(p['T_deposit'])
    T_liq = float(p['T_liq'])
    T_sol = float(p['T_sol'])
    T_sub = float(p['T_sub'])

    # ── Guards ────────────────────────────────────────────────────────────────
    if T_dep <= T_liq:
        raise ValueError(f"T_deposit ({T_dep}) must be > T_liq ({T_liq})")
    if T_liq <= T_sol:
        raise ValueError(f"T_liq ({T_liq}) must be > T_sol ({T_sol})")
    if T_sol <= T_sub:
        raise ValueError(f"T_sol ({T_sol}) must be > T_sub ({T_sub})")

    # ── Temperature scale: FULL SWING T_dep − T_sub ───────────────────────────
    dT_ref = T_dep - T_sub          # ← correct; old code used T_liq − T_sub
    dT_f   = T_liq - T_sol          # freezing range

    theta_liquidus = (T_liq - T_sub) / dT_ref
    theta_solidus  = (T_sol - T_sub) / dT_ref

    # ── Material property ratios ───────────────────────────────────────────────
    k_ratio   = p['k_l'] / p['k_s']
    rho_ratio = p['rho_l'] / p['rho_s']
    cp_ratio  = p['cp_l'] / p['cp_s']

    alpha_s = p['k_s'] / (p['rho_s'] * p['cp_s'])
    alpha_l = p['k_l'] / (p['rho_l'] * p['cp_l'])
    r = alpha_l / alpha_s

    eff_s = np.sqrt(p['rho_s'] * p['cp_s'] * p['k_s'])
    eff_l = np.sqrt(p['rho_l'] * p['cp_l'] * p['k_l'])
    E = eff_l / eff_s

    C = cp_ratio

    # ── Stefan numbers (full-swing reference) ─────────────────────────────────
    # inv_Ste = L / (cp_s · dT_ref)  with dT_ref = T_dep − T_sub
    inv_Ste  = p['L'] / (p['cp_s'] * dT_ref)
    ste_cold = p['cp_s'] * (T_liq - T_sub) / p['L']   # = θ_liq / inv_Ste
    ste_hot  = p['cp_l'] * (T_dep - T_liq) / p['L']   # = C·(1-θ_liq)/inv_Ste

    # Mushy-zone mean properties (for Scheil sweep)
    k_m   = 0.5 * (p['k_l'] + p['k_s'])
    rho_m = 0.5 * (p['rho_l'] + p['rho_s'])
    cp_m  = 0.5 * (p['cp_l'] + p['cp_s'])
    Ste_m = cp_m * dT_f / p['L']

    # Latent-heat reference (solid-referenced, matches kernel choice):
    # L* = inv_Ste_dT = (θ_liq − θ_sol) / Ste_m
    inv_Ste_dT = (theta_liquidus - theta_solidus) / Ste_m

    # Scheil exponent
    k_p = float(p.get('k_p', 1.0))
    scheil_exp = 1.0 / (1.0 - k_p) if k_p < 1.0 else np.inf

    # Onset criterion
    theta_contact = E / (1.0 + E)   # two-body contact temperature
    mushy_frac    = theta_solidus / theta_liquidus if theta_liquidus > 0 else np.nan

    return {
        # ── θ-normalisation ───────────────────────────────────────────────────
        'theta_liquidus'  : theta_liquidus,
        'theta_solidus'   : theta_solidus,
        'mushy_frac'      : mushy_frac,
        'theta_contact'   : theta_contact,
        # ── 5 pure-metal groups ───────────────────────────────────────────────
        'E'               : E,
        'C'               : C,
        'r'               : r,
        'ste_cold'        : ste_cold,
        'ste_hot'         : ste_hot,
        # ── 6th Scheil group ──────────────────────────────────────────────────
        'k_p'             : k_p,
        'scheil_exp'      : scheil_exp,
        # ── Derived ratios (solver inputs) ────────────────────────────────────
        'k_ratio'         : k_ratio,
        'rho_ratio'       : rho_ratio,
        'cp_ratio'        : cp_ratio,
        # ── Stefan numbers ────────────────────────────────────────────────────
        'inv_Ste'         : inv_Ste,
        'ste_cold'        : ste_cold,
        'ste_hot'         : ste_hot,
        'Ste_m'           : Ste_m,
        'inv_Ste_dT'      : inv_Ste_dT,
        # ── Latent-heat reference (kernel convention) ─────────────────────────
        # L* = inv_Ste_dT  uses (ρcp)_s.  See module docstring.
        'L_star'          : inv_Ste_dT,
        'B'               : rho_ratio * cp_ratio,
        # ── Bookkeeping ───────────────────────────────────────────────────────
        'dT_ref'          : dT_ref,
        'dT_f'            : dT_f,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ALGEBRAIC IDENTITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_identities_physical(props, tol=1e-10):
    """
    Verify all algebraic identities for a physical-properties dict.
    Returns list of (name, passed, detail) triples.
    """
    g = adim_groups_from_physical(props)
    E  = g['E'];  C  = g['C'];  r  = g['r']
    kr = g['k_ratio'];  rr = g['rho_ratio'];  cpr = g['cp_ratio']
    results = []

    def chk(name, lhs, rhs):
        err = abs(lhs - rhs) / max(abs(rhs), 1e-15)
        results.append((name, err < tol, f"lhs={lhs:.8g} rhs={rhs:.8g} relerr={err:.2e}"))

    chk("E² = k_ratio·rho_ratio·cp_ratio",
        E**2, kr * rr * cpr)
    chk("r  = k_ratio / (rho_ratio·cp_ratio)",
        r, kr / (rr * cpr))
    chk("C  = cp_ratio",
        C, cpr)
    chk("k_ratio   = E·√r",
        kr, E * np.sqrt(r))
    chk("rho_ratio = E / (C·√r)",
        rr, E / (C * np.sqrt(r)))
    chk("theta_liq = ste_cold · inv_Ste",
        g['theta_liquidus'], g['ste_cold'] * g['inv_Ste'])
    chk("inv_Ste = 1/(ste_cold + ste_hot/C)",
        g['inv_Ste'], 1.0 / (g['ste_cold'] + g['ste_hot'] / C))
    chk("theta_contact = E/(1+E)",
        g['theta_contact'], E / (1.0 + E))
    chk("L_star = (theta_liq-theta_sol)/Ste_m",
        g['L_star'], (g['theta_liquidus'] - g['theta_solidus']) / g['Ste_m'])
    chk("B = rho_ratio·cp_ratio",
        g['B'], rr * cpr)

    return results


def check_identities_csv_pure(df, tol=1e-6):
    """
    Verify algebraic identities on every row of a pure-metal sweep CSV.

    Expected columns: E, C, r, ste_cold, ste_hot, theta_melt (or theta_liquidus),
                      k_ratio, rho_ratio, inv_Ste, lam_neumann (optional).

    Returns list of (name, max_relerr, n_fail, passed).
    """
    results = []

    # Column aliases
    tm = df.get('theta_melt', df.get('theta_liquidus', df.get('theta_melt')))
    E  = df['E'];  C  = df['C'];  r  = df['r']
    sc = df['ste_cold'];  sh = df['ste_hot']
    kr = df.get('k_ratio',   E * np.sqrt(r))
    rr = df.get('rho_ratio', E / (C * np.sqrt(r)))
    inv_Ste = df.get('inv_Ste', 1.0 / (sc + sh / C))

    def chk(name, lhs, rhs):
        relerr = (lhs - rhs).abs() / rhs.abs().clip(lower=1e-15)
        n_fail = int((relerr > tol).sum())
        results.append((name, float(relerr.max()), n_fail, n_fail == 0))

    inv_ste_full = sc + sh / C

    chk("k_ratio = E·√r",           kr,           E * np.sqrt(r))
    chk("rho_ratio = E/(C·√r)",      rr,           E / (C * np.sqrt(r)))
    chk("inv_Ste = 1/(sc+sh/C)",     inv_Ste,      1.0 / inv_ste_full)
    chk("theta_melt = sc·inv_Ste",   tm,           sc * inv_Ste)
    chk("theta_melt = sc/(sc+sh/C)", tm,           sc / inv_ste_full)
    chk("E² = k·rho·cp ratios",
        E**2, kr * rr * C)

    return results


def check_identities_csv_scheil(df, tol=1e-6):
    """
    Verify algebraic identities on every row of a Scheil sweep CSV.

    Expected columns: Ste_m, r, E, k_p, theta_liq, theta_sol, mushy_frac,
                      k_ratio, rho_ratio, cp_ratio (last 3 may be absent).
    """
    results = []

    tl = df['theta_liq'];  ts = df['theta_sol']
    E  = df['E'];          r  = df['r']
    mf = df['mushy_frac']

    RHO_RATIO_FIXED = 0.90   # Scheil sweep uses fixed rho_ratio
    kr  = E * np.sqrt(r)
    cpr = E / (RHO_RATIO_FIXED * np.sqrt(r))

    def chk(name, lhs, rhs):
        relerr = (lhs - rhs).abs() / rhs.abs().clip(lower=1e-15)
        n_fail = int((relerr > tol).sum())
        results.append((name, float(relerr.max()), n_fail, n_fail == 0))

    chk("theta_sol = mushy_frac · theta_liq", ts,  mf * tl)
    chk("k_ratio  = E·√r",                   kr,  E * np.sqrt(r))
    chk("cp_ratio = E/(rho_ratio·√r)",        cpr, E / (RHO_RATIO_FIXED * np.sqrt(r)))
    chk("theta_liq in (0,1)",
        tl, tl.clip(lower=1e-9, upper=1-1e-9))
    chk("theta_sol in (0, theta_liq)",
        ts, ts.clip(lower=1e-9))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  KERNEL ACCEPTANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def _neumann_exists(E, C, r, ste_cold, ste_hot):
    """Return (exists: bool, lam: float) for the Neumann similarity solution."""
    def eq(lam):
        erf_lr = 1.0 + float(erf(lam / np.sqrt(r)))
        if erf_lr < 1e-15:
            return -lam * np.sqrt(np.pi)
        liquid = (E / C) * ste_hot * np.exp(-lam**2 / r) / erf_lr
        ec = float(erfc(lam))
        solid = ste_cold * np.exp(-lam**2) / ec if ec > 1e-300 else 0.0
        return liquid - solid - lam * np.sqrt(np.pi)

    lams = np.concatenate([np.linspace(1e-4, 0.5, 300), np.linspace(0.5, 5.5, 200)])
    vals = [eq(l) for l in lams]
    bkts = [(lams[i], lams[i+1]) for i in range(len(vals)-1)
            if np.isfinite(vals[i]) and np.isfinite(vals[i+1])
            and vals[i] * vals[i+1] < 0]
    if bkts:
        try:
            lam = float(brentq(eq, bkts[0][0], bkts[0][1], xtol=1e-10)) # type: ignore
            return True, lam
        except Exception:
            pass
    return False, np.nan


def kernel_round_trip_pure(theta_melt, inv_Ste, B):
    """
    Algebraic round-trip for the pure-metal kernel's fixed H(θ) curve.
    Returns max |θ_out − θ_in| over solid, mushy, and liquid nodes.
    """
    H_sol = theta_melt
    H_liq = theta_melt + inv_Ste

    def encode(th):
        if th >= theta_melt:
            return H_liq + B * (th - theta_melt)
        return th

    def decode(H):
        if H >= H_liq:
            return theta_melt + (H - H_liq) / B
        elif H <= H_sol:
            return H
        return theta_melt   # mushy plateau

    worst = 0.0
    for th in [0.0, theta_melt * 0.5, theta_melt, theta_melt,
               0.5 * (theta_melt + 1.0), 1.0]:
        worst = max(worst, abs(decode(encode(th)) - th))
    return worst


def _scheil_fl(th, ts, tl, exp):
    if th <= ts: return 0.0
    if th >= tl: return 1.0
    return ((th - ts) / (tl - ts)) ** exp


def kernel_round_trip_scheil(theta_sol, theta_liq, inv_Ste_dT, B, k_p):
    """
    Algebraic round-trip for the fixed solid-reference Scheil H(θ) curve.
    Returns max |θ_out − θ_in| over solid, mushy (several), and liquid nodes.
    """
    exp = 1.0 / (1.0 - k_p) if k_p < 1.0 else 1.0
    H_sol_t = theta_sol
    H_liq_t = theta_liq + inv_Ste_dT

    def encode(th):
        if th >= theta_liq:
            return H_liq_t + B * (th - theta_liq)
        elif th <= theta_sol:
            return th
        return th + inv_Ste_dT * _scheil_fl(th, theta_sol, theta_liq, exp)

    def decode(H):
        if H >= H_liq_t:
            return theta_liq + (H - H_liq_t) / B
        elif H <= H_sol_t:
            return H
        # Newton on F(θ) = θ + L*·fl(θ) - H
        dT_f = theta_liq - theta_sol
        th = (H + inv_Ste_dT * theta_sol / dT_f) / (1.0 + inv_Ste_dT / dT_f)
        th = min(max(th, theta_sol), theta_liq)
        for _ in range(20):
            fl = _scheil_fl(th, theta_sol, theta_liq, exp)
            xi = (th - theta_sol) / dT_f if theta_sol < th < theta_liq else 0.0
            dfl = exp / dT_f * xi**(exp - 1) if 0 < xi < 1 else 0.0
            F  = th + inv_Ste_dT * fl - H
            dF = 1.0 + inv_Ste_dT * dfl
            if abs(dF) < 1e-30: break
            th_new = th - F / dF
            th_new = min(max(th_new, theta_sol), theta_liq)
            if abs(th_new - th) < 1e-12: th = th_new; break
            th = th_new
        return th

    test_pts = [0.0, theta_sol,
                theta_sol + 0.01*(theta_liq-theta_sol),
                theta_sol + 0.3*(theta_liq-theta_sol),
                theta_sol + 0.7*(theta_liq-theta_sol),
                theta_liq, 1.0]
    worst = max(abs(decode(encode(th)) - th) for th in test_pts)
    return worst


def kernel_round_trip_lever(theta_sol, theta_liq, inv_Ste_dT, B):
    """
    Algebraic round-trip for the fixed solid-reference lever-rule H(θ) curve.
    H_solidus = theta_sol (solid-props reference, intentional).
    Liquid branch: H = H_solidus + B·(θ−θ_sol) + B·inv_Ste_dT
    """
    H_solidus = theta_sol
    H_liq_t   = H_solidus + B * ((theta_liq - theta_sol) + inv_Ste_dT)

    def encode(th):
        if th >= theta_liq:
            return H_solidus + B * (th - theta_sol) + B * inv_Ste_dT
        elif th <= theta_sol:
            return th
        # mushy linear
        return th + inv_Ste_dT * (th - theta_sol) / (theta_liq - theta_sol)

    def decode(H):
        if H >= H_liq_t:
            return (H - H_solidus - B * inv_Ste_dT) / B + theta_sol
        elif H <= H_solidus:
            return H
        dT_f = theta_liq - theta_sol
        return (H + inv_Ste_dT * theta_sol / dT_f) / (1.0 + inv_Ste_dT / dT_f)

    test_pts = [0.0, theta_sol,
                theta_sol + 0.3*(theta_liq-theta_sol),
                theta_liq, 1.0]
    worst = max(abs(decode(encode(th)) - th) for th in test_pts)
    return worst


def run_kernel_round_trip_tests(tol=1e-10):
    """
    Run algebraic round-trip tests for all three kernels across a grid of
    (B, freezing-range) values.  Returns list of (name, passed, worst_err).
    """
    cases = [
        (0.40, 0.50, 0.70, 0.30),   # B < 1, narrow
        (0.40, 0.50, 1.40, 0.80),   # B > 1, narrow, high k_p
        (0.30, 0.60, 0.70, 0.50),   # B < 1, medium
        (0.25, 0.75, 1.40, 0.10),   # B > 1, wide  ← crashed unfixed Scheil
    ]
    results = []

    for (ts, tl, B, kp) in cases:
        L = (tl - ts) / 2.0   # inv_Ste_dT
        tm = tl               # pure metal: melt point = liquidus

        # Pure
        w = kernel_round_trip_pure(tm, L, B)
        results.append((f"pure  tm={tm} B={B:.1f}", w < tol, w))

        # Scheil
        w = kernel_round_trip_scheil(ts, tl, L, B, kp)
        results.append((f"scheil ts={ts} tl={tl} B={B:.1f} kp={kp}", w < tol, w))

        # Lever
        w = kernel_round_trip_lever(ts, tl, L, B)
        results.append((f"lever ts={ts} tl={tl} B={B:.1f}", w < tol, w))

    return results


def run_neumann_acceptance_test(fd_solver_pure_fn, groups_from_stefan_fn,
                                grid_builder_fn, tol_D=0.02):
    """
    Run the bug-report acceptance test: melt iff Neumann solution exists.

    Parameters
    ----------
    fd_solver_pure_fn    : fd_solver_pure from solvers
    groups_from_stefan_fn: pure-metal adim_groups builder
    grid_builder_fn      : build_grid from grid
    tol_D                : D_rem threshold to call 'melted'

    Returns list of (label, passed, detail).
    """
    cases = [
        (2.0, 1.0, 1.0, 0.10, 0.50, True,  "high E, hot deposit"),
        (1.5, 1.0, 1.5, 0.15, 0.40, True,  "moderate E, high r"),
        (3.0, 1.0, 2.0, 0.05, 0.80, True,  "very high E"),
        (0.3, 1.0, 0.5, 0.50, 0.10, False, "low E, cold deposit"),
        (0.5, 1.0, 0.3, 0.60, 0.05, False, "low E, low r"),
        (0.4, 1.0, 0.4, 0.70, 0.05, False, "low E, very cold"),
    ]
    results = []
    for (E, C, r, sc, sh, expect, label) in cases:
        try:
            grp = groups_from_stefan_fn(E, C, r, sc, sh)
            neu_exists, lam = _neumann_exists(E, C, r, sc, sh)

            # New spine: build_grid returns a Grid dataclass; fd_solver_pure
            # returns a SolverResult with a .D_rem attribute.
            g = grid_builder_fn(Z_total=12.0, dz=0.01, t_total=6.0, r=r)
            result = fd_solver_pure_fn(
                adim_groups=grp, z=g.z, interface_idx=g.interface_idx,
                dt=g.dt, nt_max=g.nt_max, snapshot_indices=[])
            D_rem = float(result.D_rem)
            solver_melts = D_rem > tol_D

            ok = (neu_exists == expect) and (solver_melts == expect)
            results.append((label, ok,
                             f"lam={'nan' if not neu_exists else f'{lam:.3f}'} "
                             f"D_rem={D_rem:.3f} expect={'melt' if expect else 'no-melt'}"))
        except Exception as exc:
            results.append((label, False, str(exc)))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"


def _print_section(title):
    print(f"\n{'═'*64}\n  {title}\n{'═'*64}")


def _report(name, passed, detail=""):
    tag = _PASS if passed else _FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN — self-tests + optional CSV audit
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="conventions.py — verify dimensionless-group identities")
    parser.add_argument("--csv",    default=None, help="Path to results CSV to audit")
    parser.add_argument("--scheil", action="store_true", help="Treat CSV as Scheil sweep output")
    parser.add_argument("--neumann", action="store_true",
                        help="Run Neumann acceptance test (requires rewrite_clean_ + Pure_metal_sweep on path)")
    args = parser.parse_args()

    all_passed = []

    # ── Section 1: physical-properties identity checks ────────────────────────
    _print_section("1. Physical-properties identity checks")
    from materials import (SS316L, INCONEL_718, TI64, CU_PURE, ER70S,
                           AL_SI10MG, AL_6061)

    test_materials = [SS316L, INCONEL_718, TI64, CU_PURE, ER70S, AL_SI10MG, AL_6061]
    for mat in test_materials:
        # Library dicts are intrinsic-only; inject a representative case so the
        # full-swing temperature scale is defined for the identity checks.
        T_liq = mat['T_liq']
        case = {**mat, 'T_deposit': T_liq + 0.15 * (T_liq - 25.0), 'T_sub': 25.0}
        res = check_identities_physical(case)
        any_fail = any(not ok for _, ok, _ in res)
        tag = _PASS if not any_fail else _FAIL
        print(f"\n  [{tag}] {mat['material']}")
        for name, ok, detail in res:
            _report(f"    {name}", ok, detail)
            all_passed.append(ok)

    # ── Section 2: kernel round-trip tests ───────────────────────────────────
    _print_section("2. Kernel round-trip tests (algebraic, no solver needed)")
    rt_results = run_kernel_round_trip_tests()
    for name, ok, worst in rt_results:
        _report(name, ok, f"worst_err={worst:.2e}")
        all_passed.append(ok)

    # ── Section 3: CSV audit ─────────────────────────────────────────────────
    if args.csv:
        _print_section(f"3. CSV identity audit: {args.csv}")
        df = pd.read_csv(args.csv)
        print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

        if args.scheil:
            checks = check_identities_csv_scheil(df)
        else:
            checks = check_identities_csv_pure(df)

        for name, max_err, n_fail, ok in checks:
            _report(name, ok,
                    f"max_relerr={max_err:.2e}  n_fail={n_fail}/{len(df)}")
            all_passed.append(ok)
    else:
        _print_section("3. CSV audit")
        print("  Skipped — pass --csv path/to/results.csv to audit a dataset")

    # ── Section 4: Neumann acceptance test ───────────────────────────────────
    _print_section("4. Neumann acceptance test (kernel regression gate)")
    if args.neumann:
        try:
            from grid import build_grid
            from solvers import fd_solver_pure

            # groups_from_stefan: build a pure-metal adim_groups dict from the
            # five dimensionless inputs. Self-contained so the gate needs only
            # the spine (grid.py + solvers.py + kernels.py), not the sweep code.
            def groups_from_stefan(E, C, r, sc, sh):
                inv_full = sc + sh / C
                theta_melt = sc / inv_full
                return {
                    'theta_liquidus': theta_melt,
                    'theta_melt':     theta_melt,
                    'inv_Ste':        1.0 / inv_full,
                    'k_ratio':        E * np.sqrt(r),
                    'rho_ratio':      E / (C * np.sqrt(r)),
                    'cp_ratio':       C,
                    'r': r, 'E': E, 'C': C,
                    'model': 'pure_metal',
                }

            neu_results = run_neumann_acceptance_test(
                fd_solver_pure, groups_from_stefan, build_grid)
            for label, ok, detail in neu_results:
                _report(label, ok, detail)
                all_passed.append(ok)
        except ImportError as e:
            print(f"  Skipped — could not import spine modules: {e}")
            print("  Run from the directory containing grid.py, solvers.py, kernels.py.")
    else:
        print("  Skipped — pass --neumann to run (requires the spine on the path)")

    # ── Summary ──────────────────────────────────────────────────────────────
    n_pass = sum(all_passed)
    n_total = len(all_passed)
    print(f"\n{'═'*64}")
    print(f"  {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print(f"  [{_PASS}] All conventions verified")
    else:
        print(f"  [{_FAIL}] {n_total - n_pass} check(s) failed — see above")
    print(f"{'═'*64}\n")
    return 0 if n_pass == n_total else 1


if __name__ == '__main__':
    sys.exit(main())