"""
Analytical solver for melting of substrate
"""
from scipy.optimize import fsolve, brentq
import numpy as np
import pandas as pd
from materials import compute_adim_groups
from scipy.special import erf, erfc

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ANALYTICAL SOLVERS
# ═══════════════════════════════════════════════════════════════════════════════

def solve_mushy_similarity(groups, n_scan=60) -> tuple[float, float]:
    """
    Solve the two flux-continuity equations for the mushy-zone similarity
    eigenvalues (λ₁, λ₂).

    The three-zone temperature profiles in similarity variable η = x/(2√(αₛt)) are:

      Zone I  (liquid,  η < λ₁):
        T_1 = Tdep + (T_liq − T_dep)·(1 + erf(η/√r)) / (1 + erf(λ₁/√r))
          θ_I(η)  = θ_p + (1−θ_p)·(1 + erf(η/√r)) / (1 + erf(λ₁/√r))

      Zone II (mushy,  λ₁ < η < λ₂):
        Effective diffusivity ratio  p = r_eff = α_eff/α_s, with the
        apparent-heat-capacity smear  α_eff = α_m·Ste_m/(1+Ste_m).
        Similarity scaling factor  q ≡ 1/√p = √(1/r_eff)  (so q·η = η/√p).

        T_2 = T_liq + (T_liq − T_sol)·(erf(η/√p) − erf(λ₁/√p))
                                     / (erf(λ₁/√p) − erf(λ₂/√p))

          θ_II(η) = 1 + (θ_s − 1)·(erfc(q·η) − erfc(q·λ₁))
                                 / (erfc(q·λ₂) − erfc(q·λ₁))

          The erf and erfc forms are algebraically identical (erfc = 1 − erf,
          so the (θ_s−1)/erfc combination reproduces the T_2 expression). q
          controls the mushy-zone width: larger q ⟺ smaller r_eff ⟺ steeper
          mushy profile.

          erfc form is preferred for numerical stability when q is large:
          erfc of a large argument stays a small positive number, whereas erf
          saturates to 1 and differences like erf(q·λ₂) − erf(q·λ₁) lose all
          significant digits.

      Zone III (solid,  η > λ₂):
        T_3 = Tsub + (T_sol − T_sub)·erfc(η) / erfc(λ₂)
          θ_III(η) = θ_s·erfc(η) / erfc(λ₂)

    Flux continuity at each front gives two equations:

      FC1 at λ₁  (liquid flux = mushy flux):
          K·(θ_p−1)·exp(−λ₁²/r) / [√r·(1+erf(λ₁/√r))]
              = (1−θ_s)·q·exp(−q²λ₁²) / [erfc(q·λ₁) − erfc(q·λ₂)]
        E_lm · √(ste/(1+ste)) · (T_dep - T_liq) / (T_liq - T_sol)  =
         (1 + erf(λ₁/√r)) · exp(λ₁²(1/r - 1/p))/[erf(λ₂/√p) − erf(q·λ₁/√p)]

      FC2 at λ₂  (mushy flux = solid flux):
          (1−θ_s)·q·exp(−q²λ₂²) / [erfc(q·λ₁) − erfc(q·λ₂)]
              = K_s·θ_s·exp(−λ₂²) / erfc(λ₂)
        
        E_sm · √(ste/(1+ste)) · (T_sol - T_sub) / (T_liq - T_sol)  =  
         erfc(λ₂)· exp(λ₂²(1-1/p))/[erf(λ₂/√p) − erf(q·λ₁/√p)]

    These are simplified to have the material properties calculated as a constant as the LHS and the lambda are isolated on RHS
    Strategy: brute-force grid scan over (λ₁,λ₂) to find a good initial guess,
    then polish with fsolve. Returns (nan, nan) if no physical solution is found.

    Parameters
    ----------
    groups : dict from compute_groups()
    n_scan : total number of scan points per axis (default 60 → 30×30 grid)

    Returns
    -------
    (λ₁, λ₂) : float tuple, or (nan, nan) on failure.
    """
    tdep = groups['theta_deposit']
    tsub = groups['theta_substrate']
    tliq = groups['theta_liquidus']
    tsol = groups['theta_solidus']
    r  = groups['r']
    p  = groups['r_eff']
    eff_LM = groups['eff_LM']
    eff_SM = groups['eff_SM']
    ste = groups['Ste_m']

    LM_LHS = (eff_LM * (np.sqrt(ste/(1+ste)))) * (tdep - tliq)/(tliq - tsol)
    SM_LHS = (eff_SM * (np.sqrt(ste/(1+ste)))) * (tsol - tsub)/(tliq - tsol)

    def LM_RHS(l1, D12):
        """FC1 RHS: λ-dependent side of the liquid-mushy flux balance. Exponent uses λ₁."""
        return ((1 + erf(l1/np.sqrt(r)))/D12) * np.exp(l1**2 * (1/r - 1/p)) 

    def SM_RHS(l2, D12):
        """FC2 RHS: λ-dependent side of the mushy-solid flux balance. Exponent uses λ₂."""
        return erfc(l2)/D12 * np.exp(l2**2 * (1 - 1/p))

    # ── Residual function ─────────────────────────────────────────────────────
    # FC1:  liquid_flux_at_l1  = mushy_flux_at_l1  → residual[0] = 0
    # FC2:  mushy_flux_at_l2   = solid_flux_at_l2  → residual[1] = 0
    # Each equation is normalised by its own natural scale so both residuals
    # are O(1) near the solution, giving fsolve a well-conditioned Jacobian.

    def residual(lam):
        l1, l2 = float(lam[0]), float(lam[1])

        if l2 <= l1 + 1e-9:
            return [1e8, 1e8]
        
        # Zone II erfc denominator — shared by both flux terms
        D12 = float(erf(l2/np.sqrt(p))) - float(erf(l1/np.sqrt(p)))
        if abs(D12) < 1e-15:
            # λ₁ ≈ λ₂: mushy zone collapsed (near-sharp limit). Use solve_stefan_similarity instead.
            return [1e8, 1e8]

        # Zone III erfc guard
        ec_l2 = float(erfc(l2))
        if ec_l2 < 1e-15:
            return [1e8, 1e8]

        LM_RHS_1  = LM_RHS(l1, D12)
        SM_RHS_1 = SM_RHS(l2, D12)

        if not all(np.isfinite([LM_RHS_1, LM_LHS, SM_RHS_1, SM_LHS])):
            return [1e8, 1e8]

        # Normalise each equation by its own scale → both residuals O(1)
        scale1 = max(abs(LM_LHS),  abs(LM_RHS_1), 1e-15)   # scale of FC1
        scale2 = max(abs(SM_LHS), abs(SM_RHS_1),  1e-15)   # scale of FC2

        return [(LM_LHS - LM_RHS_1) / scale1,   # FC1
                (SM_LHS - SM_RHS_1) / scale2]   # FC2

    # ── Phase 1: grid scan for initial guess ──────────────────────────────────
    # Dense uniform grid for small λ (where solutions typically live for metals)
    # plus a coarser log-spaced extension up to λ=2 for unusual materials.
    # Tighter upper bound than the previous logspace(...,10) which wasted 50% of
    # scan points on physically unrealistic λ > 2.
    lv = np.concatenate([
        -np.logspace(np.log10(2), -0.3, n_scan // 4),  # coarse, log, λ ∈ [-2, -0.5]  (deposit-side, large)
        np.linspace(-0.5, -1e-4, n_scan // 4),          # fine, uniform, λ ∈ [-0.5, 0) (deposit-side, small)
        np.linspace(1e-4, 0.5, n_scan // 4),            # fine, uniform, λ ∈ (0, 0.5]  (substrate-side, small)
        np.logspace(-0.3, np.log10(2), n_scan // 4)     # coarse, log, λ ∈ [0.5, 2]    (substrate-side, large)
    ])

    best_norm = np.inf
    best      = np.array([0.2, 0.5])

    for l1 in lv:
        for l2 in lv:
            if l2 <= l1:
                continue
            try:
                rn = np.linalg.norm(residual([l1, l2]))
                if np.isfinite(rn) and rn < best_norm:
                    best_norm = rn
                    best = np.array([l1, l2])
            except Exception:
                pass

    # ── Phase 2: polish with fsolve ───────────────────────────────────────────
    try:
        sol, _, ier, _ = fsolve(residual, best, full_output=True)
        if ier == 1 and sol[1] > sol[0] + 1e-9:
            return float(sol[0]), float(sol[1])
    except Exception:
        pass

    # Fallback: if fsolve failed but the grid scan found a point close enough
    # to the root, return it as an approximate solution -- UNLESS that point
    # is pinned at the scan's own edge, which means the scan never actually
    # bracketed an interior minimum, it's reporting "as far as I could push
    # it" dressed up as a converged eigenvalue. fsolve started from exactly
    # this point in the failing cases and made zero improvement (ier=5).
    margin = 1.5e-4  # same absolute margin as the original check
    lo_edge = lv.min() + margin
    hi_edge = lv.max() - margin
    if best_norm < 0.05 and lo_edge < best[0] < hi_edge:
        return float(best[0]), float(best[1])

    return float('nan'), float('nan')


def solve_stefan_similarity(groups) -> float:
    """
    Sharp-interface (Ste_m → 0) similarity eigenvalue λ for the remelting problem.

    In the sharp-interface limit the mushy zone collapses to a single front at λ.
    The temperature profiles are:

      Zone I  (liquid, η < λ):   T_1 = T_dep − ((T_dep−T_liq)/(1 + erf(λ/√r))) · (1 + erf(η/√r))
      Zone II (solid,  η > λ):   T_2 = T_sub + ((T_sol−T_sub)/erfc(λ)) · erfc(η)

    The single Stefan condition (flux jump = latent heat) gives:

        k_l · dT_1/dη|_λ − k_s · dT_2/dη|_λ = Ste_fd · λ · √π

    which simplifies (using gradient expressions from the profiles above) to:

        (E/C) · Ste_superheat · exp(−λ²/r) / [(1 + erf(λ/√r))]
        − Ste_undercooling · exp(−λ²) / erfc(λ) = λ · √π

    where E = effusivity_l/effusivity_s = eff_ratio, 
    Ste_superheat = cp_l·(T_dep−T_liq) / L.
    Ste_undercooling = cp_s·(T_liq−T_sub) / L.

    Returns λ (float) which can be positive or negative.
    """

    r   = groups['r']                              # α_l / α_s
    Ste_superheat = groups['Ste_superheat']             # cp_l·(T_dep−T_liq) / L
    Ste_undercooling = groups['Ste_undercooling']         # cp_s·(T_liq−T_sub) / L
    eff_ratio = groups['E_ratio']                       #effusivity ratio liquid to solid
    Cp_ratio = groups['cp_ratio']                               #cp ratio liquid to solid

    def equation(lam):
        lam = float(lam)
        ec_lam  = float(erfc(lam))
        ec_l_r  = 1 + float(erf(lam / np.sqrt(r)))
        if ec_l_r < 1e-15:
            return -lam * np.sqrt(np.pi)   # liquid side saturated → −λ√π
        liquid_flux = (eff_ratio/Cp_ratio) * Ste_superheat * np.exp(-lam**2 / r) / ec_l_r
        solid_flux  = (Ste_undercooling * np.exp(-lam**2) / ec_lam) if ec_lam > 1e-300 else 0.0
        return liquid_flux - solid_flux - lam * np.sqrt(np.pi)

    try:
        lams = np.concatenate([np.linspace(-3.0, -1e-6, 300), np.linspace(1e-6, 0.5, 300), np.linspace(0.5, 5.5, 200)])
        vals = np.array([equation(l) for l in lams])
        bkts = [(lams[i], lams[i + 1]) for i in range(len(vals) - 1)
                if np.isfinite(vals[i]) and np.isfinite(vals[i + 1])
                and vals[i] * vals[i + 1] < 0]
        if bkts:
            root: float = brentq(equation, bkts[0][0], bkts[0][1], xtol=1e-10)  # type: ignore[assignment]
            return root
    except Exception:
        pass
    return float("nan")

def _nan_row(T_dep, T_sub, reason):
    """Return a result-row dict filled with NaN for a skipped point."""
    return {
        'T_deposit':      T_dep,
        'T_substrate':    T_sub,
        'lam1':           np.nan,
        'lam2':           np.nan,
        'lam_neumann':    np.nan,
        'eps_opt_neumann':    np.nan,
        's2_at_eps_neumann':  np.nan,
        'status_neumann':     reason,
        'eps_opt_lam1':       np.nan,
        's2_at_eps_lam1':     np.nan,
        'status_lam1':        reason,
    }

def _bisect_epsilon(lam: float, r: float, s2fd: float):
    """
    Find epsilon in (0,1) such that lam / (2*sqrt(r)*erfinv(eps)) == s2fd.
    Returns (eps_opt, s2_at_eps, status_string).
    """
    from scipy.special import erfinv as _erfinv
    from scipy.optimize import brentq

    if not (np.isfinite(lam) and lam > 0 and np.isfinite(s2fd) and s2fd > 0):
        return np.nan, np.nan, 'invalid_input'

    def _f(eps):
        ex = float(_erfinv(eps))
        if ex < 1e-15:
            return 1e8
        return lam / (2.0 * np.sqrt(r) * ex) - s2fd

    eps_lo, eps_hi = 1e-9, 1.0 - 1e-9
    f_lo, f_hi = _f(eps_lo), _f(eps_hi)

    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return np.nan, np.nan, 'f_not_finite'
    if f_lo * f_hi > 0:
        return np.nan, np.nan, 's2fd_below_range' if f_hi > 0 else 's2fd_above_range'

    try:
        eps_opt = float(brentq(_f, eps_lo, eps_hi, xtol=1e-10, rtol=1e-10))    # type: ignore[assignment] 
        s2_at_eps = lam / (2.0 * np.sqrt(r) * float(_erfinv(eps_opt)))
        return eps_opt, s2_at_eps, 'converged'
    except Exception as exc:
        return np.nan, np.nan, f'brentq_failed:{exc}'

def analytical_solver(
    temp_deposit,
    temp_substrate,
    material_properties,
):
    """
    For each (T_deposit, T_substrate) pair, compute:

      1. Mushy similarity eigenvalues  (lam1, lam2)  via solve_mushy_similarity
      2. Sharp-interface eigenvalue    (lam_neumann)  via solve_stefan_similarity
      3. FD max remelting depth        (s2fd)          via fd_solver  [optional]
      4. Solidus-crossing lower bound  (s2min, tau_dev) via compute_s2min
      5. Optimal epsilon               (epsilon_opt)   via bisection on
            compute_s2min_t_crit(lam2, r, x=eps)[0] == s2fd

    The bisection exploits the monotonicity of
        s2m(eps) = lam2 / (2 * sqrt(r) * erfinv(eps))
    which is strictly decreasing in eps ∈ (0, 1):
        eps → 0  ⟹  s2m → +∞
        eps → 1  ⟹  s2m →  0
    so there is at most one root, and brentq is safe.

    Parameters
    ----------
    temp_deposit, temp_substrate : 1-D array-like of temperatures [°C]
    material_properties : dict  (will be mutated then restored)
    grid_size           : Z_total passed to grid_builder
    deposit_thickness   : h     passed to grid_builder
    dz_fine             : grid spacing; coarser = faster but less accurate
    run_fd              : if False, skip the FD solve (epsilon_opt = nan)

    Returns
    -------
    df  : pd.DataFrame with one row per (T_dep, T_sub) pair and columns:
            T_deposit, T_substrate,
            lam1, lam2, lam_neumann,
            s2fd,
            s2min, tau_dev,
            epsilon_opt, s2_at_eps_opt,
            bracket_status          ← diagnostic string

    arrays : dict of 2-D np.ndarray, shape (n_dep, n_sub), for easy plotting:
            'lam1', 'lam2', 'lam_neumann',
            's2fd', 's2min', 'epsilon_opt'
    """

    temp_deposit   = np.asarray(temp_deposit,   dtype=float)
    temp_substrate = np.asarray(temp_substrate, dtype=float)
    n_dep = len(temp_deposit)
    n_sub = len(temp_substrate)

    # ── Pre-allocate 2-D result arrays ────────────────────────────────────────
    arr_lam1        = np.full((n_dep, n_sub), np.nan)
    arr_lam2        = np.full((n_dep, n_sub), np.nan)
    arr_lam_neumann = np.full((n_dep, n_sub), np.nan)
    arr_s2fd        = np.full((n_dep, n_sub), np.nan)
    arr_eps_opt_neumann = np.full((n_dep, n_sub), np.nan)
    arr_s2_at_eps_neumann = np.full((n_dep, n_sub), np.nan)
    arr_eps_opt_lam1    = np.full((n_dep, n_sub), np.nan)
    arr_s2_at_eps_lam1  = np.full((n_dep, n_sub), np.nan)

    # ── Snapshot indices (reused for every FD call) ───────────────────────────
    # Built lazily after the first grid_builder call so nt_max is known.
    snapshot_indices = None

    # ── Stash original temperatures so we can restore the dict ───────────────
    T_dep_orig = material_properties['T_deposit']
    T_sub_orig = material_properties['T_sub']

    rows = []   # accumulate per-point dicts for the DataFrame

    total = n_dep * n_sub
    done  = 0

    # ── Pre-compute effusivity ratio (temperature-independent) ───────────────
    _eff_l   = np.sqrt(material_properties['rho_l'] * material_properties['cp_l'] * material_properties['k_l'])
    _eff_s   = np.sqrt(material_properties['rho_s'] * material_properties['cp_s'] * material_properties['k_s'])
    _E_ratio = _eff_l / _eff_s
    _theta_contact = _E_ratio / (1.0 + _E_ratio)

    for i, T_dep in enumerate(temp_deposit):
        material_properties['T_deposit'] = float(T_dep)

        for j, T_sub in enumerate(temp_substrate):
            material_properties['T_sub'] = float(T_sub)
            done += 1

            # ── Skip unphysical combinations ──────────────────────────────────
            if T_sub >= material_properties['T_liq']:
                rows.append(_nan_row(T_dep, T_sub, 'T_sub >= T_liq'))
                continue
            if T_dep <= material_properties['T_liq']:
                rows.append(_nan_row(T_dep, T_sub, 'T_dep <= T_liq'))
                continue
                        # ── Skip if contact temperature cannot reach liquidus ─────────────

            # ── Dimensionless groups ──────────────────────────────────────────
            try:
                adim = compute_adim_groups(material_properties)
            except ValueError as exc:
                rows.append(_nan_row(T_dep, T_sub, f'adim error: {exc}'))
                continue
            # ── Similarity eigenvalues ────────────────────────────────────────
            lam1, lam2       = solve_mushy_similarity(adim)
            lam_neumann      = solve_stefan_similarity(adim)

            # ── Store in 2-D arrays ───────────────────────────────────────────
            arr_lam1[i, j]        = lam1
            arr_lam2[i, j]        = lam2
            arr_lam_neumann[i, j] = lam_neumann

            # ── Progress + row dict ───────────────────────────────────────────
            print(f"  [{done:4d}/{total}]  "
                f"T_dep={T_dep:6.1f}  T_sub={T_sub:5.1f}  "
                f"lam_n={lam_neumann:.4f} ")

            rows.append({
                'T_deposit':          T_dep,
                'T_substrate':        T_sub,
                'lam1':               lam1,
                'lam2':               lam2,
                'lam_neumann':        lam_neumann,
            })

    # ── Restore original material_properties ─────────────────────────────────
    material_properties['T_deposit'] = T_dep_orig
    material_properties['T_sub']     = T_sub_orig

    df = pd.DataFrame(rows)

    arrays = {
        'lam1':               arr_lam1,
        'lam2':               arr_lam2,
        'lam_neumann':        arr_lam_neumann,
    }

    return df, arrays