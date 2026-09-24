"""
grid.py
=======
Spatial grid construction and a-priori run sizing for the FD remelting solver.

Two responsibilities, both pure (no I/O, no pandas, no solver dependency):

  1. build_grid()  — construct the node positions, interface index, stable
                     timestep (CFL), and step count for a given domain, grid
                     spacing, and run time.  This is the old `grid_builder`,
                     unchanged in behaviour but with the magic CFL number now
                     sourced from pipeline_config.

  2. size_run()    — predict the run time (t_total) and domain size (Z_total) a
                     case needs to capture the TRUE maximum remelting depth,
                     from the dimensionless groups alone.  Calibrated against the
                     June-2026 censoring study (t = 2.5·tau_dev captures the peak
                     within 1% for ~86% of deep-corner cases).

  size_for_case()  — convenience wrapper: size_run() then build_grid().

All constants come from pipeline_config (CONVENTIONS.cfl, PARAMS.tau_dev_mult,
t_floor, t_cap, z_floor, z_cap, farfield_eps, deposit_h).  Nothing numeric is
hard-coded here.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import erfinv, erfcinv

from config_pipeline import CONVENTIONS as CV, PARAMS as P


# ═══════════════════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Grid:
    """A constructed spatial/temporal grid, ready to hand to a solver core."""
    z: np.ndarray            # node positions [h], shape (nz,)
    interface_idx: int       # index of the deposit/substrate interface
    dt: float                # stable timestep [h²/α_s]
    nt_max: int              # number of timesteps
    Z_total: float           # domain depth [h]
    dz: float                # grid spacing [h]
    t_total: float           # total run time [h²/α_s]


@dataclass(frozen=True)
class SizingFlags:
    """Provenance of an a-priori size_run() decision."""
    theta_melt: float
    tau_dev: float           # nan if theta_melt >= 0.5 (saturated)
    t_uncapped: float        # t_total before the cap
    z_uncapped: float        # Z_total before the cap
    time_capped: bool        # hit t_cap → case may still be time-censored
    domain_capped: bool      # hit z_cap → case may still be domain-censored
    saturated: bool          # theta_melt >= 0.5 (tau_dev fallback used)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Grid construction  (was grid_builder)
# ═══════════════════════════════════════════════════════════════════════════════

def build_grid(Z_total: float,
               dz: float,
               t_total: float,
               r: float,
               h: Optional[float] = None) -> Grid:
    """
    Construct the spatial grid, interface index, CFL-stable timestep, and step
    count.  Behaviour-identical to the original grid_builder; the CFL number now
    comes from CONVENTIONS.cfl instead of a hard-coded 0.4.

    Parameters
    ----------
    Z_total  : domain depth [h]
    dz       : grid spacing [h]
    t_total  : total run time [h²/α_s]
    r        : diffusivity ratio α_l/α_s (sets the liquid-side CFL limit)
    h        : deposit thickness [h] (defaults to CONVENTIONS.deposit_h = 1.0)

    Returns
    -------
    Grid dataclass.
    """
    if h is None:
        h = CV.deposit_h
    n_cells = int(round(Z_total / dz))
    z = (np.arange(n_cells) + 0.5) * dz          # cell centres; faces at 0, dz, 2dz, ...
    n_dep = int(round(h / dz))
    assert abs(n_dep * dz - h) < 1e-12, "deposit thickness must be an exact multiple of dz"
    interface_idx = n_dep - 1                     # last deposit node; interface = face at z = h

    # Adimensionalised diffusivities (solid = 1 by convention); liquid = r.
    alpha_max = max(1.0, r)
    dt = CV.cfl * dz**2 / alpha_max
    nt_max = int(t_total / dt)

    return Grid(z=z, interface_idx=interface_idx, dt=dt, nt_max=nt_max,
            Z_total=n_cells * dz, dz=dz, t_total=t_total)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  A-priori run sizing  (was sweep_sizing)
# ═══════════════════════════════════════════════════════════════════════════════

def _theta_melt(ste_cold: float, ste_hot: float, C: float) -> float:
    """theta_melt = ste_cold / (ste_cold + ste_hot / C)."""
    return ste_cold / (ste_cold + ste_hot / C)


def tau_dev(r: float, theta_melt: float) -> float:
    """
    Deposit-exhaustion time: when the P&G deposit-centre temperature drops to
    theta_melt (so the deposit stops driving remelting).

        tau_dev = 1 / (4 r [erfinv(2 theta_melt)]²)

    Returns nan if theta_melt >= 0.5 (erfinv argument saturates); those cases
    converge fast and are handled by the t_floor in size_run().
    """
    arg = 2.0 * theta_melt
    if arg >= 1.0:
        return np.nan
    ei = erfinv(arg)
    if ei < 1e-9:
        return np.nan
    return 1.0 / (4.0 * r * ei**2)


def neumann_depth_estimate(lam_neumann: Optional[float],
                           r: float, t_total: float) -> float:
    """
    Similarity melt depth at t_total: D = 2 lam sqrt(r t_total).  Used only to
    SIZE the domain (over-estimate is safe).  Returns 0 if no similarity
    solution exists (lam None/non-finite).
    """
    if lam_neumann is None or not np.isfinite(lam_neumann):
        return 0.0
    return 2.0 * lam_neumann * np.sqrt(r * t_total)


def thermal_penetration(t_total: float,
                        eps: Optional[float] = None) -> float:
    """
    Depth at which the substrate temperature has fallen to `eps` of its peak:

        z_pen = 2 sqrt(r t_total) erfcinv(2 eps)

    The buffer the domain needs below the melt front so the bottom adiabatic BC
    does not reflect heat.  eps defaults to PARAMS.farfield_eps.
    """
    if eps is None:
        eps = P.farfield_eps
    return 2.0 * np.sqrt(t_total) * erfcinv(2.0 * eps)


def size_run(E: float, C: float, r: float,
             ste_cold: float, ste_hot: float,
             lam_neumann: Optional[float] = None) -> tuple:
    """
    Predict (t_total, Z_total, flags) for one set of dimensionless groups.

    All tuning constants come from PARAMS (tau_dev_mult, t_floor, t_cap,
    z_floor, z_cap, farfield_eps).

    Returns
    -------
    t_total : float       recommended run time [h²/α_s]
    Z_total : float       recommended domain size [h]
    flags   : SizingFlags provenance (time_capped, domain_capped, saturated, …)
    """
    theta_m = _theta_melt(ste_cold, ste_hot, C)

    # ── Time ──────────────────────────────────────────────────────────────────
    td = tau_dev(r, theta_m)
    saturated = not np.isfinite(td)
    if saturated:
        t_uncapped = P.t_floor          # theta_melt >= 0.5 → fast convergence
    else:
        t_uncapped = max(P.tau_dev_mult * td, P.t_floor)

    t_total = min(t_uncapped, P.t_cap)
    time_capped = t_uncapped > P.t_cap

    # ── Domain ────────────────────────────────────────────────────────────────
    D_est = neumann_depth_estimate(lam_neumann, r, t_total)
    z_pen = thermal_penetration(t_total, P.farfield_eps)
    z_uncapped = max(CV.deposit_h + D_est + z_pen, P.z_floor)

    Z_total = min(z_uncapped, P.z_cap)
    domain_capped = z_uncapped > P.z_cap

    flags = SizingFlags(
        theta_melt=theta_m,
        tau_dev=td,
        t_uncapped=t_uncapped,
        z_uncapped=z_uncapped,
        time_capped=bool(time_capped),
        domain_capped=bool(domain_capped),
        saturated=bool(saturated),
    )
    return t_total, Z_total, flags


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Convenience: size then build
# ═══════════════════════════════════════════════════════════════════════════════

def size_for_case(E: float, C: float, r: float,
                  ste_cold: float, ste_hot: float,
                  dz: Optional[float] = None,
                  lam_neumann: Optional[float] = None) -> tuple:
    """
    Size and build in one call: returns (Grid, SizingFlags).

    dz defaults to PARAMS.dz_coarse (the sweep grid).  For refinement runs pass
    dz = PARAMS.dz_fine.
    """
    if dz is None:
        dz = P.dz_coarse
    t_total, Z_total, flags = size_run(E, C, r, ste_cold, ste_hot, lam_neumann)
    grid = build_grid(Z_total=Z_total, dz=dz, t_total=t_total, r=r)
    return grid, flags


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"grid.py self-test  (config {P.fingerprint()})\n")

    # A mid case and a deep low-r case
    for label, (E, C, r, sc, sh) in [
        ("mid     ", (1.0, 1.0, 1.0, 0.50, 0.25)),
        ("deep low-r", (2.6, 1.0, 0.21, 0.10, 0.50)),
        ("saturated", (1.0, 1.0, 1.0, 1.40, 0.05)),  # high theta_melt
    ]:
        t, Z, fl = size_run(E, C, r, sc, sh)
        g = build_grid(Z_total=Z, dz=P.dz_coarse, t_total=t, r=r)
        print(f"  {label}: theta_m={fl.theta_melt:.3f}  tau_dev="
              f"{fl.tau_dev if np.isfinite(fl.tau_dev) else float('nan'):.2f}  "
              f"t_total={t:.1f}  Z_total={Z:.1f}  "
              f"nz={len(g.z)}  nt={g.nt_max}  "
              f"{'[time_capped]' if fl.time_capped else ''}"
              f"{'[domain_capped]' if fl.domain_capped else ''}"
              f"{'[saturated]' if fl.saturated else ''}")

    # CFL sanity: dt must satisfy dt <= cfl*dz^2/max(1,r)
    g = build_grid(Z_total=15.0, dz=P.dz_coarse, t_total=6.0, r=4.0)
    assert abs(g.dt - CV.cfl * P.dz_coarse**2 / 4.0) < 1e-18, "CFL mismatch"
    print(f"\n  CFL check: dt={g.dt:.3e} = {CV.cfl}*dz²/max(1,r)  OK")
    print("  PASS")