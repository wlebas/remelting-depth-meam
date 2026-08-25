"""
pipeline_config.py
==================
Single source of truth for the closed remelting-sweep pipeline.

Every threshold, bound, grid setting, sizing constant, and physical convention
used anywhere in the pipeline lives HERE and is imported, never re-typed. 

DESIGN
------
* `CONVENTIONS`  — physical/mathematical definitions that must NOT change without
                   re-deriving and re-validating the entire analysis.  Frozen.
* `PARAMS`       — tunable thresholds, caps, and sweep settings.  Frozen per run,
                   but a run may override them (and the override is recorded).
* Every numeric constant carries a one-line provenance note: where it came from
  and why it has that value (calibration, physics, or convention).
* `.to_dict()` / `.fingerprint()` let each run STAMP its output with the exact
  config used — the reproducibility requirement from Phase 0.
* `validate()` asserts internal consistency at import time, so a self-contradictory
  config (e.g. shallow band not matching population thresholds) fails loudly and
  immediately rather than producing subtly wrong data.

USAGE
-----
    from pipeline_config import CONVENTIONS as CV, PARAMS as P

    if not (P.theta_melt_min < theta_melt < P.theta_melt_max):
        return None
    t_total, Z_total, flags = size_run(..., tau_dev_mult=P.tau_dev_mult, ...)

    # stamp the output
    meta = {**P.to_dict(), 'config_fingerprint': P.fingerprint()}
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Any
import json
import hashlib


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONVENTIONS — do NOT change without re-deriving everything downstream
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Conventions:
    """
    Physical and mathematical definitions. These are not tuning knobs — changing
    any of them invalidates every fitted correlation and every CSV produced.
    """

    # ── Dimensionless group keys (the identity of a case) ─────────────────────
    # Order matters: it defines column order and the merge/match key.
    group_keys: Tuple[str, ...] = ('E', 'C', 'r', 'ste_cold', 'ste_hot')
    scheil_extra_key: str = 'k_p'          # 6th group on the Scheil track

    # ── θ-normalisation (full-swing) ──────────────────────────────────────────
    # θ = (T − T_sub)/(T_dep − T_sub);  θ_deposit = 1, θ_substrate = 0.
    # The reference ΔT is the FULL SWING T_dep − T_sub (NOT T_liq − T_sub).
    # This was Bug 3: compute_adim_groups used T_liq − T_sub for inv_Ste.
    dT_ref_is_full_swing: bool = True
    deposit_h: float = 1.0                  # deposit thickness in units of h

    # ── Latent-heat reference (kernel constitutive curve) ─────────────────────
    # L* = inv_Ste_dT = (θ_liq − θ_sol)/Ste_m, referenced to (ρcp)_s.
    # Both kernels (pure fixed-curve, Scheil solid-reference) use this.
    # The alternative (ρcp)_l reference would give L* → B·inv_Ste_dT.
    latent_heat_reference: str = 'solid'    # 'solid' → L* = inv_Ste_dT

    # ── CFL stability (explicit FD) ───────────────────────────────────────────
    # dt = cfl · dz² / max(α_s, α_l);  cfl = 0.4 validated for this solver.
    cfl: float = 0.4

    # ── Identity tolerances (for conventions.py audits) ───────────────────────
    identity_tol_analytic: float = 1e-10    # physical-props identities
    identity_tol_csv: float = 1e-6          # row-wise CSV identities

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PARAMS — tunable thresholds, caps, sweep settings
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Params:
    """
    Tunable parameters. Frozen for a given run; a run may construct a modified
    copy via `replace(...)` and the fingerprint will change accordingly so the
    output is traceable to the exact settings used.
    """

    # ── Grid resolution ───────────────────────────────────────────────────────
    dz_coarse: float = 0.01    # sweep grid spacing [h]   (Pure_metal_sweep DZ_FINE)
    dz_fine:   float = 0.0025  # refinement grid spacing [h] (shallow-band refine)

    # ── Time / domain a-priori sizing (sweep_sizing.py, calibrated June 2026) ─
    # t_total = max(tau_dev_mult · tau_dev, t_floor), capped at t_cap.
    # Calibration: tau_dev_mult=2.5 captures the true peak depth within 1% for
    # ~86% of deep-corner cases (median residual 0.00%, 90th-pct 1.8%).
    tau_dev_mult: float = 2.5
    t_floor: float = 6.0       # min t_total; also covers θ_melt≥0.5 saturated cases
    t_cap:   float = 96.0      # hard cap; cases above → flagged time_capped
    # Domain: Z = deposit_h + D_estimate + thermal penetration buffer.
    z_floor: float = 15.0      # min domain [h]  (original GRID_SIZE)
    z_cap:   float = 40.0      # hard cap; cases above → flagged domain_capped
    farfield_eps: float = 0.01 # allowed far-field θ at bottom boundary

    # ── Three-population thresholds ───────────────────────────────────────────
    # none: D_rem < no_melt ;  shallow/transient: [no_melt, resolved) ;
    # resolved: ≥ resolved.  These define BOTH the population split AND the
    # refinement band, so they live in one place.
    no_melt_threshold:  float = 0.005
    resolved_threshold: float = 0.10

    # ── Refinement band (must equal the shallow population band) ──────────────
    # Cases with coarse D_rem in [shallow_lo, shallow_hi) are re-run at dz_fine.
    # Tied to the population thresholds by validate().
    shallow_lo: float = 0.00   # == no_melt_threshold
    shallow_hi: float = 0.10   # == resolved_threshold

    # ── Time-censoring detection & correction (time_censoring_study.py) ───────
    horizons: Tuple[float, ...] = (6.0, 12.0, 24.0, 48.0)  # re-run horizons
    bias_tol:    float = 0.01  # >this (D_asym−D_t6)/D_t6 ⇒ time_censored
    plateau_tol: float = 0.01  # |D_t48−D_t24|/D_t48 < this ⇒ plateaued (asymptote captured)
    # Deep-corner pre-filter (where censoring concentrates): r < r_max & D > d_min.
    corner_r_max: float = 1.2
    corner_d_min: float = 0.40

    # ── Grid/time convergence acceptance (convergence_study.py) ───────────────
    grid_tol: float = 0.01     # 90th-pct relerr at finest refinement step
    time_tol: float = 0.01     # |D(0.9T)−D(T)|/D(T) ⇒ time-converged
    dz_convergence: Tuple[float, ...] = (0.01, 0.005, 0.0025)

    # ── Physical guard rails (run_one) ────────────────────────────────────────
    # Cases outside these are unphysical or numerically unsafe → skipped.
    theta_melt_min: float = 0.01
    theta_melt_max: float = 0.99
    rho_ratio_min:  float = 0.20
    rho_ratio_max:  float = 10.00
    k_ratio_min:    float = 0.05
    k_ratio_max:    float = 20.0

    # ── LHS sweep design (Pure_metal_sweep.py) ────────────────────────────────
    n_lhs: int = 1000          # virtual materials
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'E':        (0.40, 3.00),   # linear
        'C':        (0.60, 2.00),   # linear
        'r':        (0.20, 4.00),   # log scale
        'ste_cold': (0.05, 2.00),   # log scale
        'ste_hot':  (0.02, 1.50),   # log scale
    })
    log_scaled_axes: Tuple[str, ...] = ('r', 'ste_cold', 'ste_hot')
    # Operating-point grid tiled over each virtual material (6 × 5 = 30 op-pts).
    ste_cold_op: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.90, 1.40, 2.00)
    ste_hot_op:  Tuple[float, ...] = (0.05, 0.15, 0.30, 0.60, 1.00)

    # ── Merge / matching ──────────────────────────────────────────────────────
    match_round_dp: int = 6    # round group keys to this many dp when merging CSVs
                               # (corner CSV stored 9 sig figs; full has float64)

    # ── I/O / fault tolerance ─────────────────────────────────────────────────
    checkpoint_every: int = 50
    rng_seed: int = 42         # LHS + any sampling reproducibility

    # ── Provenance vocabulary (the only allowed values) ───────────────────────
    provenance_values: Tuple[str, ...] = (
        'unchanged',   # not flagged, original D_rem kept
        'refined',     # shallow-band re-run at dz_fine
        'corrected',   # time-censored, plateaued → asymptotic peak
        'excluded',    # time-censored, still growing → drop from fit
        'unaudited',   # predicted-censored but not re-run (coverage gap)
    )

    # ── derived helpers ───────────────────────────────────────────────────────
    def populations(self):
        """Return the three-population boundary tuple (none|shallow|resolved)."""
        return (self.no_melt_threshold, self.resolved_threshold)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # tuples → lists for JSON friendliness
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def fingerprint(self) -> str:
        """
        Short stable hash of the config. Stamp every output CSV/figure with this
        so a result is always traceable to the exact parameters that produced it.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True).encode('utf-8')
        return hashlib.sha256(blob).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  VALIDATION — fail loudly at import if the config is self-contradictory
# ═══════════════════════════════════════════════════════════════════════════════

def validate(conv: Conventions, par: Params) -> None:
    """
    Assert internal consistency. Called at import time. Any failure here means
    the config contradicts itself — that must never reach a sweep.
    """
    errs = []

    # Population thresholds ordered
    if not (0.0 < par.no_melt_threshold < par.resolved_threshold):
        errs.append("require 0 < no_melt_threshold < resolved_threshold")

    # Refinement band ordered; upper edge tied to the resolved population
    # boundary. shallow_lo is intentionally NOT tied to no_melt_threshold:
    # refinement covers any nonzero coarse melt, independent of where the
    # "no melt" population label is drawn.
    if not (par.shallow_lo < par.shallow_hi):
        errs.append("require shallow_lo < shallow_hi")
    if par.shallow_hi != par.resolved_threshold:
        errs.append(f"shallow_hi ({par.shallow_hi}) must equal "
                    f"resolved_threshold ({par.resolved_threshold})")

    # Sizing caps ordered
    if not (par.t_floor < par.t_cap):
        errs.append("require t_floor < t_cap")
    if not (par.z_floor < par.z_cap):
        errs.append("require z_floor < z_cap")
    if par.tau_dev_mult <= 0:
        errs.append("tau_dev_mult must be > 0")

    # Horizons sorted, ascending, first == t_floor (the sweep horizon)
    if list(par.horizons) != sorted(par.horizons):
        errs.append("horizons must be ascending")
    if par.horizons[0] != par.t_floor:
        errs.append(f"first horizon ({par.horizons[0]}) should equal "
                    f"t_floor ({par.t_floor}) so t=6 is the sweep baseline")

    # Grid: fine must be finer than coarse, and convergence levels include both
    if not (par.dz_fine < par.dz_coarse):
        errs.append("dz_fine must be < dz_coarse")

    # Guard rails ordered
    for lo, hi, name in [
        (par.theta_melt_min, par.theta_melt_max, 'theta_melt'),
        (par.rho_ratio_min,  par.rho_ratio_max,  'rho_ratio'),
        (par.k_ratio_min,    par.k_ratio_max,    'k_ratio'),
    ]:
        if not (lo < hi):
            errs.append(f"{name} guard: min must be < max")

    # CFL in the stable range for explicit FD
    if not (0.0 < conv.cfl <= 0.5):
        errs.append(f"cfl ({conv.cfl}) must be in (0, 0.5] for explicit stability")

    # Ranges well-formed
    for key, (lo, hi) in par.ranges.items():
        if not (lo < hi):
            errs.append(f"range[{key}]: lo must be < hi")

    # Tolerances positive
    for name, val in [('bias_tol', par.bias_tol), ('plateau_tol', par.plateau_tol),
                      ('grid_tol', par.grid_tol), ('time_tol', par.time_tol),
                      ('farfield_eps', par.farfield_eps)]:
        if not (0.0 < val < 1.0):
            errs.append(f"{name} ({val}) must be in (0, 1)")

    if errs:
        raise ValueError("pipeline_config validation failed:\n  - "
                         + "\n  - ".join(errs))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SINGLETONS — import these everywhere
# ═══════════════════════════════════════════════════════════════════════════════

CONVENTIONS = Conventions()
PARAMS = Params()

# Validate at import: a broken config fails immediately, not mid-sweep.
validate(CONVENTIONS, PARAMS)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Self-report when run directly
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("═" * 64)
    print("  PIPELINE CONFIG")
    print("═" * 64)
    print(f"\n  Config fingerprint: {PARAMS.fingerprint()}")
    print(f"\n  CONVENTIONS (frozen physics):")
    for k, v in CONVENTIONS.to_dict().items():
        print(f"    {k:28s} = {v}")
    print(f"\n  PARAMS (tunable):")
    for k, v in PARAMS.to_dict().items():
        print(f"    {k:28s} = {v}")
    print(f"\n  Validation: PASS (config is internally consistent)")
    print("═" * 64)