"""
solvers.py
==========
Public wrappers for the FD remelting solvers, returning a UNIFIED result type.

Both the pure/eutectic and the Scheil solvers return a single `SolverResult`
dataclass, so downstream code (sweep, refinement, peak-detection, analysis)
never branches on model type or unpacks model-specific tuples.

THE UNIFICATION
---------------
The only physical difference between the two solvers is the mushy zone:

  * pure / eutectic — no mushy zone; a single front separates solid and liquid,
    so the liquidus and solidus fronts COINCIDE.  In SolverResult they are set
    equal: solidus_front_position == liquidus_front_position and D_sol == D_rem.

  * Scheil — a finite mushy zone; the solidus front runs deeper than the
    liquidus front, and mushy_width = D_sol − D_rem at the peak-remelt instant.

Every other field has the same meaning for both.  A caller that only wants the
remelting depth reads `result.D_rem`; one that wants the front history reads
`result.liquidus_front_position`; neither needs to know which solver ran.

KERNELS ARE UNTOUCHED
---------------------
The JIT cores in kernels.py are verified and frozen.  These wrappers only
unpack the cores' raw output, compute scalars, and package everything into
SolverResult.  No physics changes here.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from kernels import _fd_solver_core_pure, _fd_solver_core_scheil


# ═══════════════════════════════════════════════════════════════════════════════
# Unified result type
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolverResult:
    """
    Unified return type for all FD solvers.

    For pure/eutectic metals the liquidus and solidus fronts coincide, so
    `solidus_front_position == liquidus_front_position` and `D_sol == D_rem`.
    """

    # ── Model identity ────────────────────────────────────────────────────────
    model: str                              # 'pure_metal' | 'scheil'

    # ── Headline scalars (the quantities the correlation predicts) ────────────
    D_rem: float                            # max liquidus (remelting) depth [h]
    D_sol: float                            # max solidus depth [h] (= D_rem if pure)

    # ── Time-resolved fronts (length nt_max) — for peak detection / censoring ─
    liquidus_front_position: np.ndarray
    solidus_front_position: np.ndarray      # == liquidus for pure
    interface_temperature: np.ndarray

    # ── Spatial fields (length nz) ────────────────────────────────────────────
    temperature_field: np.ndarray           # final θ(z)
    T_profile_at_max_remelt: np.ndarray     # θ(z) at the peak-remelt instant

    # ── Snapshots ─────────────────────────────────────────────────────────────
    theta_hist: np.ndarray                  # (nz, n_snaps)
    snap_ctr: int

    # ── DataFrames (for plotting / inspection) ────────────────────────────────
    df_time: pd.DataFrame
    df_space: pd.DataFrame

    # ── Peak timing ───────────────────────────────────────────────────────────
    time_to_max_remelt: int
    time_to_max_solidus: int                # == time_to_max_remelt for pure
    final_interface_temp: float

    # ── Scheil-specific (sensible defaults for pure) ──────────────────────────
    mushy_width_at_max_remelt: float = 0.0
    k_p: float = 1.0

    # ── Escape hatch: original scalar dict, for anything not promoted above ───
    scalars: dict = field(default_factory=dict)

    # ── Convenience ───────────────────────────────────────────────────────────
    def row(self) -> dict:
        """
        Flat dict of headline numbers, suitable as a sweep CSV row.
        Model-agnostic: always the same keys.
        """
        return {
            'model'                    : self.model,
            'D_rem'                    : self.D_rem,
            'D_sol'                    : self.D_sol,
            'mushy_width_at_max_remelt': self.mushy_width_at_max_remelt,
            'time_to_max_remelt'       : self.time_to_max_remelt,
            'time_to_max_solidus'      : self.time_to_max_solidus,
            'final_interface_temp'     : self.final_interface_temp,
            'k_p'                      : self.k_p,
        }

    @property
    def melt_front_position(self) -> np.ndarray:
        """Alias for liquidus_front_position (pure-metal naming)."""
        return self.liquidus_front_position


# ═══════════════════════════════════════════════════════════════════════════════
# Pure / eutectic wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def fd_solver_pure(adim_groups, z, interface_idx, dt, nt_max,
                   snapshot_indices) -> SolverResult:
    """
    Public entry point for the pure-metal / eutectic FD solver.

    Parameters
    ----------
    adim_groups      : dict   Must include theta_liquidus, inv_Ste, k_ratio,
                              rho_ratio, cp_ratio.
    z                : array  Node positions from build_grid().
    interface_idx    : int    Deposit/substrate interface index.
    dt               : float  Time step from build_grid().
    nt_max           : int    Number of time steps.
    snapshot_indices : list   Step indices at which to store θ(z).

    Returns
    -------
    SolverResult with model='pure_metal'. Liquidus and solidus fronts coincide.
    """
    theta_melt    = adim_groups['theta_liquidus']  # liquidus == solidus for pure
    inv_Ste       = adim_groups['inv_Ste']
    k_ratio       = adim_groups['k_ratio']
    rho_ratio     = adim_groups['rho_ratio']
    cp_ratio      = adim_groups['cp_ratio']
    interface_idx = int(interface_idx)

    snapshot_indices_arr = np.array(snapshot_indices, dtype=np.int64)

    (temperature_field,
     melt_front_position,
     interface_temperature,
     theta_hist,
     snap_ctr,
     T_profile_at_max_remelt) = _fd_solver_core_pure(
        theta_melt, inv_Ste,
        k_ratio, rho_ratio, cp_ratio,
        z, interface_idx, dt, nt_max,
        snapshot_indices_arr,
    )

    nonzero_steps = np.nonzero(interface_temperature)[0]
    last_step = int(nonzero_steps[-1]) if len(nonzero_steps) > 0 else 0

    max_remelting = float(np.max(melt_front_position))
    dz = float(z[1] - z[0])

    if max_remelting < 0.5 * dz:
        t_max_remelt = 0
    else:
        near_max = np.where(melt_front_position >= max_remelting - 0.5 * dz)[0]
        t_max_remelt = int(round(0.5 * (near_max[0] + near_max[-1])))

    scalars = {
        'max_remelting'       : max_remelting,
        'time_to_max_remelt'  : t_max_remelt,
        'final_interface_temp': float(interface_temperature[last_step]),
    }

    df_time = pd.DataFrame({
        'timestep'               : np.arange(nt_max),
        'interface_temperature'  : interface_temperature,
        'liquidus_front_position': melt_front_position,
        'solidus_front_position' : melt_front_position,   # coincide for pure
    }).iloc[:last_step + 1]

    df_space = pd.DataFrame({
        'z'                : z,
        'temperature_field': temperature_field,
        'T_at_max_remelt'  : T_profile_at_max_remelt,
    })

    return SolverResult(
        model='pure_metal',
        D_rem=max_remelting,
        D_sol=max_remelting,                         # solidus == liquidus
        liquidus_front_position=melt_front_position,
        solidus_front_position=melt_front_position,  # same array, by physics
        interface_temperature=interface_temperature,
        temperature_field=temperature_field,
        T_profile_at_max_remelt=T_profile_at_max_remelt,
        theta_hist=theta_hist,
        snap_ctr=snap_ctr,
        df_time=df_time,
        df_space=df_space,
        time_to_max_remelt=t_max_remelt,
        time_to_max_solidus=t_max_remelt,            # same instant
        final_interface_temp=float(interface_temperature[last_step]),
        mushy_width_at_max_remelt=0.0,               # no mushy zone
        k_p=1.0,
        scalars=scalars,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Scheil wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def fd_solver_scheil(adim_groups, z, interface_idx, dt, nt_max,
                     snapshot_indices) -> SolverResult:
    """
    Public entry point for the Scheil FD solver.

    Parameters
    ----------
    adim_groups      : dict   Must include theta_liquidus, theta_solidus, Ste_m,
                              k_p, k_ratio, rho_ratio, cp_ratio.
    z, interface_idx, dt, nt_max, snapshot_indices : as for fd_solver_pure.

    Returns
    -------
    SolverResult with model='scheil'. Solidus front runs deeper than liquidus;
    mushy_width = D_sol − D_rem at the peak-remelt instant.
    """
    theta_liquidus = adim_groups['theta_liquidus']
    theta_solidus  = adim_groups['theta_solidus']
    Ste_m          = adim_groups['Ste_m']
    k_p            = adim_groups['k_p']
    k_ratio        = adim_groups['k_ratio']
    rho_ratio      = adim_groups['rho_ratio']
    cp_ratio       = adim_groups['cp_ratio']
    interface_idx  = int(interface_idx)

    snapshot_indices_arr = np.array(snapshot_indices, dtype=np.int64)

    (temperature_field,
     liquidus_front_position,
     solidus_front_position,
     interface_temperature,
     theta_hist,
     snap_ctr,
     T_profile_at_max_remelt) = _fd_solver_core_scheil(
        theta_liquidus, theta_solidus, Ste_m, k_p,
        k_ratio, rho_ratio, cp_ratio,
        z, interface_idx, dt, nt_max,
        snapshot_indices_arr,
    )

    max_remelting = float(np.max(liquidus_front_position))
    max_solidus   = float(np.max(solidus_front_position))
    dz = float(z[1] - z[0])

    if max_remelting < 0.5 * dz:
        t_max_remelt = 0
    else:
        near_max_liq = np.where(liquidus_front_position >= max_remelting - 0.5 * dz)[0]
        t_max_remelt = int(round(0.5 * (near_max_liq[0] + near_max_liq[-1])))

    if max_solidus < 0.5 * dz:
        t_max_solidus = 0
    else:
        near_max_sol = np.where(solidus_front_position >= max_solidus - 0.5 * dz)[0]
        t_max_solidus = int(round(0.5 * (near_max_sol[0] + near_max_sol[-1])))
    mushy_width_at_max_remelt = float(
        solidus_front_position[t_max_remelt] - liquidus_front_position[t_max_remelt]
    )

    scalars = {
        'max_remelting'            : max_remelting,
        'max_solidus'              : max_solidus,
        'time_to_max_remelt'       : t_max_remelt,
        'time_to_max_solidus'      : t_max_solidus,
        'final_interface_temp'     : float(interface_temperature[-1]),
        'mushy_width_at_max_remelt': mushy_width_at_max_remelt,
        'k_p'                      : k_p,
    }

    df_time = pd.DataFrame({
        'timestep'               : np.arange(nt_max),
        'interface_temperature'  : interface_temperature,
        'liquidus_front_position': liquidus_front_position,
        'solidus_front_position' : solidus_front_position,
        'mushy_width'            : solidus_front_position - liquidus_front_position,
    })

    df_space = pd.DataFrame({
        'z'                : z,
        'temperature_field': temperature_field,
        'T_at_max_remelt'  : T_profile_at_max_remelt,
    })

    return SolverResult(
        model='scheil',
        D_rem=max_remelting,
        D_sol=max_solidus,
        liquidus_front_position=liquidus_front_position,
        solidus_front_position=solidus_front_position,
        interface_temperature=interface_temperature,
        temperature_field=temperature_field,
        T_profile_at_max_remelt=T_profile_at_max_remelt,
        theta_hist=theta_hist,
        snap_ctr=snap_ctr,
        df_time=df_time,
        df_space=df_space,
        time_to_max_remelt=t_max_remelt,
        time_to_max_solidus=t_max_solidus,
        final_interface_temp=float(interface_temperature[-1]),
        mushy_width_at_max_remelt=mushy_width_at_max_remelt,
        k_p=k_p,
        scalars=scalars,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch helper: pick the solver from the model tag
# ═══════════════════════════════════════════════════════════════════════════════

def fd_solve(adim_groups, z, interface_idx, dt, nt_max, snapshot_indices,
             model: Optional[str] = None) -> SolverResult:
    """
    Model-agnostic entry point. `model` defaults to adim_groups['model'] if
    present, else 'pure_metal'.  Lets a caller run any case without branching.
    """
    if model is None:
        model = adim_groups.get('model', 'pure_metal')

    if model in ('pure_metal', 'pure', 'eutectic'):
        return fd_solver_pure(adim_groups, z, interface_idx, dt, nt_max,
                              snapshot_indices)
    elif model == 'scheil':
        return fd_solver_scheil(adim_groups, z, interface_idx, dt, nt_max,
                               snapshot_indices)
    else:
        raise ValueError(f"unknown model '{model}' "
                         f"(expected 'pure_metal' or 'scheil')")