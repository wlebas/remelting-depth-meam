"""
This file contains the kernels for the remelted depth solver
There are two solvers, one for alloys and one for pure metals or eutectics. Note that the Scheil solver takes longer and is more expensive to run.

<<<<<<< HEAD
_fd_solver_core_pure 
=======
_fd_solver_core_pure — CORRECTED KERNEL
>>>>>>> 47d307f75ee58d62e76d610fd85c80031ba9eae1
=======================================
This is an enthalpy based solver for the pure-metal / eutectic case, with a fixed isothermal constitutive curve at the melting temperature.  
The enthalpy-temperature relationship is piecewise linear with constant thresholds:
"""
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def _fd_solver_core_pure(
        theta_melt, inv_Ste,
        k_ratio, rho_ratio, cp_ratio,
        z, interface_idx,
        dt, nt_max, snapshot_indices):
    """
    JIT-compiled time loop for the pure-metal / eutectic isothermal
    enthalpy solver. Fixed constitutive curve (see module docstring):

        solid  :  theta = H
        mushy  :  theta = theta_m            (f_l = (H - H_sol_m)/L*)
        liquid :  theta = theta_m + (H - H_liq_m)/B

    A single melt_front_position array is tracked.
    """

    nz = len(z)
    dz = z[1] - z[0]              # uniform grid assumed (as before)
    z_iface = z[interface_idx] + 0.5 * dz      # == h exactly

    # ── Fixed constitutive constants ───────────────────────────────────────────
    B_liq   = rho_ratio * cp_ratio      # (rho*cp)_l / (rho*cp)_s
    L_star  = inv_Ste                   # latent heat per volume (rho_s ref.)
    H_sol_m = theta_melt                # solid/mushy threshold  (constant)
    H_liq_m = theta_melt + L_star       # mushy/liquid threshold (constant)

    # ── Property array (only conductivity is phase-dependent now) ─────────────
    k_tilde = np.ones(nz)

    # ── Output arrays ──────────────────────────────────────────────────────────
    interface_temperature = np.zeros(nt_max)
    melt_front_position   = np.zeros(nt_max)
    theta_hist            = np.zeros((nz, len(snapshot_indices)))
    snap_ctr              = 0
    current_max_melt      = 0.0

    # ── Initial temperature field ──────────────────────────────────────────────
    temperature_field = np.zeros(nz)
    for i in range(nz):
        temperature_field[i] = 1.0 if z[i] <= z[interface_idx] else 0.0

    # ── Initial enthalpy field (FIX 1: evaluated on the fixed curve) ───────────
    # Deposit nodes are fully liquid (theta = 1 >= theta_m): they carry the
    # full sensible heat of the solid branch up to theta_m, the full latent
    # heat L*, and liquid sensible heat with slope B above theta_m.
    enthalpy_field = np.empty(nz)
    for i in range(nz):
        th = temperature_field[i]
        if th >= theta_melt:
            enthalpy_field[i] = H_liq_m + B_liq * (th - theta_melt)   # liquid
        else:
            enthalpy_field[i] = th                                    # solid

    # ── Initial inversion pass (FIX 3): sets theta and k consistently ─────────
    for i in range(nz):
        Hi = enthalpy_field[i]
        if Hi >= H_liq_m:
            temperature_field[i] = theta_melt + (Hi - H_liq_m) / B_liq
            k_tilde[i] = k_ratio
        elif Hi <= H_sol_m:
            temperature_field[i] = Hi
            k_tilde[i] = 1.0
        else:
            temperature_field[i] = theta_melt
            fl = (Hi - H_sol_m) / L_star
            if fl < 0.0:
                fl = 0.0
            elif fl > 1.0:
                fl = 1.0
            k_tilde[i] = 1.0 + (k_ratio - 1.0) * fl

    T_profile_at_max_remelt = temperature_field.copy()

    # ═════════════════════════════════════════════════════════════════════════
    # Time loop
    # ═════════════════════════════════════════════════════════════════════════
    for n in range(nt_max):

        # ── Flux divergence update (interior nodes) ───────────────────────────
        enthalpy_field_new = enthalpy_field.copy()

        k_right = (2.0 * k_tilde[1:-1] * k_tilde[2:])   / (k_tilde[1:-1] + k_tilde[2:])
        k_left  = (2.0 * k_tilde[:-2]  * k_tilde[1:-1]) / (k_tilde[:-2]  + k_tilde[1:-1])
        div_q   = (k_right * (temperature_field[2:]   - temperature_field[1:-1])
                 - k_left  * (temperature_field[1:-1] - temperature_field[:-2])) / dz**2
        enthalpy_field_new[1:-1] = enthalpy_field[1:-1] + dt * div_q

        # ── Adiabatic boundary conditions (ghost nodes, unchanged) ────────────
        enthalpy_field_new[0] = enthalpy_field[0] + dt * (
            k_tilde[0] * (temperature_field[1] - temperature_field[0]) / dz**2
        )
        enthalpy_field_new[-1] = enthalpy_field[-1] + dt * (
            k_tilde[-1] * (temperature_field[-2] - temperature_field[-1]) / dz**2
        )

        enthalpy_field = enthalpy_field_new

        # ── H -> T inversion + property update, single pass (FIX 2 + FIX 3) ───
        for i in range(nz):
            Hi = enthalpy_field[i]
            if Hi >= H_liq_m:
                temperature_field[i] = theta_melt + (Hi - H_liq_m) / B_liq
                k_tilde[i] = k_ratio
            elif Hi <= H_sol_m:
                temperature_field[i] = Hi
                k_tilde[i] = 1.0
            else:
                temperature_field[i] = theta_melt
                fl = (Hi - H_sol_m) / L_star
                if fl < 0.0:
                    fl = 0.0
                elif fl > 1.0:
                    fl = 1.0
                k_tilde[i] = 1.0 + (k_ratio - 1.0) * fl

        # ── Interface temperature ─────────────────────────────────────────────
        ki, kj = k_tilde[interface_idx], k_tilde[interface_idx + 1]
        interface_temperature[n] = (ki * temperature_field[interface_idx]
                              + kj * temperature_field[interface_idx + 1]) / (ki + kj)

        # ── Single melt front tracking (sub-cell linear interpolation) ────────
        # Search from interface_idx+1: remelting is only meaningful when a
        # substrate node reaches theta_melt. Using the fraction liquid to interpolate the 
        # position of the melting front
        last_melt = -1
        for j in range(interface_idx + 1, nz):
            if temperature_field[j] >= theta_melt:
                last_melt = j

        if last_melt >= 0:
            Hi = enthalpy_field[last_melt]
            if Hi >= H_liq_m:
                fl_last = 1.0
            elif Hi <= H_sol_m:
                fl_last = 0.0
            else:
                fl_last = (Hi - H_sol_m) / L_star
            z_front = z[last_melt] + (fl_last - 0.5) * dz
            melt_front_position[n] = z_front - z_iface

        # ── Track temperature profile at maximum remelting ────────────────────
        if melt_front_position[n] > current_max_melt:
            current_max_melt = melt_front_position[n]
            T_profile_at_max_remelt = temperature_field.copy()

        # ── Snapshot ──────────────────────────────────────────────────────────
        if snap_ctr < len(snapshot_indices) and n == snapshot_indices[snap_ctr]:
            theta_hist[:, snap_ctr] = temperature_field.copy()
            snap_ctr += 1

    return (
        temperature_field,
        melt_front_position,
        interface_temperature,
        theta_hist,
        snap_ctr,
        T_profile_at_max_remelt,
    )



# ═══════════════════════════════════════════════════════════════════════════════
# 3.  JIT core — Scheil enthalpy method
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════
 
@jit(nopython=True, cache=True)
def _scheil_fl(theta, theta_sol, theta_liq, scheil_exp):
    """
    Scheil liquid fraction  f_l(θ) = ((θ − θ_sol)/(θ_liq − θ_sol))^scheil_exp
    Clamped to [0, 1].  scheil_exp = 1/(1 − k_p).
    """
    if theta <= theta_sol:
        return 0.0
    if theta >= theta_liq:
        return 1.0
    xi = (theta - theta_sol) / (theta_liq - theta_sol)
    return xi ** scheil_exp
 
 
@jit(nopython=True, cache=True)
def _scheil_dfl_dtheta(theta, theta_sol, theta_liq, scheil_exp):
    """Derivative df_l/dθ of the Scheil liquid fraction."""
    if theta <= theta_sol or theta >= theta_liq:
        return 0.0
    xi = (theta - theta_sol) / (theta_liq - theta_sol)
    return scheil_exp / (theta_liq - theta_sol) * xi ** (scheil_exp - 1.0)
 
 
@jit(nopython=True, cache=True)
def _theta_from_H_mushy_newton(
        H_known, theta_sol, theta_liq,
        L_star, scheil_exp,
        max_iter=20, tol=1e-12):
    """
    Invert H → θ in the mushy zone using Newton's method on the
    SOLID-REFERENCE curve:
 
        F(θ) = θ + L* · f_l(θ) − H = 0
        F'(θ) = 1 + L* · df_l/dθ
 
    Initial guess: linear lever-rule approximation
        θ_lin = (H + L*·θ_sol/ΔT_f) / (1 + L*/ΔT_f)
 
    Falls back to bisection towards the violated boundary if Newton wanders
    outside [θ_sol, θ_liq].
 
    NOTE: rho_ratio and cp_ratio are NOT arguments — the solid-reference
    curve absorbs them into the threshold constants H_liq_thresh = θ_liq + L*.
    """
    dtheta_f = theta_liq - theta_sol
 
    # ── Initial guess (exact for lever rule, good for Scheil) ─────────────────
    theta = (H_known + L_star * theta_sol / dtheta_f) / (1.0 + L_star / dtheta_f)
    theta = min(max(theta, theta_sol), theta_liq)
 
    # ── Newton iterations ──────────────────────────────────────────────────────
    for _ in range(max_iter):
        fl  = _scheil_fl(theta, theta_sol, theta_liq, scheil_exp)
        dfl = _scheil_dfl_dtheta(theta, theta_sol, theta_liq, scheil_exp)
 
        F  = theta + L_star * fl - H_known
        dF = 1.0 + L_star * dfl
 
        if abs(dF) < 1e-30:
            break
 
        theta_new = theta - F / dF
 
        # Bisect back into bracket if Newton escapes
        if theta_new < theta_sol:
            theta_new = 0.5 * (theta_sol + theta)
        elif theta_new > theta_liq:
            theta_new = 0.5 * (theta + theta_liq)
 
        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
 
    return min(max(theta, theta_sol), theta_liq)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# JIT core
# ═══════════════════════════════════════════════════════════════════════════════
 
@jit(nopython=True, cache=True)
def _fd_solver_core_scheil(
        theta_liquidus, theta_solidus, Ste_m, k_p,
        k_ratio, rho_ratio, cp_ratio,
        z, interface_idx,
        dt, nt_max, snapshot_indices):
    """
    JIT-compiled time loop for the Scheil enthalpy solver.
    """
 
    nz         = len(z)
    dz         = z[1] - z[0]
    z_iface = z[interface_idx] + 0.5 * dz      # == h exactly

    scheil_exp = 1.0 / (1.0 - k_p)
    B          = rho_ratio * cp_ratio               # (ρcp)_l / (ρcp)_s
    L_star     = (theta_liquidus - theta_solidus) / Ste_m  # latent heat, solid-ref
 
    # ── Fixed thresholds (CONSTANTS) ──────────────────────────────────────────
    H_sol_thresh = theta_solidus                    # solid / mushy boundary
    H_liq_thresh = theta_liquidus + L_star          # mushy / liquid boundary
 
    # ── Only conductivity is phase-dependent ──────────────────────────────────
    k_tilde = np.ones(nz)
 
    # ── Output arrays ─────────────────────────────────────────────────────────
    interface_temperature   = np.zeros(nt_max)
    liquidus_front_position = np.zeros(nt_max)
    solidus_front_position  = np.zeros(nt_max)
    theta_hist              = np.zeros((nz, len(snapshot_indices)))
    snap_ctr                = 0
    current_max_liq         = 0.0
 
    # ── Initial temperature field ──────────────────────────────────────────────
    temperature_field = np.zeros(nz)
    for i in range(nz):
        temperature_field[i] = 1.0 if z[i] <= z[interface_idx] else 0.0
 
    # ── Initial enthalpy field — solid-reference curve ────────────────────────
    # At init, every node is either fully liquid (deposit, θ=1) or fully solid
    # (substrate, θ=0). No mushy nodes at t=0.
    #
    #   deposit (θ=1 ≥ θ_liq): H = H_liq_thresh + B·(1 − θ_liq)
    #   substrate (θ=0 ≤ θ_sol): H = 0
    #
    # This is purely algebraic — rho_tilde/cp_tilde play no role.
    enthalpy_field = np.empty(nz)
    for i in range(nz):
        th = temperature_field[i]
        if th >= theta_liquidus:
            enthalpy_field[i] = H_liq_thresh + B * (th - theta_liquidus)
        elif th <= theta_solidus:
            enthalpy_field[i] = th
        else:
            # Should not occur at init, but handle gracefully
            fl = _scheil_fl(th, theta_solidus, theta_liquidus, scheil_exp)
            enthalpy_field[i] = th + L_star * fl
 
    # ── Initial inversion pass: set k_tilde consistently ─────────────────────
    for i in range(nz):
        Hi = enthalpy_field[i]
        if Hi >= H_liq_thresh:
            k_tilde[i] = k_ratio
        elif Hi <= H_sol_thresh:
            k_tilde[i] = 1.0
        else:
            th = temperature_field[i]
            fl = _scheil_fl(th, theta_solidus, theta_liquidus, scheil_exp)
            k_tilde[i] = 1.0 + (k_ratio - 1.0) * fl
 
    T_profile_at_max_remelt = temperature_field.copy()
 
    # ═════════════════════════════════════════════════════════════════════════
    # Time loop
    # ═════════════════════════════════════════════════════════════════════════
    for n in range(nt_max):
 
        # ── Flux divergence (interior nodes) ──────────────────────────────────
        enthalpy_field_new = enthalpy_field.copy()
 
        k_right = (2.0 * k_tilde[1:-1] * k_tilde[2:])   / (k_tilde[1:-1] + k_tilde[2:])
        k_left  = (2.0 * k_tilde[:-2]  * k_tilde[1:-1]) / (k_tilde[:-2]  + k_tilde[1:-1])
        div_q   = (k_right * (temperature_field[2:]   - temperature_field[1:-1])
                 - k_left  * (temperature_field[1:-1] - temperature_field[:-2])) / dz**2
        enthalpy_field_new[1:-1] = enthalpy_field[1:-1] + dt * div_q
 
        # ── Adiabatic BCs (ghost nodes) ───────────────────────────────────────
        enthalpy_field_new[0] = enthalpy_field[0] + dt * (
            k_tilde[0] * (temperature_field[1] - temperature_field[0]) / dz**2
        )
        enthalpy_field_new[-1] = enthalpy_field[-1] + dt * (
            k_tilde[-1] * (temperature_field[-2] - temperature_field[-1]) / dz**2
        )
 
        enthalpy_field = enthalpy_field_new
 
        # ── H → θ inversion + k update, single pass ───────────────────────────
        # Uses FIXED thresholds — no rho/cp state dependency.
        for i in range(nz):
            Hi = enthalpy_field[i]
 
            if Hi >= H_liq_thresh:
                # Liquid branch (analytic)
                temperature_field[i] = theta_liquidus + (Hi - H_liq_thresh) / B
                k_tilde[i] = k_ratio
 
            elif Hi <= H_sol_thresh:
                # Solid branch (analytic)
                temperature_field[i] = Hi
                k_tilde[i] = 1.0
 
            else:
                # Mushy branch — Newton inversion of solid-reference Scheil curve
                temperature_field[i] = _theta_from_H_mushy_newton(
                    Hi, theta_solidus, theta_liquidus,
                    L_star, scheil_exp
                )
                fl = _scheil_fl(temperature_field[i],
                                theta_solidus, theta_liquidus, scheil_exp)
                k_tilde[i] = 1.0 + (k_ratio - 1.0) * fl
 
        # ── Interface temperature ──────────────────────────────────────────────
        ki = k_tilde[interface_idx]
        kj = k_tilde[interface_idx + 1]
        interface_temperature[n] = (ki * temperature_field[interface_idx]
            + kj * temperature_field[interface_idx + 1]) / (ki + kj)
 
        # ── Front tracking (sub-cell linear interpolation) ────────────────────
        last_liq = -1
        last_sol = -1
        for j in range(interface_idx + 1, nz):
            if temperature_field[j] > theta_liquidus:
                last_liq = j
            if temperature_field[j] > theta_solidus:
                last_sol = j
 
        if 0 <= last_liq < nz - 1:
            fl_liq = _scheil_fl(temperature_field[last_liq + 1],
                theta_solidus, theta_liquidus, scheil_exp)
            z_front = z[last_liq + 1] + (fl_liq - 0.5) * dz
            liquidus_front_position[n] = z_front - z_iface
        elif last_liq == nz - 1:
            liquidus_front_position[n] = nz*dz - z_iface
 
        if last_sol >= 0:
            fl_sol = _scheil_fl(temperature_field[last_sol],
                theta_solidus, theta_liquidus, scheil_exp)
            z_front = z[last_sol] + (fl_sol - 0.5) * dz
            solidus_front_position[n] = z_front - z_iface
 
        # ── Track temperature profile at maximum remelting ────────────────────
        if liquidus_front_position[n] > current_max_liq:
            current_max_liq = liquidus_front_position[n]
            T_profile_at_max_remelt = temperature_field.copy()
 
        # ── Snapshot ──────────────────────────────────────────────────────────
        if snap_ctr < len(snapshot_indices) and n == snapshot_indices[snap_ctr]:
            theta_hist[:, snap_ctr] = temperature_field.copy()
            snap_ctr += 1
 
    return (
        temperature_field,
        liquidus_front_position,
        solidus_front_position,
        interface_temperature,
        theta_hist,
        snap_ctr,
        T_profile_at_max_remelt,
    )
 
