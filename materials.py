"""
This is the library of materials and the adimensionalisation function, which are then used to construct the grid in grid.py 
and run the solver in rewrite_clean_.py.  
While this library exists, dimensional groups can be passed directly to the kernels to elicit the underlying physics without the need for a material library. 
However, the material library provides a convenient way to run cases with real materials and compare with experiments.  
The adimensionalisation function computes the dimensionless groups from the material properties and case parameters, which are then used in the solver.
The materials are defined as dictionaries of properties, and the adimensionalisation function computes the dimensionless groups from these properties and the case parameters.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Sources:
#   316L  — NIST SRM 1155a: Pottlacher et al., J. Mater. Sci. 2019; Mills 2002
#   IN718 — Hosaeus et al., High Temp–High Press 2001; Mills 2002
#   IN625 — Kaschnitz & Pottlacher, Int. J. Thermophys. 2019; Mills 2002
#   Ti64  — Boivineau et al., Int. J. Thermophys. 2006; Mohr et al., Adv. Eng. Mater. 2020
#   Cu    — Mills 2002; Touloukian Vol.1 & Vol.10
#   ER70S — Ding & Colegrove, J. Mater. Process. Technol. 2011; Mills 2002
#   H625  — Haynes International datasheet; Mills 2002 (same alloy as IN625, different source)
#   AlSi10Mg — Kempen et al., Phys. Procedia 2012; Aboulkhair et al., Addit. Manuf. 2019
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ALUMINIUM ALLOYS
# ═══════════════════════════════════════════════════════════════════════════════

AL_SI12 = {
    'material': 'AlSi12',
    # Near-eutectic Al-Si.  Narrow mushy zone (4 °C, 0.7 % of T_sol).
    # Lowest k_p in the set — Si strongly rejects to liquid.
    # Ste_s = 0.700.  A×Ste confirmed 0.110 (within universal band).
    'T_liq': 581.0,  'T_sol': 577.0,
    'k_s': 121.0,    'k_l': 91.8,
    'rho_s': 2657.0, 'rho_l': 2371.0,
    'cp_s': 963.0,   'cp_l': 1127.5,
    'L': 3.89e5,
    'k_p': 0.13, 'model': 'scheil',  # Scheil model required to capture near-eutectic behaviour
}

# Al-6061 (heat-treatable structural aluminium, common WAAM substrate/feedstock)
# Wider freezing range (~95 K) relative to near-eutectic alloys — significant mushy zone.
# theta_s ≈ 0.60, Ste_m ≈ 0.27, q ≈ 2.6 — the most mushy aluminium alloy in this set.
# Important because 6061 substrates are commonly used with Al-Si wire deposition.
AL_6061 = {
    'material': 'Al6061',
    # Heat-treatable structural Al.  Mushy zone 70 °C (12 % of T_sol).
    # Most mushy Al alloy in this set.
    # Ste_s = 0.759.  A×Ste confirmed 0.096.
    'T_liq': 652.0,  'T_sol': 582.0,
    'k_s': 167.0,    'k_l': 100.0,
    'rho_s': 2700.0, 'rho_l': 2380.0,
    'cp_s': 896.0,   'cp_l': 1080.0,
    'L': 3.96e5,
    'k_p': 0.30, 'model': 'scheil',
}
 
AL_2024 = {
    'material': 'Al2024',
    # Cu-dominant Al alloy.  Widest mushy zone in set (136 °C, 27 % of T_sol).
    # D_rem and D_sol decouple strongly — fit both targets separately.
    # Ste_s = 0.879.  A×Ste confirmed 0.097.
    'T_liq': 638.0,  'T_sol': 502.0,
    'k_s': 121.0,    'k_l': 75.0,
    'rho_s': 2780.0, 'rho_l': 2420.0,
    'cp_s': 875.0,   'cp_l': 1040.0,
    'L': 3.86e5,
    'k_p': 0.17, 'model': 'scheil',
}

# Al-10Si-Mg (AlSi10Mg — most common SLM aluminium alloy)
# Near-eutectic Al-Si, very small freezing range (~10 K).
# theta_s ≈ 0.93, Ste_m ≈ 0.02 — slightly wider mushy zone than Al-Si12.
# Allows direct comparison with Al-Si12 to show the effect of Mg addition.
AL_SI10MG = {
    'material': "Al-Si10Mg",
    'T_liq': 600.0,  'T_sol': 570.0, 
    'k_s': 130.0,     'k_l': 90.0,   
    'rho_s': 2670.0, 'rho_l': 2380.0, 
    'cp_s': 900.0, 'cp_l': 1050.0,
    'L': 4.25e5, 'k_p': 0.13,  'model': 'scheil',
}

PURE_AL = {
    'material': "Pure Al",
    'T_liq': 660.0, 'T_sol': 659.9,
    'k_s': 237.0,  'rho_s': 2700.0, 'cp_s': 900.0,
    'k_l': 91.0,   'rho_l': 2375.0, 'cp_l': 1080.0,
    'L': 3.97e5, 'k_p': 1.0,  'model': 'pure_metal',
}

AL_CU45 = {
    'material': "Al-Cu45",
    'T_liq': 646.0, 'T_sol': 548.0, 
    'k_s': 190.0,  'rho_s': 2780.0, 'cp_s': 920.0,
    'k_l': 85.0,   'rho_l': 2450.0, 'cp_l': 1050.0,
    'L': 3.89e5, 'k_p': 0.17, 'model': 'scheil',
}

# ══════════════════════════════════════════════════════════════════════════════
#  COPPER ALLOYS
# ══════════════════════════════════════════════════════════════════════════════
 
CU_PURE = {
    'material': 'Cu_pure',
    # Pure OFHC copper.  Near-zero mushy zone (5 °C, 0.5 % of T_sol).
    # k_p = 1.0 → use pure_metal solver; mushy eigenvalues undefined.
    # α_l/α_s = 0.37 (liquid less diffusive) — no CFL issue.
    # Theoretical end-member: highest α_s (112 mm²/s), k_p = 1.0 sharp interface.
    # Ste_s = 0.503.  Predicted A×Ste ≈ 0.095.
    'T_liq': 1085.0, 'T_sol': 1080.0,
    'k_s': 385.0,    'k_l': 165.0,
    'rho_s': 8960.0, 'rho_l': 8000.0,
    'cp_s': 385.0,   'cp_l': 495.0,
    'L': 2.09e5,
    'k_p': 1.00,
    'model': 'pure_metal',
}
 
CU_SN10 = {
    'material': 'CuSn10',
    # Tin bronze.  Wide mushy zone (150 °C, 17.6 % of T_sol).
    # Lowest k_p after AlSi12 — Sn strongly rejected to liquid.
    # α_l/α_s = 0.49 — no CFL issue.  Deposit CFL = 0.19.
    # Ste_s = 0.557.  In original report Table 1: R²(D_rem)=0.951.
    'T_liq': 1000.0, 'T_sol': 850.0,
    'k_s': 50.0,     'k_l': 28.0,
    'rho_s': 8800.0, 'rho_l': 8000.0,
    'cp_s': 380.0,   'cp_l': 480.0,
    'L': 1.80e5,
    'k_p': 0.16,
    'model': 'scheil',
}
 
# ══════════════════════════════════════════════════════════════════════════════
#  NICKEL-BASED SUPERALLOYS
# ══════════════════════════════════════════════════════════════════════════════
 
 # Inconel 625 (nickel superalloy)
# Similar to IN718 but wider freezing range (~100 K) — extreme mushy zone.
# theta_s ≈ 0.43, Ste_m ≈ 0.24, q ≈ 2.8 — the most mushy material in this set.
# Popular for corrosion-resistant WAAM cladding applications.
INCONEL_625 = {
    'material': 'Inconel625',
    # Corrosion-resistant Ni superalloy.  Mushy zone 60 °C (4.7 % of T_sol).
    # Nb/Mo partition to liquid.
    # Ste_s = 0.514.  A×Ste confirmed 0.081.
    'T_liq': 1350.0, 'T_sol': 1290.0,
    'k_s': 10.8,     'k_l': 17.0,
    'rho_s': 8440.0, 'rho_l': 7600.0,
    'cp_s': 410.0,   'cp_l': 580.0,
    'L': 2.72e5,
    'k_p': 0.60, 'model': 'scheil',
}
 
# Inconel 718 (nickel superalloy, most common AM superalloy)
# Large freezing range (~80 K), very low conductivity — strongly mushy behaviour.
# theta_s ≈ 0.47, Ste_m ≈ 0.20, q ≈ 3.3 — well outside sharp-interface limit.
# Extremely well characterised; key reference for WAAM and SLM papers.
INCONEL_718 = {
    'material': 'Inconel718',
    # Precipitation-hardened Ni superalloy.  Mushy zone 76 °C (6.0 % of T_sol).
    # Nb partitioning drives Laves phase formation.
    # T_sol = 1260 per master_summary (canonical; earlier analysis used 1255).
    # Ste_s = 0.474.  A×Ste confirmed 0.104.
    'T_liq': 1336.0, 'T_sol': 1260.0,
    'k_s': 11.4,     'k_l': 16.5,
    'rho_s': 8190.0, 'rho_l': 7420.0,
    'cp_s': 435.0,   'cp_l': 595.0,
    'L': 2.60e5,
    'k_p': 0.65, 'model': 'scheil',
}
 
WASPALOY = {
    'material': 'Waspaloy',
    # Precipitation-hardened Ni superalloy for turbine disc repair by DED/WAAM.
    # Mushy zone 125 °C (10.2 % of T_sol) — widest Ni-alloy mushy zone in set.
    # Fills critical gap: all other Ni/Fe alloys have < 7 % mushy.
    # α_l/α_s = 1.28 → deposit CFL = 0.51 — REQUIRES CFL FIX (grid_builder).
    # Ste_s = 0.503.  Predicted A×Ste ≈ 0.095.
    # Refs: Special Metals datasheet; Zhao et al. (2019) J. Alloys Compd.
    'T_liq': 1355.0, 'T_sol': 1230.0,
    'k_s': 11.7,     'k_l': 19.0,     
    'rho_s': 8190.0, 'rho_l': 7400.0,
    'cp_s': 420.0,   'cp_l': 590.0,
    'L': 2.60e5,
    'k_p': 0.50, 'model': 'scheil',
}
 
NI_PURE = {
    'material': 'Ni_pure',
    # Pure nickel (Ni 200/201).  Near-zero mushy zone (2 °C, 0.1 % of T_sol).
    # k_p = 1.0 → use pure_metal solver; mushy eigenvalues undefined.
    # α_l/α_s = 0.57 — no CFL issue.  α_s = 23 mm²/s.
    # Second pure-metal end-member alongside Cu_pure; tests universality of
    # the 2-param model at a very different conductivity (91 vs 385 W/mK).
    # Ste_s = 0.453.  Predicted A×Ste ≈ 0.095.
    # Refs: Lide (ed.) CRC Handbook; Smithells Metals Reference Book.
    'T_liq': 1455.0, 'T_sol': 1453.0,
    'k_s': 91.0,     'k_l': 60.0,
    'rho_s': 8908.0, 'rho_l': 7810.0,
    'cp_s': 444.0,   'cp_l': 590.0,
    'L': 2.92e5,
    'k_p': 1.00,
    'model': 'pure_metal',
}
 
# ══════════════════════════════════════════════════════════════════════════════
#  STAINLESS STEELS
# ══════════════════════════════════════════════════════════════════════════════

# 316L stainless steel (NIST SRM 1155a)
# Wide freezing range (~50 K), low thermal conductivity — mushy zone is physically significant.
# theta_s ≈ 0.56, Ste_m ≈ 0.14, q ≈ 4.2 — the mushy zone will be much wider than Al-Si12.
# Canonical AM/WAAM material, excellent published data.
SS316L = {
    'material': 'SS316L',
    # Austenitic stainless.  Mushy zone 75 °C (5.5 % of T_sol).
    # T_sol = 1375 per master_summary (canonical; earlier analysis used 1400).
    # Ste_s = 0.396.  A×Ste confirmed 0.100.
    'T_liq': 1450.0, 'T_sol': 1375.0,
    'k_s': 16.3,     'k_l': 22.0,
    'rho_s': 8000.0, 'rho_l': 7200.0,
    'cp_s': 500.0,   'cp_l': 650.0,
    'L': 2.72e5,
    'k_p': 0.80,
    'model': 'scheil',
}
 
SS304 = {
    'material': 'SS304',
    # Austenitic stainless.  Nearly identical feature space to SS316L.
    # Treated as validation material, not added to universal model training.
    # Ste_s = 0.388.
    'T_liq': 1455.0, 'T_sol': 1400.0,
    'k_s': 16.2,     'k_l': 21.5,
    'rho_s': 8000.0, 'rho_l': 7200.0,
    'cp_s': 500.0,   'cp_l': 650.0,
    'L': 2.72e5,
    'k_p': 0.82, 'model': 'scheil',
}
 
SS2507 = {
    'material': 'SS2507',
    # Super-duplex stainless (25Cr-7Ni-4Mo-0.3N).  Mushy zone 110 °C (8.2 %).
    # Significantly wider than SS316L (5.5 %) — breaks Fe-alloy mushy-zone
    # degeneracy, tests dT_Ts sensitivity outside the Al cluster.
    # α_l/α_s = 1.11 → deposit CFL = 0.44.  No CFL issue.
    # Ste_s = 0.420.  Predicted A×Ste ≈ 0.095.
    # Refs: Nilsson (1992) Mater. Sci. Technol.; Sieurin & Sandström (2007).
    'T_liq': 1450.0, 'T_sol': 1340.0,
    'k_s': 14.0,     'k_l': 18.0,
    'rho_s': 7800.0, 'rho_l': 7000.0,
    'cp_s': 480.0,   'cp_l': 620.0,
    'L': 2.70e5,
    'k_p': 0.72, 'model': 'scheil',
}
 
# ══════════════════════════════════════════════════════════════════════════════
#  COBALT ALLOYS
# ══════════════════════════════════════════════════════════════════════════════
 
COCR_F75 = {
    'material': 'CoCr_F75',
    # ASTM F75 cobalt-chrome (62Co-28Cr-6Mo).
    # Extremely common in AM: medical implants, aerospace tooling, dental.
    # Mushy zone 110 °C (8.3 % of T_sol).
    # eff/Ste = 1.86 fills the gap between IN625 (1.37) and SS316L (2.01).
    # α_l/α_s = 1.11 → deposit CFL = 0.44.  No CFL issue.
    # Ste_s = 0.418.  Predicted A×Ste ≈ 0.095.
    # Refs: ASM Handbook Vol 2; Boyer et al. 2010 (Materials Properties Handbook).
    'T_liq': 1440.0, 'T_sol': 1330.0,
    'k_s': 14.0,     'k_l': 19.0,
    'rho_s': 8400.0, 'rho_l': 7700.0,
    'cp_s': 450.0,   'cp_l': 600.0,
    'L': 2.50e5,
    'k_p': 0.55, 'model': 'scheil',
}
 
# ══════════════════════════════════════════════════════════════════════════════
#  TITANIUM ALLOYS
# ══════════════════════════════════════════════════════════════════════════════
 
# Ti-6Al-4V (alpha-beta titanium)
# Small freezing range (~35 K), but extremely low conductivity in both phases.
# theta_s ≈ 0.87, Ste_m ≈ 0.09, q ≈ 5.7 — somewhat mushy.
# Most widely studied AM titanium alloy. ISS measurements (Mohr 2020) for liquid phase.
TI64 = {
    'material': 'Ti6Al4V',
    # Most widely studied AM titanium alloy.  Mushy zone 56 °C (3.5 % of T_sol).
    # CRITICAL: α_l/α_s = 2.92 → deposit CFL = 1.17 — REQUIRES CFL FIX.
    #   Liquid is ~3× more diffusive than solid (k_l/k_s = 3.4, unique in set).
    #   Without fix: numerical instability cascades to domain boundary after
    #   ~7000 steps, producing spurious D_rem ≈ 22h for all (T_dep, T_sub).
    # Ste_s = 0.339.  Predicted A×Ste ≈ 0.095 → A ≈ 0.28.
    # Refs: Mills (2002) Recommended values of thermophysical properties;
    #       Mohr et al. (2020) ISS measurements for liquid phase.
    'T_liq': 1660.0, 'T_sol': 1604.0,
    'k_s': 7.0,      'k_l': 24.0,     
    'rho_s': 4420.0, 'rho_l': 3900.0,
    'cp_s': 526.0,   'cp_l': 700.0,
    'L': 2.86e5,
    'k_p': 0.95, 'model': 'scheil',
}
 
# ══════════════════════════════════════════════════════════════════════════════
#  CERMETS
# ══════════════════════════════════════════════════════════════════════════════
 
WCCO_88_12 = {
    'material': 'WCCo_88_12',
    # WC-Co 88/12 cermet (thermal spray feedstock).
    # Treated as pseudo-alloy: Co binder melts ~1495 °C; WC particles stay solid.
    # WARNING: the pseudo-alloy model is physically questionable.
    #   Only 12 wt% Co melts, so effective composite latent heat ≈ 0.12 × L.
    #   Using bulk L overestimates the energy barrier → anomalously small A.
    #   A×Ste_s = 0.046 (confirmed outlier; all metallic alloys: 0.065–0.110).
    #   Treat D_sol as reliable; D_rem predictions carry higher uncertainty.
    # Ste_s = 0.748.  A×Ste = 0.046 (cermet outlier — exclude from universal fit).
    'T_liq': 1495.0, 'T_sol': 1380.0,
    'k_s': 63.0,     'k_l': 45.0,
    'rho_s': 14500.0,'rho_l': 13000.0,
    'cp_s': 250.0,   'cp_l': 350.0,
    'L': 2.58e5,
    'k_p': 0.40, 'model': 'scheil',
}
 
# ══════════════════════════════════════════════════════════════════════════════
#  ZINC ALLOYS
# ══════════════════════════════════════════════════════════════════════════════
 
ZN_AL4 = {
    'material': 'ZnAl4',
    # Zn-4Al die-casting alloy.  Near-eutectic, narrow mushy zone (6 °C, 1.6 %).
    # Lowest T_sol in set (381 °C) — confirms T_sol normalisation works correctly
    # at low temperatures.  R²_CV = 0.983 (D_rem), 0.987 (D_sol).
    # Ste_s = 0.708.  A×Ste confirmed 0.107 — within metallic band.
    'T_liq': 387.0,  'T_sol': 381.0,
    'k_s': 109.0,    'k_l': 60.5,
    'rho_s': 6600.0, 'rho_l': 6200.0,
    'cp_s': 419.0,   'cp_l': 480.0,
    'L': 1.13e5,
    'k_p': 0.45, 'model': 'scheil',
}
 
# ══════════════════════════════════════════════════════════════════════════════
#  LOW-CARBON STEEL
# ══════════════════════════════════════════════════════════════════════════════
 
# ER70S-6 mild steel (most common WAAM wire feedstock)
# Moderate freezing range (~50 K), moderate conductivity.
# theta_s ≈ 0.60, Ste_m ≈ 0.13 — intermediate mushy behaviour.
# Directly relevant to the largest installed base of WAAM machines worldwide. 
ER70S = {
    'material': 'ER70S-6',
    # Most common WAAM wire feedstock.  Mushy zone 50 °C (3.4 % of T_sol).
    # Highest eff/Ste (3.31) in set — outlier on that axis.
    # A×Ste = 0.065 (lowest metallic alloy; high eff/Ste drives this).
    # Ste_s = 0.303.
    'T_liq': 1500.0, 'T_sol': 1450.0,
    'k_s': 38.0,     'k_l': 35.0,
    'rho_s': 7800.0, 'rho_l': 7000.0,
    'cp_s': 620.0,   'cp_l': 750.0,
    'L': 2.72e5,
    'k_p': 0.85, 'model': 'scheil',
}


MATERIAL_LIBRARY = {
    # ── Aluminium alloys ──────────────────────────────────────────────
    'AlSi12':      AL_SI12,
    'Al6061':      AL_6061,
    'Al2024':      AL_2024,
    'AlSi10Mg':    AL_SI10MG,
    'AlCu45':      AL_CU45,
    'Pure_Al':     PURE_AL,
    # ── Copper / bronze ───────────────────────────────────────────────
    'Cu_pure':     CU_PURE,
    'CuSn10':      CU_SN10,
    # ── Zinc ──────────────────────────────────────────────────────────
    'ZnAl4':       ZN_AL4,
    # ── Nickel / superalloys ──────────────────────────────────────────
    'Ni_pure':     NI_PURE,
    'Waspaloy':    WASPALOY,
    'Inconel625':  INCONEL_625,
    'Inconel718':  INCONEL_718,
    # ── Steel / iron ──────────────────────────────────────────────────
    'ER70S-6':     ER70S,
    'SS304':       SS304,
    'SS316L':      SS316L,
    'SS2507':      SS2507,
    # ── Cobalt ────────────────────────────────────────────────────────
    'CoCr_F75':    COCR_F75,
    # ── Titanium ──────────────────────────────────────────────────────
    'Ti6Al4V':     TI64,
    # ── Cemented carbide ──────────────────────────────────────────────
    'WC-Co_88_12': WCCO_88_12,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Adimensionalisation
# ═══════════════════════════════════════════════════════════════════════════════
"""
This section contains the adimensionalisation function, which transforms the physical parameters and variables into their dimensionless counterparts. 
This process simplifies the governing equations and reduces the number of parameters, making it easier to analyze the system's behavior 
and identify key dimensionless groups that govern the dynamics of the heat diffusion process.
"""

def compute_adim_groups(props):
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

    Temperature scale is FULL SWING: dT_ref = T_dep − T_sub. (The earlier
    function used T_liq − T_sub, giving a wrong inv_Ste for superheated
    deposits.) The scheil exponent and inv_Ste_dT are recomputed inside the
    jit kernels, so they are not exported here.
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
    dT_ref = T_dep - T_sub
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

    # ── Mushy-zone mean properties (for Scheil sweep) ─────────────────────────
    k_m   = 0.5 * (p['k_l'] + p['k_s'])
    rho_m = 0.5 * (p['rho_l'] + p['rho_s'])
    cp_m  = 0.5 * (p['cp_l'] + p['cp_s'])
    Ste_m = cp_m * dT_f / p['L']

    # Latent-heat reference (solid-referenced, matches kernel choice):
    # L* = inv_Ste_dT = (θ_liq − θ_sol) / Ste_m
    inv_Ste_dT = (theta_liquidus - theta_solidus) / Ste_m

    k_p = float(p.get('k_p', 1.0))

    # ── Onset criterion / mushy fraction ──────────────────────────────────────
    theta_contact = E / (1.0 + E)   # two-body contact temperature
    mushy_frac    = theta_solidus / theta_liquidus if theta_liquidus > 0 else np.nan

    # ── Analytical similarity-solver groups ───────────────────────────────────
    alpha_m   = k_m / (rho_m * cp_m)
    alpha_eff = alpha_m * Ste_m / (1.0 + Ste_m)   # apparent-heat-capacity smear
    r_eff     = alpha_eff / alpha_s

    eff_m   = np.sqrt(rho_m * cp_m * k_m)
    eff_LM  = eff_l / eff_m
    eff_SM  = eff_s / eff_m
    E_ratio = E                       # alias the Neumann solver expects

    Ste_superheat    = p['cp_l'] * (T_dep - T_liq) / p['L']
    Ste_undercooling = p['cp_s'] * (T_liq - T_sub) / p['L']

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
        # ── Derived ratios (solver inputs) ────────────────────────────────────
        'k_ratio'         : k_ratio,
        'rho_ratio'       : rho_ratio,
        'cp_ratio'        : cp_ratio,
        # ── Stefan numbers ────────────────────────────────────────────────────
        'inv_Ste'         : inv_Ste,
        'Ste_m'           : Ste_m,
        # ── Latent-heat reference (kernel convention, uses (ρcp)_s) ────────────
        'L_star'          : inv_Ste_dT,
        'B'               : rho_ratio * cp_ratio,
        # ── Analytical similarity-solver groups ───────────────────────────────
        'theta_deposit'    : 1.0,
        'theta_substrate'  : 0.0,
        'r_eff'            : r_eff,
        'eff_LM'           : eff_LM,
        'eff_SM'           : eff_SM,
        'E_ratio'          : E_ratio,
        'Ste_superheat'    : Ste_superheat,
        'Ste_undercooling' : Ste_undercooling,
    }