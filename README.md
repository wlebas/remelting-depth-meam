# Predicting the Substrate Remelting Depth in Liquid Metal Deposition Processes

Analytical and numerical framework for predicting inter-layer remelting during
metal extrusion additive manufacturing (MEAM). The code implements a
one-dimensional enthalpy-based finite-difference solver for the two-domain
Stefan problem (hot deposit on cold substrate), a dimensionless
Latin-hypercube sweep driver, and the analysis pipeline used to derive and
validate the correlations reported in the accompanying manuscript(s).

Two solidification models are supported:

- **Pure metal / eutectic** — isothermal phase change at `θ_m`.
- **Scheil** — non-equilibrium mushy-zone solidification.

The problem is parameterised by the dimensionless groups swept in
`dimensionless_sweep_pure.py`: effusivity ratio `E`, heat-capacity ratio `C`,
diffusivity ratio `r`, Stefan numbers `Ste_hot` and `Ste_cold`, and the
normalised melting temperature `θ_m`.

## Repository contents

```
.
├── kernels.py                    # Numba FD solver cores (pure-metal, Scheil)
├── solvers.py                    # fd_solve — top-level solver entry point
├── grid.py                       # grid construction (build_grid) and run sizing (size_run)
├── materials.py                  # dimensional properties → dimensionless groups
├── analytical.py                 # Stefan / Neumann similarity solutions
├── conventions.py                # normalisation conventions + self-check suite
├── config_pipeline.py            # CONVENTIONS / PARAMS, config fingerprinting
├── dimensionless_sweep_pure.py   # LHS sweep driver (pure-metal / eutectic)
├── dimensionless_sweep_scheil.py # LHS sweep driver (Scheil)
├── analysis_pipeline.py          # post-processing, correlation fitting, metrics
├── case_study.py                 # single-case study / dimensional worked example
├── Dockerfile                    # container definition used for cluster runs
├── requirements.txt
└── progress - Pure Full Results Original - Copy.csv   # canonical pure-metal sweep output
```

## Installation

```bash
git clone https://github.com/wlebas/remelting-depth-meam.git
cd remelting-depth-meam
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

## Usage

**Verify conventions and solver self-checks:**

```bash
python conventions.py
```

Optionally audit an existing results file:

```bash
python conventions.py --csv "progress - Pure Full Results Original - Copy.csv"
```

**Run the dimensionless sweep (pure-metal / eutectic):**

```bash
python dimensionless_sweep_pure.py --n_lhs 1000 --workers 8 --output results_pure.csv
```

Useful flags: `--resume` (continue an interrupted sweep from its checkpoint),
`--seed` (reproducible LHS), `--dry_run` (build the job list without solving),
`--neumann` (Neumann-comparison mode). The Scheil driver
(`dimensionless_sweep_scheil.py`) takes the same interface.

**Post-process and fit correlations:**

```bash
python analysis_pipeline.py --csv "progress - Pure Full Results Original - Copy.csv" \
                            --model pure \
                            --outdir analysis_out
```

`--model` is required and selects `pure` or `scheil`. Phases can be selected
with `--phases` (default `0,1,2,3,4,5`); `--no-plots` suppresses figure
generation.

## Data

`progress - Pure Full Results Original - Copy.csv` is the canonical pure-metal
sweep output referenced in the manuscript. Each row is one
(material point × operating point) case; columns comprise the swept
dimensionless groups, derived quantities (`lam_neumann`, `k_ratio`,
`rho_ratio`, …), solver outputs (`D_rem`, `t_at_max_remelt`, `Z_total`), and
provenance flags (`refined`, `time_capped`, `domain_capped`, `status`,
`config_fingerprint`).

## License

MIT — see [LICENSE](LICENSE).

