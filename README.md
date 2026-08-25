# Inter-Layer Remelting in Metal Extrusion Additive Manufacturing

Analytical and numerical framework for predicting inter-layer remelting during
metal extrusion additive manufacturing (MEX/WAAM), developed as part of a
two-paper series:

- **Paper 1** — Melt/no-melt onset classifier: closed-form existence condition
  `E/(1+E) > θ_liq` derived from the λ→0 limit of the Neumann similarity
  solution, validated numerically (AUC ≈ 1.0).
- **Paper 2** — Remelting depth correlation:
  `D_rem ≈ 1.69 · Π_c^0.775 · margin`, alongside a fully-derived Lambert-W
  chain (zero fitted constants).

Six governing dimensionless groups (effusivity ratio `E`, heat capacity ratio
`C`, diffusivity ratio `r`, `Ste_hot`, `Ste_cold`, and `margin =
E/(1+E) − θ_liq`) parameterize both the onset criterion and the depth model.
Process maps are generated via finite-difference sweeps across the pure-metal
set (AlSi12, Pure Al, Pure Cu, Pure Ti) and a broader alloy set (Scheil
solidification track).

## Status

Preprint / manuscript in preparation, target venue IJHMT. This repository
will be tagged at the commit corresponding to submission (`v1.0-submission`)
once ready.

## Repository structure

```
.
├── kernels.py                          # FD solver cores (pure-metal, Scheil)
├── solvers.py                          # fd_solve — top-level solver entry point
├── grid.py                             # grid construction / sizing
├── materials.py                        # dimensionless group computation
├── analytical.py                       # Stefan/Neumann similarity solutions
├── conventions.py                      # sign/normalisation conventions + checks
├── config_pipeline.py                  # CONVENTIONS / PARAMS, config fingerprinting
├── dimensionless_sweep_pure.py         # LHS sweep generation & job runner (pure)
├── dimensionless_sweep_scheil.py       # LHS sweep generation & job runner (Scheil)
├── analysis_pipeline.py                # post-processing / metrics / correlation fitting
├── time_single_case_v2.py              # single-case timing/profiling (pure)
├── time_single_case_scheil.py          # single-case timing/profiling (Scheil)
├── progress__Pure_Full_Results_Original.csv   # canonical pure-metal results dataset
└── __Verification___Analysis_Plan_v4___Pure.md # verification & analysis plan
```

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

## Reproducing a result

Run a single case (useful sanity check / timing baseline):

```bash
python time_single_case_v2.py
```

Run the full dimensionless sweep (pure-metal set):

```bash
python dimensionless_sweep_pure.py
```

> Full sweeps (tens of thousands of cases) were run on EPFL RCP
> (Kubernetes/Run:ai) using the Docker image
> `registry.rcp.epfl.ch/lmm-lebas/remelting_depth:<commit-hash>`. The image
> definition is not publicly hosted, but the sweep scripts here run
> equivalently on a local multicore machine at reduced throughput — the
> workload is memory-bandwidth-limited, not cluster-specific.

Post-process results and generate fit/validation metrics:

```bash
python analysis_pipeline.py
```

## Data

`progress__Pure_Full_Results_Original.csv` (~30k rows) is the canonical
pure-metal results dataset referenced in the manuscript. Columns are the raw
dimensionless groups, FD outputs (`D_rem`, `t_at_max_remelt`, etc.), and
provenance flags (`refined`, `status`, `config_fingerprint`).

The larger Scheil sweep dataset (~35k rows) is not included directly in this
repo; see the paper's data availability statement / Zenodo link (TBD) for the
archived version with a DOI.

## Grid convergence

Default spatial step `dz = 0.0025` carries ~0.53% bias relative to
`dz = 0.00125`. A coarser `dz = 0.005` is ~8× cheaper and within tolerance if
compute time is constrained; see the verification plan for details.

## Citation

If you use this code, please cite the associated paper(s) (citation details
to be added on publication). A `CITATION.cff` will be added once a DOI is
assigned.

## License

MIT (or specify — see `LICENSE`).

