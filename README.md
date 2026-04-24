# cc-lattice-residual

Reproducibility archive for the numerical residual-factor results of

> M. M. Hanash, *The cosmological-constant branch response on the
> Harrington–Shepard caloron background: a one-loop lattice
> computation* (2026).

This archive contains the LaTeX source of the paper, the production
pipeline modules, all scripts needed to regenerate the published
numerical results and figures, and the cached eigenspectrum and
overlap measurements.

## Layout
cc-lattice-residual/
├── paper/               LaTeX source, bibliography, figures, PDF
│   ├── CC_2.tex
│   ├── bib.bib
│   ├── CC_2.pdf
│   └── figures/
│       ├── fig_three_tier_rowA.pdf          (Fig. 1)
│       ├── fig_transport_overlaps_rowA.pdf  (Fig. 2, two-panel)
│       └── fig_icl_correlation.pdf          (Fig. 3)
├── src/                 Production pipeline modules
│   ├── init.py
│   ├── config.py             Locked normalisation conventions
│   ├── caloron.py            HS caloron seed construction
│   ├── lattice.py            SU(2) primitives, link/plaquette routines
│   ├── spectral.py           Δ₁, K₀, adjoint Laplacian, ghost operator
│   ├── observables.py        I_cl, δ₊, topological charge, plaquettes
│   ├── topology.py           Cooling-invariant topological diagnostics
│   ├── cooling.py            Gradient-flow / stout cooling
│   ├── heatbath.py           SU(2) heatbath sweep primitives
│   ├── instanton.py          Single-instanton utilities (R⁴ reference)
│   └── minimiser.py          Gauge-orbit minimisation for LT basis
├── scripts/             Production scripts and figure generators
│   │ --- Figure generators ---
│   ├── make_fig1_three_tier_rowA.py          Figure 1 (Row A scatter)
│   ├── make_fig2_transport_overlaps_rowA.py  Figure 2 compute (3 eigensolves)
│   ├── make_fig2_subspace_diagnostic.py      Figure 2 S(ξ) + replot
│   ├── make_fig3_icl_correlation.py          Figure 3 (I_cl correlation)
│   │ --- Table 3 data ---
│   ├── table3_rows_BC_eigenspectrum.py       Rows B, C eigenspectra
│   │ --- Production pipeline ---
│   ├── jacobian_exact.py                     J_phys = 0.118164 (Eq. 14)
│   ├── jacobian_left_triv.py                 LT-5 cross-check
│   ├── mode_diagnostic.py                    Eigenmode classification
│   ├── exact_prime_F_G.py                    δF anisotropic insertion
│   ├── exact_prime_projected.py              Primed-trace construction
│   ├── ipr_gated_prime.py                    Adaptive-prime diagnostics
│   ├── cross_check_row_A_N800.py             N=800 Row A validation
│   ├── gate2_runpod_h100_v2.py               Production Simpson (H100)
│   │ --- Infrastructure tests ---
│   └── verify_jacobian_infra.py              Unit tests for jacobian_exact
├── results/             Cached measurements and production logs
│   ├── fig1_rowA_eigenspectrum.npz
│   ├── fig1_rowB_eigenspectrum.npz
│   ├── fig1_rowC_eigenspectrum.npz
│   ├── fig2_rowA_transport.npz
│   ├── cross_check_N800.log
│   ├── cross_check_row_A.log
│   ├── gate2_h100_output.txt
│   └── gate2_runpod_h100_v2_output.txt
├── README.md            This file
├── reproducibility.md   Paper equation/table → script/data map
├── requirements.txt     Python dependencies
├── LICENSE              MIT licence (code); CC-BY-4.0 (paper, figures)
├── CITATION.cff         Machine-readable citation metadata
└── .gitignore

## Quick start

```bash
# Reproduce Figure 3 (takes 3 seconds on CPU)
python scripts/make_fig3_icl_correlation.py

# Reproduce Figure 1 (needs RTX 5060 Ti or equivalent, ~90 s on GPU)
python scripts/make_fig1_three_tier_rowA.py

# Reproduce Figure 2 (three eigensolves + subspace diagnostic, ~2 min on GPU)
python scripts/make_fig2_transport_overlaps_rowA.py
python scripts/make_fig2_subspace_diagnostic.py

# Reproduce Table 3 Rows B and C eigenspectra (~20 min on GPU)
python scripts/table3_rows_BC_eigenspectrum.py

# Verify the Jacobian value J_phys = 0.118164
python scripts/jacobian_exact.py

# Run the Jacobian infrastructure unit tests (no GPU needed)
python scripts/verify_jacobian_infra.py
```

All scripts are self-contained, print their own arithmetic chains to
stdout, and write outputs to `paper/figures/` or `results/`. The
`--cache` flag on figure scripts replots from cached npz files
without repeating the GPU eigensolve.

## Hardware requirements

Production results were obtained on:

- AMD Ryzen 9 3900X, 64 GB RAM
- NVIDIA RTX 5060 Ti, 16 GB VRAM (primary GPU)
- NVIDIA H100 80 GB (Simpson campaign; see `scripts/gate2_runpod_h100_v2.py`)

Eigensolves use CuPy 14.0 on CUDA 12.09 via
`cupyx.scipy.sparse.linalg.eigsh(..., which='SA')`. The same code
paths fall back to `scipy.sparse.linalg.eigsh` on CPU if CuPy is
unavailable; CPU runs are substantially slower (~10× for Row A).

## Conventions

Fixed throughout the paper and codebase (see `src/config.py`):

- Link index layout: `U[x][μ][a,b]`; sparse operator index
  `v[12*x + 3*μ + a]`
- Mass regulator `m² = 0.01`, Simpson step `ε = 0.09`
- Paired noise seed 42 throughout
- Jacobian J_phys = +0.118164, phase cos Θ_γ = +1

## Licence

Code released under the MIT licence (see `LICENSE`). Paper source
and figures (`paper/CC_2.tex`, `paper/bib.bib`, `paper/figures/*.pdf`)
are released under CC-BY-4.0.

## Citation

Please cite both the paper and this archive. Machine-readable
citation metadata is in `CITATION.cff`.
