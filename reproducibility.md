# Reproducibility map

Maps each published numerical result in `paper/CC_2.tex` to the
script that produces it and the raw data cached in `results/`.

## Figures

| Paper figure | Produced by | Cache file |
|---|---|---|
| Fig. 1 (Row A three-tier scatter) | `scripts/make_fig1_three_tier_rowA.py` | `results/fig1_rowA_eigenspectrum.npz` |
| Fig. 2 (transport diagnostics, panels a+b) | `scripts/make_fig2_transport_overlaps_rowA.py` → `scripts/make_fig2_subspace_diagnostic.py` | `results/fig2_rowA_transport.npz` |
| Fig. 3 (I_cl correlation, App. D) | `scripts/make_fig3_icl_correlation.py` | — (data in script) |

## Tables

| Paper table | Produced by | Cache file(s) |
|---|---|---|
| Table 3 (three-tier localisation, all rows) | `scripts/table3_rows_BC_eigenspectrum.py` (Rows B, C) + `scripts/make_fig1_three_tier_rowA.py` (Row A) | `results/fig1_row{A,B,C}_eigenspectrum.npz` |
| Table 11 (I_cl-degraded rows, App. D) | Data inlined in `scripts/make_fig3_icl_correlation.py` | — |

## Key numerical values

| Paper value | Source script | Cache / log |
|---|---|---|
| Eq. 14: J_phys = 0.118164 | `scripts/jacobian_exact.py` (and `jacobian_left_triv.py` for LT-5 cross-check) | — (computed live, ~30 s) |
| Eq. 18: δ_ξ log ζ = −1.669 ± 0.125, bracket 0.813 ± 0.026 | `scripts/gate2_runpod_h100_v2.py` (H100 production) | `results/gate2_runpod_h100_v2_output.txt` |
| Eq. 38 + App. D: weighted LSQ fit and extrapolation | `scripts/make_fig3_icl_correlation.py` | — |
| Table 3 values | `scripts/table3_rows_BC_eigenspectrum.py` | `results/fig1_row{A,B,C}_eigenspectrum.npz` |
| Subspace overlaps S(1.045) = 0.99, S(1.090) = 0.96 | `scripts/make_fig2_subspace_diagnostic.py` | `results/fig2_rowA_transport.npz` |

## Verification workflow for a referee

Quick path (no GPU required, < 5 minutes):

1. Create a fresh Python 3.11 environment, install `requirements.txt`.
2. Run `python scripts/make_fig3_icl_correlation.py`. Inspect stdout:
   the weighted LSQ `a = -18.7, b = -49.5`, χ²/dof = 3.3, inflation
   factor ≈ 1.82, and 0.03σ tension with the primary result should
   appear. The regenerated PDF in `paper/figures/` should match the
   archived copy.
3. Run `python scripts/verify_jacobian_infra.py` for Jacobian
   infrastructure unit tests. All five tests should pass.

Full path (GPU recommended, ~25 minutes):

4. Run `python scripts/make_fig1_three_tier_rowA.py`. Stdout shows
   tier counts (12, 8, 20) and bulk gap 0.0836.
5. Run `python scripts/make_fig2_transport_overlaps_rowA.py` then
   `python scripts/make_fig2_subspace_diagnostic.py`. Stdout shows
   S(1.045) = 0.9919 and S(1.090) = 0.9644.
6. Run `python scripts/table3_rows_BC_eigenspectrum.py --cache`.
   Rows A, B, C tier counts all (12, 8, 20), bulk gaps 0.084, 0.050,
   0.031 respectively.
7. Run `python scripts/jacobian_exact.py`. J_phys = 0.118164.

All arithmetic is self-verifying at stdout level; no external tools
are required beyond the Python dependencies listed in
`requirements.txt`.

## Dependencies not in this archive

The production Simpson campaign (`scripts/gate2_runpod_h100_v2.py`)
was executed on an NVIDIA H100 via RunPod and is included for
documentation. Its cached output log
(`results/gate2_runpod_h100_v2_output.txt`) contains the
three-row Simpson measurements summarised in Eq. 18 of the paper.
Re-running this script requires an H100-class GPU.
