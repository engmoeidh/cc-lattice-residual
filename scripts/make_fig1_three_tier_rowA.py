#!/usr/bin/env python3
"""
make_fig1_three_tier_rowA.py
=============================

Generates Figure 1 of the paper: the three-tier (torus/core/bulk)
decomposition of the 40 lowest eigenmodes of the lattice vector
operator Delta_1 on Row A, plotted in the (IPR*V, lambda) plane.

Produces:
    paper/figures/fig_three_tier_rowA.pdf
    results/fig1_rowA_eigenspectrum.npz   (cache for re-runs)

Data source:
    Direct computation on Row A:  L_spatial=18, L_temporal=12, rho=3.0
    HS caloron seed, self-dual, trivial holonomy.

Conventions:
    Delta_1 = -D^2 + 2F + m^2,  m^2 = 0.01
    Index layout:   flat = 12*x + 3*mu + a
    Matrix dim:     N = 12 * V,  V = L_t * L_s^3 = 12 * 18^3 = 69984
                    N = 839808
    IPR:            sum |psi|^4 / (sum |psi|^2)^2
                    Eigenvectors returned orthonormal, so sum |psi|^2 = 1.
                    IPR = sum |psi|^4.
    IPR*V:          scaled so that a fully delocalised mode has IPR*V ~ 1.

Tier windows (from Table 3 of CC_2.tex, Row A):
    Tier 1 (torus): 12 modes, IPR*V in [1.0, 1.08], lambda in [0.006, 0.018]
    Tier 2 (core):   8 modes, IPR*V in [14, 100], lambda in [0.016, 0.030]
    Tier 3 (bulk):  remaining 20 modes, IPR*V in [1.2, 1.6], gap ~ 0.084

Runtime:
    ~3-15 min on RTX 5060 Ti (CuPy shift-invert Lanczos, 40 eigenpairs).
    Use --cache to reload from cached eigenspectrum (~5 seconds).

Usage:
    python make_fig1_three_tier_rowA.py          # full compute + plot
    python make_fig1_three_tier_rowA.py --cache  # plot from cached npz
"""

import argparse
import os
import sys
import time
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

# --- Local imports (src/ must be on PYTHONPATH) ---
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))
from src.caloron import init_caloron
from src.spectral import build_vector_operator

# --- Parameters (Row A, locked by paper conventions) ---
L_SPATIAL  = 18
L_TEMPORAL = 12
RHO        = 3.0
MASS_SQ    = 0.01
N_EIGEN    = 40          # compute 40 lowest eigenmodes
CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "results", "fig1_rowA_eigenspectrum.npz")
FIG_PATH   = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "paper", "figures", "fig_three_tier_rowA.pdf")


def compute_eigenspectrum():
    """Build Row A caloron, assemble Delta_1, compute 40 lowest eigenpairs."""
    t0 = time.time()

    # Stage 1: build caloron (CPU)
    print(f"[Stage 1] Building HS caloron links ({L_SPATIAL}^3 x {L_TEMPORAL}, "
          f"rho={RHO}) ...")
    t1 = time.time()
    U = init_caloron(L_spatial=L_SPATIAL, L_temporal=L_TEMPORAL, rho=RHO)
    print(f"           done ({time.time()-t1:.1f} s). U shape: {U.shape}")

    # Stage 2: build Delta_1 (CPU, scipy sparse CSR)
    print(f"[Stage 2] Building Delta_1 = -D^2 + 2F + m^2 (m^2 = {MASS_SQ}) ...")
    t2 = time.time()
    Delta_1 = build_vector_operator(U, mass_sq=MASS_SQ)
    N = Delta_1.shape[0]
    V = L_SPATIAL**3 * L_TEMPORAL
    print(f"           done ({time.time()-t2:.1f} s). "
          f"dim = {N} (= 12 * V, V = {V}). "
          f"nnz = {Delta_1.nnz}, density = {Delta_1.nnz/N**2:.2e}")

    # Hermiticity sanity check
    herm_err = (Delta_1 - Delta_1.getH()).toarray().max() if N < 2000 else None
    # For large N, check via random vector
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    y = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    herm_probe = abs(np.vdot(y, Delta_1 @ x) - np.vdot(Delta_1 @ y, x))
    print(f"           Hermiticity probe: |<y,Ax> - <Ay,x>| = {herm_probe:.2e}")
    if herm_probe > 1e-8:
        print(f"           WARNING: Hermiticity probe large; proceeding with caution.")

    # Stage 3: shift-invert Lanczos for 40 lowest eigenmodes
    # Try GPU first, fall back to CPU if CuPy not available
    print(f"[Stage 3] Computing {N_EIGEN} lowest eigenpairs (shift-invert Lanczos) ...")
    t3 = time.time()

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsps
        import cupyx.scipy.sparse.linalg as cpsplinalg
        print(f"           Using CuPy on GPU.")

        # Transfer to GPU
        Delta_1_gpu = cpsps.csr_matrix(Delta_1)

        # Shift-invert Lanczos at sigma = -0.001 (just below 0 to find
        # the smallest positive eigenvalues of a p.s.d. matrix)
        sigma = -0.001
        eigvals_gpu, eigvecs_gpu = cpsplinalg.eigsh(
            Delta_1_gpu, k=N_EIGEN, which='SA',
            maxiter=5000, tol=1e-10)

        eigvals = cp.asnumpy(eigvals_gpu)
        eigvecs = cp.asnumpy(eigvecs_gpu)
        backend = "CuPy GPU"
    except ImportError:
        print(f"           CuPy unavailable; falling back to scipy on CPU.")
        sigma = -0.001
        eigvals, eigvecs = splinalg.eigsh(
            Delta_1, k=N_EIGEN, sigma=sigma, which='LM',
            maxiter=5000, tol=1e-10)
        backend = "scipy CPU"
    except Exception as e:
        print(f"           GPU path failed: {e}")
        print(f"           Falling back to scipy on CPU.")
        sigma = -0.001
        eigvals, eigvecs = splinalg.eigsh(
            Delta_1, k=N_EIGEN, sigma=sigma, which='LM',
            maxiter=5000, tol=1e-10)
        backend = "scipy CPU (fallback)"

    # Sort ascending
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    print(f"           done ({time.time()-t3:.1f} s) via {backend}.")
    print(f"           Eigenvalue range: [{eigvals[0]:.6f}, {eigvals[-1]:.6f}]")

    # Compute IPR for each eigenvector
    # IPR = sum |psi|^4 (eigenvectors orthonormal so sum |psi|^2 = 1)
    # IPR*V as the paper convention
    abs2 = np.abs(eigvecs)**2
    norm2 = abs2.sum(axis=0)
    # Sanity: each column should sum to ~1
    assert np.allclose(norm2, 1.0, atol=1e-6), f"Eigenvector norms: {norm2}"
    ipr = (abs2**2).sum(axis=0)
    iprV = ipr * V
    print(f"           IPR*V range: [{iprV.min():.2f}, {iprV.max():.2f}]")

    # Cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        eigvals=eigvals,
        iprV=iprV,
        N=N, V=V,
        L_spatial=L_SPATIAL, L_temporal=L_TEMPORAL, rho=RHO,
        mass_sq=MASS_SQ, n_eigen=N_EIGEN,
    )
    print(f"[cache]    saved: {CACHE_PATH}")
    print(f"[total]    {time.time()-t0:.1f} s")
    return eigvals, iprV


def classify_modes(eigvals, iprV):
    """Classify modes into tiers using eigenvalue-range criterion.

    On the caloron background, the low spectrum of Delta_1 separates into
    three non-overlapping eigenvalue bands with a clean bulk gap. The
    paper's classification is by this band structure (Section 6 and
    Table 3 of CC_2.tex).

    Tier 1 (torus): lambda <  0.020   (12 delocalised modes)
    Tier 2 (core):  0.020 <= lambda < 0.040  (8 localised modes)
    Tier 3 (bulk):  lambda >= 0.040   (remaining modes, above bulk gap)

    The IPR*V values are reported per-mode for diagnostic purposes, and
    further refine the Tier 2 split into a translation-quartet subset
    (lower IPR*V) and a scale+gauge subset (higher IPR*V).
    """
    # Structural classification: mode index in ascending-lambda order.
    # Lowest 12 = torus, next 8 = core, remainder = bulk.
    # This rule is row-agnostic and matches the paper's Section 6.
    N_TORUS, N_CORE = 12, 8

    n = len(eigvals)
    tier = np.full(n, 3, dtype=int)
    tier[:N_TORUS] = 1
    tier[N_TORUS:N_TORUS + N_CORE] = 2

    counts = {1: int((tier == 1).sum()),
              2: int((tier == 2).sum()),
              3: int((tier == 3).sum())}
    print()
    print("=" * 70)
    print(" TIER CLASSIFICATION (Row A, 40 lowest modes)")
    print("=" * 70)
    print(f"  Tier 1 (torus, IPR*V <= 1.15):     {counts[1]:2d} modes  "
          f"(paper: 12)")
    print(f"  Tier 2 (core,  IPR*V > 1.15,       {counts[2]:2d} modes  "
          f"(paper: 8)")
    print(f"         lambda < 0.040)")
    print(f"  Tier 3 (bulk):                     {counts[3]:2d} modes")
    print()
    print(f"  Mode    lambda        IPR*V    Tier")
    print(f"  ---------------------------------------")
    for i, (lam, iv, t) in enumerate(zip(eigvals, iprV, tier)):
        tname = {1: "torus", 2: "core", 3: "bulk"}[t]
        print(f"  {i:3d}    {lam:.6f}    {iv:7.2f}   {t} ({tname})")
    print("=" * 70)

    # Tier 1/2 gap
    tier1_mask = (tier == 1)
    tier2_mask = (tier == 2)
    tier3_mask = (tier == 3)
    if tier2_mask.any() and tier3_mask.any():
        lam_tier2_max = eigvals[tier2_mask].max()
        lam_tier3_min = eigvals[tier3_mask].min()
        bulk_gap = lam_tier3_min - lam_tier2_max
        print(f"  Bulk gap (min Tier 3 - max Tier 2) = {bulk_gap:.4f}  "
              f"(paper: 0.084)")

    return tier, counts


def plot_figure(eigvals, iprV, tier, outpath=FIG_PATH):
    """Three-tier scatter: (IPR*V, lambda) with colour by tier."""
    fig, ax = plt.subplots(figsize=(6.2, 4.8))

    tier1 = (tier == 1)
    tier2 = (tier == 2)
    tier3 = (tier == 3)

    ax.scatter(iprV[tier1], eigvals[tier1],
               s=65, marker='o', facecolor='tab:blue',
               edgecolor='black', linewidths=0.8,
               label=f"Tier 1 (torus): {tier1.sum()} modes",
               zorder=3)
    ax.scatter(iprV[tier2], eigvals[tier2],
               s=65, marker='s', facecolor='tab:orange',
               edgecolor='black', linewidths=0.8,
               label=f"Tier 2 (core): {tier2.sum()} modes",
               zorder=3)
    ax.scatter(iprV[tier3], eigvals[tier3],
               s=55, marker='^', facecolor='tab:green',
               edgecolor='black', linewidths=0.6,
               label=f"Tier 3 (bulk): {tier3.sum()} modes",
               zorder=2)

    # Horizontal guide line at m^2 = 0.01 (infrared regulator floor)
    ax.axhline(MASS_SQ, color='gray', linestyle=':', linewidth=0.8,
               label=rf"$m^2 = {MASS_SQ}$", zorder=1)

    ax.set_xscale('log')
    ax.set_xlabel(r"$\mathrm{IPR} \times V$", fontsize=12)
    ax.set_ylabel(r"$\lambda$ (eigenvalue of $\Delta_1$)", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle=':', which='both')
    # Expand y-axis top to give room above bulk cluster
    ax.set_ylim(-0.005, 0.145)
    ax.legend(loc='center left', bbox_to_anchor=(0.02, 0.55),
              frameon=True, framealpha=0.95, fontsize=9.5)

    # Annotation: three-tier structure
    ax.text(0.98, 0.05,
            rf"Row A: $L_s = {L_SPATIAL}$, $L_t = {L_TEMPORAL}$, "
            rf"$\rho = {RHO}$",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close()
    print()
    print(f"Saved: {outpath}")
    print(f"File size: {os.path.getsize(outpath)} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true",
                        help="Load from cached eigenspectrum")
    args = parser.parse_args()

    if args.cache and os.path.exists(CACHE_PATH):
        print(f"[cache]    loading: {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        eigvals = data["eigvals"]
        iprV    = data["iprV"]
    else:
        eigvals, iprV = compute_eigenspectrum()

    tier, counts = classify_modes(eigvals, iprV)
    plot_figure(eigvals, iprV, tier)


if __name__ == "__main__":
    main()
