#!/usr/bin/env python3
"""
table3_rows_BC_eigenspectrum.py
===============================

Computes the 40 lowest eigenmodes of Delta_1 on Rows B and C, using the
same pipeline as make_fig1_three_tier_rowA.py, and reports the measured
IPR*V ranges and bulk-gap values for each row. This script does not
generate a figure; it produces the numerical values needed for Table 3
of CC_2.tex.

Row definitions (from Section 3 of CC_2.tex):
    Row A: L_spatial = 18, L_temporal = 12, rho = 3.0  (for comparison)
    Row B: L_spatial = 24, L_temporal = 16, rho = 3.5
    Row C: L_spatial = 32, L_temporal = 24, rho = 4.5

Classification (row-agnostic, structural):
    Modes sorted ascending by lambda.
    Tier 1 (torus): lowest 12 modes
    Tier 2 (core):  next 8 modes
    Tier 3 (bulk):  remainder

Usage:
    python table3_rows_BC_eigenspectrum.py          # full compute
    python table3_rows_BC_eigenspectrum.py --cache  # re-report from cache
"""

import argparse
import os
import sys
import time
import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))
from src.caloron import init_caloron
from src.spectral import build_vector_operator

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
MASS_SQ = 0.01
N_EIGEN = 40
N_TORUS = 12   # structural: lowest 12 modes are torus
N_CORE  = 8    # next 8 are core; remainder bulk

ROWS = [
    {"label": "B", "L_spatial": 24, "L_temporal": 16, "rho": 3.5},
    {"label": "C", "L_spatial": 32, "L_temporal": 24, "rho": 4.5},
]


def compute_one_row(row):
    t0 = time.time()
    L_s, L_t, rho = row["L_spatial"], row["L_temporal"], row["rho"]
    label = row["label"]
    V = L_t * L_s**3

    print(f"\n{'='*70}")
    print(f" Row {label}: L_spatial={L_s}, L_temporal={L_t}, rho={rho}  "
          f"(V = {V}, N = {12*V})")
    print(f"{'='*70}")

    print(f"[Stage 1] HS caloron links ({L_s}^3 x {L_t}, rho={rho}) ...")
    t1 = time.time()
    U = init_caloron(L_spatial=L_s, L_temporal=L_t, rho=rho)
    print(f"           done ({time.time()-t1:.1f} s). U shape: {U.shape}")

    print(f"[Stage 2] Delta_1 = -D^2 + 2F + m^2 (m^2 = {MASS_SQ}) ...")
    t2 = time.time()
    Delta_1 = build_vector_operator(U, mass_sq=MASS_SQ)
    N = Delta_1.shape[0]
    print(f"           done ({time.time()-t2:.1f} s). dim = {N}, "
          f"nnz = {Delta_1.nnz}")

    rng = np.random.default_rng(42)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    y = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    herm_probe = abs(np.vdot(y, Delta_1 @ x) - np.vdot(Delta_1 @ y, x))
    print(f"           Hermiticity probe: {herm_probe:.2e}")

    print(f"[Stage 3] {N_EIGEN} lowest eigenpairs (CuPy shift-invert) ...")
    t3 = time.time()

    import cupy as cp
    import cupyx.scipy.sparse as cpsps
    import cupyx.scipy.sparse.linalg as cpsplinalg

    Delta_1_gpu = cpsps.csr_matrix(Delta_1)
    eigvals_gpu, eigvecs_gpu = cpsplinalg.eigsh(
        Delta_1_gpu, k=N_EIGEN, sigma=-0.001, which='LM',
        maxiter=5000, tol=1e-10)

    eigvals = cp.asnumpy(eigvals_gpu)
    eigvecs = cp.asnumpy(eigvecs_gpu)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    del Delta_1_gpu, eigvals_gpu, eigvecs_gpu
    cp.get_default_memory_pool().free_all_blocks()

    print(f"           done ({time.time()-t3:.1f} s)")

    abs2 = np.abs(eigvecs)**2
    norm2 = abs2.sum(axis=0)
    assert np.allclose(norm2, 1.0, atol=1e-6)
    ipr = (abs2**2).sum(axis=0)
    iprV = ipr * V

    cache_path = os.path.join(RESULTS_DIR, f"fig1_row{label}_eigenspectrum.npz")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path, eigvals=eigvals, iprV=iprV,
        N=N, V=V, L_spatial=L_s, L_temporal=L_t, rho=rho,
        mass_sq=MASS_SQ, n_eigen=N_EIGEN)
    print(f"[cache]    {cache_path}")
    print(f"[total]    {time.time()-t0:.1f} s")

    return eigvals, iprV, V


def summarize(eigvals, iprV, V, label):
    """Structural classification: lowest N_TORUS = torus, next N_CORE = core,
    remainder = bulk."""
    n = len(eigvals)
    tier = np.full(n, 3, dtype=int)
    tier[:N_TORUS] = 1
    tier[N_TORUS:N_TORUS + N_CORE] = 2
    # rest stays 3

    t1 = (tier == 1)
    t2 = (tier == 2)
    t3 = (tier == 3)

    torus_ipr = (iprV[t1].min(), iprV[t1].max())
    core_ipr  = (iprV[t2].min(), iprV[t2].max())
    bulk_ipr  = (iprV[t3].min(), iprV[t3].max())
    bulk_gap  = eigvals[t3].min() - eigvals[t2].max()

    # Core subsplit: find the break in IPR*V within core
    core_iprs_sorted = np.sort(iprV[t2])
    # The 4+4 split: look for largest gap in sorted IPR*V
    gaps = np.diff(core_iprs_sorted)
    break_idx = np.argmax(gaps) + 1
    core_low = core_iprs_sorted[:break_idx]
    core_high = core_iprs_sorted[break_idx:]

    print(f"\n  Row {label} summary:")
    print(f"    Tier counts (torus, core, bulk) = ({t1.sum()}, {t2.sum()}, {t3.sum()})")
    print(f"    Torus IPR*V range: [{torus_ipr[0]:.2f}, {torus_ipr[1]:.2f}]")
    print(f"    Core  IPR*V range: [{core_ipr[0]:.2f}, {core_ipr[1]:.2f}]")
    print(f"    Core  4+4 subsplit:  low  [{core_low.min():.2f}, {core_low.max():.2f}]  ({len(core_low)} modes)")
    print(f"                        high [{core_high.min():.2f}, {core_high.max():.2f}]  ({len(core_high)} modes)")
    print(f"    Bulk  IPR*V range: [{bulk_ipr[0]:.2f}, {bulk_ipr[1]:.2f}]")
    print(f"    Bulk gap (lambda):  {bulk_gap:.4f}")
    print(f"    Eigenvalue bands:")
    print(f"      Tier 1: [{eigvals[t1].min():.4f}, {eigvals[t1].max():.4f}]")
    print(f"      Tier 2: [{eigvals[t2].min():.4f}, {eigvals[t2].max():.4f}]")
    print(f"      Tier 3: [{eigvals[t3].min():.4f}, {eigvals[t3].max():.4f}]")

    return {
        "label": label, "V": V,
        "counts": (int(t1.sum()), int(t2.sum()), int(t3.sum())),
        "torus_iprV": torus_ipr, "core_iprV": core_ipr,
        "core_low_iprV": (float(core_low.min()), float(core_low.max())),
        "core_high_iprV": (float(core_high.min()), float(core_high.max())),
        "bulk_iprV": bulk_ipr,
        "bulk_gap": float(bulk_gap),
    }


def load_cached(label):
    path = os.path.join(RESULTS_DIR, f"fig1_row{label}_eigenspectrum.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["eigvals"], data["iprV"], int(data["V"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    summaries = []

    rowA_data = load_cached("A")
    if rowA_data is not None:
        print("\n[Row A — from existing cache]")
        summaries.append(summarize(*rowA_data, label="A"))

    for row in ROWS:
        label = row["label"]
        if args.cache:
            cached = load_cached(label)
            if cached is None:
                print(f"ERROR: no cache for Row {label}; run without --cache")
                sys.exit(1)
            eigvals, iprV, V = cached
        else:
            eigvals, iprV, V = compute_one_row(row)
        summaries.append(summarize(eigvals, iprV, V, label))

    # Final Table 3 format
    print(f"\n{'='*80}")
    print(" Table 3 of CC_2.tex — measured values")
    print(f"{'='*80}")
    print(f"  {'Row':<4} {'torus IPR*V':>12} {'core low':>12} {'core high':>12} "
          f"{'bulk gap':>10}")
    for s in summaries:
        t = s["torus_iprV"]
        cl = s["core_low_iprV"]
        ch = s["core_high_iprV"]
        print(f"  {s['label']:<4} "
              f"{t[0]:.2f}-{t[1]:.2f}  "
              f"{cl[0]:.2f}-{cl[1]:.2f}  "
              f"{ch[0]:.2f}-{ch[1]:.2f}  "
              f"{s['bulk_gap']:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
