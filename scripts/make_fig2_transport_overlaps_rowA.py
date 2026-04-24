#!/usr/bin/env python3
"""
make_fig2_transport_overlaps_rowA.py
====================================

Generates Figure 2 of the paper: mode-wise transport overlaps on Row A
at the two non-anchor Simpson nodes xi_1 = 1.045 and xi_2 = 1.090.

Produces:
    paper/figures/fig_transport_overlaps_rowA.pdf
    results/fig2_rowA_transport.npz   (cache for re-runs)

Data source:
    Row A direct computation: L_spatial=18, L_temporal=12, rho=3.0.
    Same caloron background as Figure 1; eigensolve of
    Delta_1(xi) at xi = 1.000, 1.045, 1.090 with 25 modes each,
    followed by greedy overlap matching of the anchor 20 modes.

Anisotropic operator (Approach X):
    Delta_1(xi) = Delta_1(1) + (xi^2 - 1) * K_0 + 2 * (xi - 1) * delta_F

where K_0 is the temporal adjoint Laplacian (built via
build_adjoint_laplacian with temporal_only=True) and delta_F is the
anisotropic field-strength insertion (built via build_delta_F_operator
from the exact_prime_F_G production module).

Algorithm B.1 of the paper:
    1. Compute 20 anchor modes {psi_alpha(1)} at xi_0 = 1.
    2. At each xi_j, compute N_c > 20 lowest eigenpairs of Delta_1(xi_j).
    3. Form overlap matrix O_{alpha, k} = |<psi_alpha(1) | phi_k(xi_j)>|.
    4. Greedy assignment: iteratively pick max O_{alpha, k} across
       unassigned alpha, k pairs, assigning alpha -> k.
    5. Transported basis = {phi_{k*(alpha)}(xi_j)}, mode-wise overlap
       is Omega_alpha(xi_j) = |<psi_alpha(1) | phi_{k*(alpha)}(xi_j)>|.

Runtime:
    3 eigensolves at ~10 min each on RTX 5060 Ti = ~30 min total.
    Use --cache to replot from cached overlaps (~5 seconds).

Usage:
    python make_fig2_transport_overlaps_rowA.py          # full compute
    python make_fig2_transport_overlaps_rowA.py --cache  # cached replot
"""

import argparse
import os
import sys
import time
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))
from src.caloron import init_caloron
from src.spectral import build_vector_operator, build_adjoint_laplacian

# build_delta_F_operator lives in scripts/exact_prime_F_G.py — import it
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from exact_prime_F_G import build_delta_F_operator

# --- Row A parameters (paper conventions, locked) ---
L_SPATIAL  = 18
L_TEMPORAL = 12
RHO        = 3.0
MASS_SQ    = 0.01
XI_NODES   = [1.000, 1.045, 1.090]   # Simpson grid
N_ANCHOR   = 20                      # mode count at xi = 1 to transport
N_CANDIDATE = 25                     # wider Lanczos window at each xi

CACHE_PATH = os.path.join(REPO_ROOT, "results",
                          "fig2_rowA_transport.npz")
FIG_PATH   = os.path.join(REPO_ROOT, "paper", "figures",
                          "fig_transport_overlaps_rowA.pdf")


def build_ops_row_A():
    """Build Delta_1(1), K_0, delta_F for Row A. Returns scipy CSRs."""
    print(f"[ops]  Building Row A caloron ({L_SPATIAL}^3 x {L_TEMPORAL}, "
          f"rho={RHO})...")
    t0 = time.time()
    U = init_caloron(L_spatial=L_SPATIAL, L_temporal=L_TEMPORAL, rho=RHO)
    print(f"       done ({time.time()-t0:.1f} s). U shape: {U.shape}")

    print(f"[ops]  Delta_1(xi=1) = -D^2 + 2F + m^2 (m^2 = {MASS_SQ})...")
    t1 = time.time()
    Delta_1 = build_vector_operator(U, mass_sq=MASS_SQ)
    N = Delta_1.shape[0]
    print(f"       done ({time.time()-t1:.1f} s). dim = {N}, "
          f"nnz = {Delta_1.nnz}")

    print(f"[ops]  Temporal Laplacian K_0 = -D_0^2 (no mass) ...")
    t2 = time.time()
    # Note: K_0 acts on adjoint scalars (dim = 3V), but Delta_1 acts on
    # adjoint vectors (dim = 12V). We need K_0 embedded in 12V-space,
    # diagonal in the Lorentz index mu.
    # Strategy: build scalar K_0 once, then tile it block-diagonally.
    K_0_scalar = build_adjoint_laplacian(U, mass_sq=0.0, temporal_only=True)
    print(f"       K_0_scalar dim = {K_0_scalar.shape[0]} = 3V")

    # Embed in 12V-space: Delta_1 index is [12*x + 3*mu + a],
    # K_0_scalar index is [3*x + a]. So for each mu in 0..3,
    # we place a copy of K_0_scalar at offset 3*mu within each 12-block.
    V = L_SPATIAL**3 * L_TEMPORAL
    assert K_0_scalar.shape[0] == 3 * V, \
        f"K_0_scalar dim {K_0_scalar.shape[0]} != 3V={3*V}"

    # Build the permutation: map adjoint-scalar index (3*x + a)
    # to all four Lorentz slots (12*x + 3*mu + a), mu = 0..3.
    # Easier: construct K_0_12V by tiling K_0_scalar four times with
    # appropriate index remapping.
    K_0_12V = _embed_scalar_in_vector(K_0_scalar, V)
    print(f"       K_0_12V dim = {K_0_12V.shape[0]} = 12V, "
          f"nnz = {K_0_12V.nnz} ({time.time()-t2:.1f} s)")

    print(f"[ops]  delta_F (anisotropic field-strength insertion) ...")
    t3 = time.time()
    dims = (L_TEMPORAL, L_SPATIAL, L_SPATIAL, L_SPATIAL)
    delta_F = build_delta_F_operator(U, dims)
    print(f"       delta_F dim = {delta_F.shape[0]}, "
          f"nnz = {delta_F.nnz} ({time.time()-t3:.1f} s)")

    return Delta_1, K_0_12V, delta_F, N


def _embed_scalar_in_vector(K_scalar, V):
    """Embed a 3V x 3V sparse matrix block-diagonally into 12V x 12V
    with identity in the Lorentz index. The result satisfies
        K_12V[12*x + 3*mu + a, 12*y + 3*mu + b] = K_scalar[3*x+a, 3*y+b]
    and is zero for mu != mu'.
    """
    # Extract COO form
    K_coo = K_scalar.tocoo()
    rows_s, cols_s, data_s = K_coo.row, K_coo.col, K_coo.data
    nnz_s = len(data_s)

    # For each non-zero entry (i, j, v) in scalar:
    #   i = 3*x + a, j = 3*y + b
    # We want entries (12*x + 3*mu + a, 12*y + 3*mu + b, v) for mu = 0..3.
    # So decompose i, j into (x, a) and (y, b):
    x_s, a_s = divmod(rows_s, 3)
    y_s, b_s = divmod(cols_s, 3)

    # Tile over mu = 0..3
    rows_v = np.concatenate([12 * x_s + 3 * mu + a_s for mu in range(4)])
    cols_v = np.concatenate([12 * y_s + 3 * mu + b_s for mu in range(4)])
    data_v = np.concatenate([data_s for _ in range(4)])

    K_12V = sps.coo_matrix((data_v, (rows_v, cols_v)),
                           shape=(12 * V, 12 * V)).tocsr()
    return K_12V


def compute_transport():
    """Main compute path: build ops, eigensolve at 3 xi values, overlap."""
    t_all = time.time()

    Delta_1, K_0_12V, delta_F, N = build_ops_row_A()

    # Quick Hermiticity probe
    rng = np.random.default_rng(42)
    xvec = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    yvec = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    h1 = abs(np.vdot(yvec, Delta_1 @ xvec) - np.vdot(Delta_1 @ yvec, xvec))
    h2 = abs(np.vdot(yvec, K_0_12V @ xvec) - np.vdot(K_0_12V @ yvec, xvec))
    h3 = abs(np.vdot(yvec, delta_F @ xvec) - np.vdot(delta_F @ yvec, xvec))
    print(f"[probe] Hermiticity: Delta_1 = {h1:.2e}, "
          f"K_0 = {h2:.2e}, delta_F = {h3:.2e}")

    # Eigensolve at each xi
    # Anchor: xi = 1.0, need N_ANCHOR = 20 modes
    # Each non-anchor xi: need N_CANDIDATE = 25 modes (overlap pool)
    import cupy as cp
    import cupyx.scipy.sparse as cpsps
    import cupyx.scipy.sparse.linalg as cpsplinalg

    eigenspectra = {}   # xi -> (eigvals[k], eigvecs[N, k])

    for xi in XI_NODES:
        print(f"\n[eigs] xi = {xi:.3f}:")
        t_eig = time.time()

        # Construct Delta_1(xi) = Delta_1(1) + (xi^2 - 1) * K_0 + 2*(xi-1)*delta_F
        D_xi = Delta_1 + (xi**2 - 1) * K_0_12V + 2 * (xi - 1) * delta_F
        D_xi = D_xi.tocsr()

        k_request = N_ANCHOR if abs(xi - 1.0) < 1e-9 else N_CANDIDATE
        print(f"       requesting {k_request} lowest eigenpairs ...")

        D_gpu = cpsps.csr_matrix(D_xi)
        # 'SA' = smallest algebraic: no shift-invert, matches production
        # pipeline (gate2_closure_v3.py). CuPy 14 does not support sigma=.
        eigvals_gpu, eigvecs_gpu = cpsplinalg.eigsh(
            D_gpu, k=k_request, which='SA',
            maxiter=5000, tol=1e-10)
        eigvals = cp.asnumpy(eigvals_gpu)
        eigvecs = cp.asnumpy(eigvecs_gpu)

        # Sort ascending
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        print(f"       eigenvalue range: "
              f"[{eigvals[0]:.6f}, {eigvals[-1]:.6f}] "
              f"({time.time()-t_eig:.1f} s)")

        eigenspectra[xi] = (eigvals, eigvecs)

        del D_gpu, eigvals_gpu, eigvecs_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # Anchor
    xi0 = XI_NODES[0]
    anchor_vals, anchor_vecs = eigenspectra[xi0]
    assert anchor_vecs.shape[1] >= N_ANCHOR

    # Overlap matching at each non-anchor xi
    min_overlaps = {}
    assigned_overlaps = {}
    mode_indices = {}
    overlap_matrices = {}   # full O_{alpha, k}, alpha in [0,20), k in [0, N_c)
    for xi in XI_NODES[1:]:
        cand_vals, cand_vecs = eigenspectra[xi]
        print(f"\n[over] xi = {xi:.3f}: overlap matrix "
              f"({N_ANCHOR} anchor x {cand_vecs.shape[1]} candidates)")

        # Overlap matrix: O_{alpha, k} = |<psi_alpha(1) | phi_k(xi)>|
        # Using column slice anchor_vecs[:, :N_ANCHOR] for the 20 anchors
        O = np.abs(
            anchor_vecs[:, :N_ANCHOR].conj().T @ cand_vecs
        )
        overlap_matrices[xi] = O.copy()  # save for subspace-overlap diagnostic

        # Greedy assignment
        assigned = np.full(N_ANCHOR, -1, dtype=int)
        used_alpha = np.zeros(N_ANCHOR, dtype=bool)
        used_k     = np.zeros(cand_vecs.shape[1], dtype=bool)
        for _ in range(N_ANCHOR):
            # Mask assigned rows and columns
            Om = O.copy()
            Om[used_alpha, :] = -1
            Om[:, used_k] = -1
            alpha, k = np.unravel_index(np.argmax(Om), Om.shape)
            assigned[alpha] = k
            used_alpha[alpha] = True
            used_k[k] = True

        # Extract per-mode overlap
        mode_overlaps = np.array(
            [O[alpha, assigned[alpha]] for alpha in range(N_ANCHOR)]
        )
        omega_min = mode_overlaps.min()

        print(f"       min overlap across 20 assigned modes: "
              f"{omega_min:.4f}")
        print(f"       mean overlap: {mode_overlaps.mean():.4f}")
        print(f"       all >= 0.5: "
              f"{(mode_overlaps >= 0.5).all()}")

        min_overlaps[xi] = float(omega_min)
        assigned_overlaps[xi] = mode_overlaps
        mode_indices[xi] = assigned

    # Cache
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache_dict = {
        "anchor_eigvals": anchor_vals,
        "xi_nodes": np.array(XI_NODES),
        "n_anchor": N_ANCHOR,
    }
    for xi in XI_NODES[1:]:
        key = f"overlaps_xi_{xi:.3f}".replace(".", "p")
        cache_dict[key] = assigned_overlaps[xi]
        cache_dict[f"eigvals_xi_{xi:.3f}".replace(".", "p")] = \
            eigenspectra[xi][0]
        cache_dict[f"O_matrix_xi_{xi:.3f}".replace(".", "p")] = \
            overlap_matrices[xi]
    np.savez_compressed(CACHE_PATH, **cache_dict)
    print(f"\n[cache]    saved: {CACHE_PATH}")
    print(f"[total]    {time.time()-t_all:.1f} s")

    return assigned_overlaps, eigenspectra


def load_cached():
    if not os.path.exists(CACHE_PATH):
        return None
    data = np.load(CACHE_PATH)
    out = {}
    for xi in XI_NODES[1:]:
        key = f"overlaps_xi_{xi:.3f}".replace(".", "p")
        out[xi] = data[key]
    return out


def plot_figure(assigned_overlaps):
    """Mode-wise overlap plot for the two non-anchor xi values."""
    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    colors = {1.045: "tab:blue", 1.090: "tab:orange"}
    markers = {1.045: "o", 1.090: "s"}

    for xi, overlaps in assigned_overlaps.items():
        alpha = np.arange(1, len(overlaps) + 1)
        ax.plot(alpha, overlaps,
                marker=markers[xi],
                color=colors[xi],
                markersize=7,
                linewidth=1.0,
                linestyle='-',
                markeredgecolor='black',
                markeredgewidth=0.5,
                label=rf"$\xi = {xi:.3f}$, "
                      rf"$\Omega_{{\min}} = {overlaps.min():.3f}$",
                zorder=3)

    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8,
               alpha=0.7, zorder=1)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8,
               alpha=0.5, zorder=1,
               label="0.5 reference (acceptable transport)")

    ax.set_xlabel(r"mode index $\alpha$", fontsize=12)
    ax.set_ylabel(
        r"$|\langle\psi_\alpha(\xi_0)\,|\,\hat\psi_\alpha(\xi_j)\rangle|$",
        fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='lower left', frameon=True, framealpha=0.95, fontsize=9.5)

    ax.set_xlim(0, 21)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(1, 21))

    # Annotation: Row A, algorithm
    ax.text(0.98, 0.02,
            rf"Row A: $L_s = {L_SPATIAL}$, $L_t = {L_TEMPORAL}$, "
            rf"$\rho = {RHO}$",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    plt.savefig(FIG_PATH, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {FIG_PATH}")
    print(f"File size: {os.path.getsize(FIG_PATH)} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true",
                        help="Load from cached overlaps")
    args = parser.parse_args()

    if args.cache:
        cached = load_cached()
        if cached is None:
            print(f"No cache at {CACHE_PATH}; running full compute.")
            assigned_overlaps, _ = compute_transport()
        else:
            assigned_overlaps = cached
            print(f"[cache]    loaded: {CACHE_PATH}")
            for xi, ov in cached.items():
                print(f"  xi = {xi:.3f}: {len(ov)} modes, "
                      f"min = {ov.min():.4f}, mean = {ov.mean():.4f}")
    else:
        assigned_overlaps, _ = compute_transport()

    plot_figure(assigned_overlaps)


if __name__ == "__main__":
    main()
