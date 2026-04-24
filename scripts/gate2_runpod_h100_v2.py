#!/usr/bin/env python3
"""
gate2_runpod_h100_v2.py
=======================
FULLY GPU-OPTIMIZED Gate 2 closure for H100 GPU cluster.

Changes in v2:
  - Completely removed the scipy (CPU) projected_solve.
  - Implemented compute_traces_gpu: keeps the 9.4M dimension matrices 
    strictly inside the H100 VRAM during the 80-sample loop.
  - Uses cupyx.scipy.sparse.linalg.cg to utilize all 14,000+ CUDA cores.
"""
import sys, time, os
import numpy as np
from scipy import sparse

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_splinalg
    GPU = True
    mempool = cp.get_default_memory_pool()
    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props['name'].decode()
    dev = cp.cuda.Device(0)
    total_mem = dev.mem_info[1] / 1e9
    free_mem = dev.mem_info[0] / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_mem:.1f} GB total, {free_mem:.1f} GB free")
except ImportError:
    GPU = False
    print("FATAL WARNING: CuPy not available. This script requires a GPU.")
    sys.exit(1)

sys.path.insert(0, '.')
from src.caloron import init_caloron
from src.observables import full_diagnostic
from src.spectral import build_vector_operator, build_adjoint_laplacian

try:
    from exact_prime_F_G import build_delta_F_operator
    HAS_DELTA_F = True
    print("delta-F operator: available")
except ImportError:
    HAS_DELTA_F = False
    print("WARNING: delta-F not available")

M2 = 0.01
EPS = 0.09
XI_VALUES = [1.0, 1.0 + EPS/2, 1.0 + EPS]
N_S = 80         
N_GHOST_LOW = 3
N_LOW_REF = 20
N_EIGSH = 35

# Commented out the smaller configs so you can resume straight on the 32x24
CONFIGS = [
    # (18, 12, 3.0, [5.5, 8.5, 8.5, 8.5], "18x12_rho3.0"),
    # (24, 16, 3.5, [7.5, 11.5, 11.5, 11.5], "24x16_rho3.5"),
    (32, 24, 4.5, [11.5, 15.5, 15.5, 15.5], "32x24_rho4.5"),
]


def gpu_eigsh_sa(mat, k, maxiter=8000):
    """GPU-accelerated smallest-algebraic eigsh."""
    t0 = time.time()
    gm = cp_sparse.csr_matrix(mat.tocsr())
    t_xfer = time.time() - t0
    t0 = time.time()
    ev, ec = cp_splinalg.eigsh(gm, k=k, which='SA', maxiter=maxiter)
    t_solve = time.time() - t0
    out_ev, out_ec = cp.asnumpy(ev), cp.asnumpy(ec)
    del gm, ev, ec
    mempool.free_all_blocks()
    print(f"      GPU eigsh: xfer={t_xfer:.1f}s, solve={t_solve:.1f}s")
    return out_ev, out_ec


def build_K0_12V(K0_3V, V):
    """Promote 3V ghost temporal kinetic to 12V vector space."""
    K0_coo = K0_3V.tocoo()
    n_nz = len(K0_coo.data)
    rows = np.empty(4 * n_nz, dtype=np.int64)
    cols = np.empty(4 * n_nz, dtype=np.int64)
    data = np.empty(4 * n_nz, dtype=np.float64)
    r_3V = K0_coo.row.astype(np.int64)
    c_3V = K0_coo.col.astype(np.int64)
    d_3V = K0_coo.data.astype(np.float64)
    for mu in range(4):
        sl = slice(mu * n_nz, (mu + 1) * n_nz)
        rows[sl] = 12 * (r_3V // 3) + 3 * mu + (r_3V % 3)
        cols[sl] = 12 * (c_3V // 3) + 3 * mu + (c_3V % 3)
        data[sl] = d_3V
    return sparse.csr_matrix((data, (rows, cols)), shape=(12*V, 12*V))


def match_modes_by_overlap(evecs_ref, evecs_new, n_match):
    """Greedy overlap matching of n_match reference modes."""
    n_new = evecs_new.shape[1]
    O = (evecs_ref[:, :n_match].T @ evecs_new)**2
    matched, overlaps, used = [], [], set()
    for i in range(n_match):
        best_j, best_ov = -1, -1
        for j in range(n_new):
            if j not in used and O[i, j] > best_ov:
                best_ov, best_j = O[i, j], j
        matched.append(best_j)
        overlaps.append(best_ov)
        used.add(best_j)
    return np.array(matched), np.array(overlaps)


def compute_traces_gpu(A_mat, K0_mat, deltaF_mat, P_low, Z_noise, N_S, dim, name):
    """
    ULTRA-FAST GPU TRACE SOLVER:
    Pushes the entire sparse matrix to the H100 VRAM and runs the CG solver 
    using all 14,000+ CUDA cores. Bypasses the CPU completely.
    """
    # 1. Push operators to VRAM
    A_gpu = cp_sparse.csr_matrix(A_mat)
    K0_gpu = cp_sparse.csr_matrix(K0_mat)
    dF_gpu = cp_sparse.csr_matrix(deltaF_mat) if deltaF_mat is not None else None
        
    P_gpu = cp.array(P_low)
    Z_gpu = cp.array(Z_noise)

    def project(v):
        return v - P_gpu @ (P_gpu.T @ v)

    def matvec(v):
        return project(A_gpu @ project(v))

    A_op = cp_splinalg.LinearOperator((dim, dim), matvec=matvec, dtype=cp.float64)

    T_res = np.zeros(N_S)
    F_res = np.zeros(N_S)
    n_fail = 0

    # 2. Run the loop entirely on the GPU
    for s in range(N_S):
        z_s = Z_gpu[:, s]
        b_p = project(z_s)
        
        try:
            x_s, info = cp_splinalg.cg(A_op, b_p, tol=1e-10, maxiter=8000)
        except Exception as e:
            info = -1
            x_s = b_p
            
        if info != 0:
            n_fail += 1
            
        x_s_proj = project(x_s)
        
        T_res[s] = float(cp.dot(z_s, K0_gpu @ x_s_proj))
        if dF_gpu is not None:
            F_res[s] = float(cp.dot(z_s, dF_gpu @ x_s_proj))

    # 3. Dump the VRAM to prevent Out-Of-Memory crashing
    del A_gpu, K0_gpu, P_gpu, Z_gpu, A_op, b_p, z_s, x_s, x_s_proj
    if dF_gpu is not None: del dF_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return T_res, F_res, n_fail


def run_config(L_s, L_t, rho, x0, label):
    print(f"\n{'='*70}")
    print(f"  {label}: {L_s}^3 x {L_t}, rho={rho}")
    print(f"  V = {L_t * L_s**3}, dim = {12 * L_t * L_s**3}")
    print(f"{'='*70}")

    shape = (L_t, L_s, L_s, L_s)
    V = int(np.prod(shape))
    dim_v = 12 * V
    dim_g = 3 * V

    t0_total = time.time()
    print(f"\n  Building caloron...")
    t0 = time.time()
    U = init_caloron(L_s, L_t, rho=rho, x0=x0, self_dual=True, n_sub=16)
    print(f"  Caloron: {time.time()-t0:.0f}s")
    full_diagnostic(U, label)

    print(f"  Building operators...")
    t0 = time.time()
    Delta1_base = build_vector_operator(U, mass_sq=M2).tocsr()
    Delta0_base = build_adjoint_laplacian(U, mass_sq=M2).tocsr()
    K0_ghost = build_adjoint_laplacian(U, temporal_only=True).tocsr()
    K0_12V = build_K0_12V(K0_ghost, V)
    print(f"  Main operators: {time.time()-t0:.0f}s")

    if HAS_DELTA_F:
        t0 = time.time()
        deltaF_12V = build_delta_F_operator(U, (L_t, L_s, L_s, L_s)).tocsr()
        print(f"  deltaF: {time.time()-t0:.0f}s, nnz={deltaF_12V.nnz}")
    else:
        deltaF_12V = None

    U_free = np.zeros_like(U)
    for mu in range(4):
        U_free[..., mu, :, :] = np.eye(2, dtype=complex)
    Delta1_free = build_vector_operator(U_free, mass_sq=M2).tocsr()
    Delta0_free = build_adjoint_laplacian(U_free, mass_sq=M2).tocsr()
    K0_ghost_free = build_adjoint_laplacian(U_free, temporal_only=True).tocsr()
    K0_12V_free = build_K0_12V(K0_ghost_free, V)
    print(f"  All operators: {time.time()-t0_total:.0f}s")

    print(f"  Ghost prime...")
    gh_evals, gh_evecs = gpu_eigsh_sa(Delta0_base, k=N_GHOST_LOW + 2)
    gh_order = np.argsort(gh_evals)
    gh_evecs_inst = gh_evecs[:, gh_order[:N_GHOST_LOW]]
    gh_evals_f, gh_evecs_f = gpu_eigsh_sa(Delta0_free, k=N_GHOST_LOW + 2)
    gh_order_f = np.argsort(gh_evals_f)
    gh_evecs_free = gh_evecs_f[:, gh_order_f[:N_GHOST_LOW]]

    print(f"  Generating {N_S} paired noise vectors...")
    rng = np.random.RandomState(42)
    Z_v = rng.choice([-1.0, 1.0], size=(dim_v, N_S))
    Z_g = rng.choice([-1.0, 1.0], size=(dim_g, N_S))

    print(f"\n  REFERENCE LOW MANIFOLD:")
    evals_ref, evecs_ref = gpu_eigsh_sa(Delta1_base, k=N_EIGSH)
    order_ref = np.argsort(evals_ref)
    evals_ref, evecs_ref = evals_ref[order_ref], evecs_ref[:, order_ref]
    P_ref = evecs_ref[:, :N_LOW_REF]

    evals_ref_f, evecs_ref_f = gpu_eigsh_sa(Delta1_free, k=N_EIGSH)
    order_ref_f = np.argsort(evals_ref_f)
    evecs_ref_f = evecs_ref_f[:, order_ref_f]
    P_ref_free = evecs_ref_f[:, :N_LOW_REF]

    results_by_xi = {}

    for xi_idx, xi in enumerate(XI_VALUES):
        print(f"\n  {'='*60}")
        print(f"  xi = {xi:.4f} (index {xi_idx})")
        print(f"  {'='*60}")

        xi2m1, xim1 = xi**2 - 1.0, xi - 1.0

        Delta1_xi = Delta1_base + xi2m1 * K0_12V
        if deltaF_12V is not None:
            Delta1_xi = Delta1_xi + xim1 * deltaF_12V
        Delta1_xi = Delta1_xi.tocsr()

        Delta0_xi = (Delta0_base + xi2m1 * K0_ghost).tocsr()
        Delta1_free_xi = (Delta1_free + xi2m1 * K0_12V_free).tocsr()
        Delta0_free_xi = (Delta0_free + xi2m1 * K0_ghost_free).tocsr()

        print(f"    Eigsh at xi={xi:.4f}...")
        evals_xi, evecs_xi = gpu_eigsh_sa(Delta1_xi, k=N_EIGSH)
        order = np.argsort(evals_xi)
        evals_xi, evecs_xi = evals_xi[order], evecs_xi[:, order]

        matched_idx, matched_ov = match_modes_by_overlap(P_ref, evecs_xi, N_LOW_REF)
        min_ov = np.min(matched_ov)
        max_lam = max(evals_xi[j] for j in matched_idx)
        print(f"    Transport: min_ov={min_ov:.4f}, max_lam={max_lam:.6f}")

        P_low_v = evecs_xi[:, matched_idx]
        P_low_v, _ = np.linalg.qr(P_low_v, mode='reduced')

        unmatched = sorted(set(range(len(evals_xi))) - set(matched_idx))
        unmatched_above = [j for j in unmatched if evals_xi[j] > max_lam]
        eff_gap = (evals_xi[unmatched_above[0]] - max_lam if unmatched_above else 0)

        evals_f_xi, evecs_f_xi = gpu_eigsh_sa(Delta1_free_xi, k=N_EIGSH)
        order_f = np.argsort(evals_f_xi)
        evecs_f_xi = evecs_f_xi[:, order_f]
        matched_f, _ = match_modes_by_overlap(P_ref_free, evecs_f_xi, N_LOW_REF)
        P_low_v_free = evecs_f_xi[:, matched_f]
        P_low_v_free, _ = np.linalg.qr(P_low_v_free, mode='reduced')

        print(f"    Traces ({N_S} paired, n_low={N_LOW_REF}) executing on GPU...")
        t0 = time.time()
        
        # This replaces the entire slow CPU loop!
        T_v, F_v, f_v = compute_traces_gpu(Delta1_xi, K0_12V, deltaF_12V, P_low_v, Z_v, N_S, dim_v, "Vec")
        T_vf, _, f_vf = compute_traces_gpu(Delta1_free_xi, K0_12V_free, None, P_low_v_free, Z_v, N_S, dim_v, "Free Vec")
        G_gh, _, f_g = compute_traces_gpu(Delta0_xi, K0_ghost, None, gh_evecs_inst, Z_g, N_S, dim_g, "Ghost")
        G_ghf, _, f_gf = compute_traces_gpu(Delta0_free_xi, K0_ghost_free, None, gh_evecs_free, Z_g, N_S, dim_g, "Free Gh")

        n_fail = f_v + f_vf + f_g + f_gf
        print(f"    Done in {time.time()-t0:.0f}s ({n_fail} failures)")

        T_sub = T_v - T_vf; G_sub = G_gh - G_ghf; F_sub = F_v
        T_m, T_e = np.mean(T_sub), np.std(T_sub)/np.sqrt(N_S)
        F_m, F_e = np.mean(F_sub), np.std(F_sub)/np.sqrt(N_S)
        G_m, G_e = np.mean(G_sub), np.std(G_sub)/np.sqrt(N_S)

        h_m = -xi*T_m - 0.5*F_m + 2*xi*G_m
        h_e = np.sqrt((xi*T_e)**2 + (0.5*F_e)**2 + (2*xi*G_e)**2)
        h_samp = -xi*T_sub - 0.5*F_sub + 2*xi*G_sub

        print(f"    T={T_m:.3f}+/-{T_e:.3f}, F={F_m:.3f}+/-{F_e:.3f}, G={G_m:.3f}+/-{G_e:.3f}")
        print(f"    h={h_m:.3f}+/-{h_e:.3f} [gap={eff_gap:.4f}, ov={min_ov:.3f}]")

        results_by_xi[xi] = {
            'T': T_m, 'F': F_m, 'G': G_m, 'h': h_m, 'h_err': h_e,
            'h_samples': h_samp, 'eff_gap': eff_gap, 'min_ov': min_ov, 'max_lam': max_lam,
        }

    print(f"\n  {'='*60}")
    print(f"  INTEGRATION AND ASSEMBLY: {label}")
    print(f"  {'='*60}")

    h0 = results_by_xi[XI_VALUES[0]]['h_samples']
    h1 = results_by_xi[XI_VALUES[1]]['h_samples']
    h2 = results_by_xi[XI_VALUES[2]]['h_samples']
    h_vals = [results_by_xi[xi]['h'] for xi in XI_VALUES]

    dx = EPS
    simp_samp = (dx/6)*(h0 + 4*h1 + h2)
    simp_m, simp_e = np.mean(simp_samp), np.std(simp_samp)/np.sqrt(N_S)

    J = 0.118164
    dlz_1st = EPS*h_vals[0] + J
    dlz_simp = simp_m + J
    B_1st = 1.0 - np.exp(dlz_1st)
    B_simp = 1.0 - np.exp(dlz_simp)
    nonlin = dlz_simp - dlz_1st
    nonlin_pct = 100*abs(nonlin)/abs(dlz_1st)

    print(f"\n  Integrands:")
    for xi in XI_VALUES:
        r = results_by_xi[xi]
        print(f"    h({xi:.3f}) = {r['h']:.3f}+/-{r['h_err']:.3f}")

    print(f"\n  dlz_full (Simpson)   = {dlz_simp:.4f} +/- {simp_e:.4f}")
    print(f"  Bracket (Simpson)    = {B_simp:.4f}")
    
    in_range = -2.0 < dlz_simp < -1.3
    small_corr = nonlin_pct < 20
    no_flip = dlz_simp < 0
    gap_ok = all(results_by_xi[xi]['eff_gap'] > 0.003 for xi in XI_VALUES)

    gate2_pass = in_range and small_corr and no_flip and gap_ok
    print(f"\n  GATE 2 PASS: {gate2_pass}")

    return {'label': label, 'dlz_1st': dlz_1st, 'dlz_simp': dlz_simp, 'dlz_err': simp_e, 'nonlin_pct': nonlin_pct, 'B_simp': B_simp, 'gate2_pass': gate2_pass}


if __name__ == '__main__':
    print("=" * 70)
    print("GATE 2 DEFINITIVE: RunPod H100 (V2 GPU OPTIMIZED)")
    print("=" * 70)

    results = []
    for L_s, L_t, rho, x0, label in CONFIGS:
        try:
            r = run_config(L_s, L_t, rho, x0, label)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR on {label}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nDONE")