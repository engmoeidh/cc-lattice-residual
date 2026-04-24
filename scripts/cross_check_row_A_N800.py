#!/usr/bin/env python3
"""
cross_check_row_A_N800.py
=========================
Higher-sample rerun of the projected-complement exact prime on Row A.
N_S = 800 paired samples, seed = 42, same pipeline as N_S = 80 run.

Target: tighten exact-prime Simpson error from 3.12 to ~1.0, enabling a
meaningful test of scheme agreement at the ~1σ level.

Expected wall time on H100: ~7.5 hours.
"""

import sys
import os
import time
import numpy as np
from scipy import sparse

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_splinalg
    USE_GPU = True
    mempool = cp.get_default_memory_pool()
    dev = cp.cuda.Device(0)
    print(f"GPU detected: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"VRAM: {dev.mem_info[1]/1e9:.1f} GB total, {dev.mem_info[0]/1e9:.1f} GB free")
except ImportError:
    USE_GPU = False
    print("FATAL: CuPy required for this run.")
    sys.exit(1)

sys.path.insert(0, '.')

from src.caloron import init_caloron
from src.spectral import build_vector_operator, build_adjoint_laplacian

from jacobian_left_triv import raw_tangent_translation_LT, raw_tangent_scale_LT
from jacobian_exact import gauge_clean_tangent
from exact_prime_F_G import build_delta_F_operator

# ============================================================
#  Configuration
# ============================================================

CONFIG = dict(
    L_s=18, L_t=12, rho=3.0,
    x0=[5.5, 8.5, 8.5, 8.5],
    mass_sq=0.01,
    eps=0.09,
    n_samples=800,                  # << 10× the original 80
    rtol=1e-8,
    maxiter=3000,
    seed=42,
    label="Row A: 18^3 x 12, rho=3.0",
    adaptive_prime_delta=-1.660,
    adaptive_prime_err=0.186,
    jacobian=0.118164,
    checkpoint_dir="./checkpoints_N800",
    progress_every=100,
)

XI_VALUES = [1.0, 1.0 + CONFIG['eps']/2, 1.0 + CONFIG['eps']]

# ============================================================
#  Helpers (identical to N=80 run)
# ============================================================

def build_K0_12V(K0_3V, V_sites):
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
    return sparse.csr_matrix((data, (rows, cols)), shape=(12*V_sites, 12*V_sites))

def orthonormalize(vectors):
    ortho = []
    for v in vectors:
        w = v.copy()
        for u in ortho:
            w -= np.vdot(u, w) * u
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            ortho.append(w / norm)
    return ortho

def projected_cg_gpu(A_gpu, b_gpu, V_gpu_stacked, rtol=1e-8, maxiter=3000):
    n = b_gpu.size
    def project(v):
        if V_gpu_stacked is None:
            return v
        return v - V_gpu_stacked @ (V_gpu_stacked.conj().T @ v)
    x = cp.zeros(n, dtype=A_gpu.dtype)
    r = project(b_gpu.copy())
    p = r.copy()
    rs_old = cp.vdot(r, r).real
    b_norm = cp.linalg.norm(b_gpu)
    if b_norm < 1e-30:
        return x, 0
    for _ in range(maxiter):
        Ap = project(A_gpu @ p)
        pAp = cp.vdot(p, Ap).real
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = cp.vdot(r, r).real
        if cp.sqrt(rs_new) / b_norm < rtol:
            return project(x), 0
        beta = rs_new / rs_old
        p = r + beta * p
        p = project(p)
        rs_old = rs_new
    return project(x), 1

def run_trace_projected(A_sparse, B_sparse, V_basis_np, n_samples, rtol, maxiter,
                        rng, dim, label, progress_every=100):
    n_fail = 0
    samples = []
    t_start = time.time()

    A_gpu = cp_sparse.csr_matrix(A_sparse)
    B_gpu = cp_sparse.csr_matrix(B_sparse) if B_sparse is not None else None
    if len(V_basis_np) > 0:
        V_gpu = cp.asarray(np.column_stack(V_basis_np), dtype=A_gpu.dtype)
    else:
        V_gpu = None

    for s in range(n_samples):
        eta_np = rng.choice([-1.0, 1.0], size=dim).astype(np.float64)
        eta = cp.asarray(eta_np, dtype=A_gpu.dtype)
        b = B_gpu @ eta if B_gpu is not None else eta.copy()
        b_p = b - V_gpu @ (V_gpu.conj().T @ b) if V_gpu is not None else b
        x, info = projected_cg_gpu(A_gpu, b_p, V_gpu, rtol=rtol, maxiter=maxiter)
        if info != 0:
            n_fail += 1
        val = cp.vdot(eta, x).real
        samples.append(float(val))

        if (s + 1) % progress_every == 0:
            arr = np.array(samples)
            running_mean = float(np.mean(arr))
            running_err = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            elapsed = time.time() - t_start
            rate = (s + 1) / elapsed
            remaining = (n_samples - s - 1) / rate
            print(f"    [{label}]  s={s+1}/{n_samples}  "
                  f"mean={running_mean:+.4f}  err={running_err:.4f}  "
                  f"n_fail={n_fail}  eta={remaining/60:.1f}min")

    del A_gpu, V_gpu, eta, b, b_p, x
    if B_gpu is not None:
        del B_gpu
    mempool.free_all_blocks()

    samples = np.array(samples)
    mean = float(np.mean(samples))
    err = float(np.std(samples, ddof=1) / np.sqrt(n_samples))
    total_time = time.time() - t_start
    print(f"    [{label}] FINAL: {mean:+.4f} ± {err:.4f}  "
          f"(n_fail={n_fail}, {total_time/60:.1f}min)")
    return mean, err, samples, n_fail

# ============================================================
#  Main
# ============================================================

def main():
    print("="*78)
    print("CROSS-CHECK: PROJECTED-COMPLEMENT EXACT PRIME ON ROW A (N=800)")
    print("="*78)
    print(f"  Config: {CONFIG['label']}")
    print(f"  n_samples = {CONFIG['n_samples']}, seed = {CONFIG['seed']}")
    print(f"  Target error: ~1.0 (10× tighter than N=80 run)")
    print(f"  Expected runtime: ~7.5h on H100")
    print()

    L_s, L_t, rho = CONFIG['L_s'], CONFIG['L_t'], CONFIG['rho']
    x0, m2, eps = CONFIG['x0'], CONFIG['mass_sq'], CONFIG['eps']
    N_S = CONFIG['n_samples']
    rtol, maxiter, seed = CONFIG['rtol'], CONFIG['maxiter'], CONFIG['seed']
    progress_every = CONFIG['progress_every']

    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    dims = (L_t, L_s, L_s, L_s)
    V_sites = int(np.prod(dims))
    dim_v, dim_g = 12 * V_sites, 3 * V_sites

    t0 = time.time()
    print("[1] Building caloron...")
    U = init_caloron(L_s, L_t, rho=rho, x0=x0, self_dual=True, n_sub=16)
    print(f"    done ({time.time()-t0:.0f}s)")

    U_free = np.zeros_like(U)
    for mu in range(4):
        U_free[..., mu, :, :] = np.eye(2, dtype=complex)

    print("\n[2] Building five physical moduli tangents...")
    raw_tangents = [raw_tangent_translation_LT(U, L_s, L_t, rho, x0, mu, 0.1,
                                                self_dual=True, n_sub=16)
                    for mu in range(4)]
    raw_tangents.append(raw_tangent_scale_LT(U, L_s, L_t, rho, x0, 0.05,
                                              self_dual=True, n_sub=16))

    cleaned = []
    for I, Yraw in enumerate(raw_tangents):
        Z, _, info = gauge_clean_tangent(U, Yraw, mass_sq=1e-6,
                                          cg_tol=1e-10, cg_maxiter=5000)
        cleaned.append(Z)
        print(f"    Mode {I}: cleaning ratio = {info['cleaning_ratio']:.3e}")

    vecs = [t.reshape(dim_v) for t in cleaned]
    V_mod = orthonormalize(vecs)

    V_free = []
    for mu in range(4):
        for a in range(3):
            v = np.zeros(dim_v)
            v[3*mu + a::12] = 1.0 / np.sqrt(V_sites)
            V_free.append(v)

    print("\n[3] Building operators...")
    t0 = time.time()
    Delta1_base = build_vector_operator(U, mass_sq=m2).tocsr()
    Delta0_base = build_adjoint_laplacian(U, mass_sq=m2).tocsr()
    K0_ghost = build_adjoint_laplacian(U, temporal_only=True).tocsr()
    K0_12V = build_K0_12V(K0_ghost, V_sites)
    dF_inst = build_delta_F_operator(U, dims).tocsr()

    Delta1_free = build_vector_operator(U_free, mass_sq=m2).tocsr()
    Delta0_free = build_adjoint_laplacian(U_free, mass_sq=m2).tocsr()
    K0_ghost_free = build_adjoint_laplacian(U_free, temporal_only=True).tocsr()
    K0_12V_free = build_K0_12V(K0_ghost_free, V_sites)

    V_ghost_free = []
    for a in range(3):
        v = np.zeros(dim_g)
        v[a::3] = 1.0 / np.sqrt(V_sites)
        V_ghost_free.append(v)
    print(f"    operators built in {time.time()-t0:.0f}s")

    print("\n[4] Running projected-complement exact prime at three ξ values...")
    print(f"    Progress reported every {progress_every} samples.")
    results_by_xi = {}
    rng = np.random.RandomState(seed)

    for i_xi, xi in enumerate(XI_VALUES):
        print(f"\n  ═══════ ξ = {xi:.4f}  (xi {i_xi+1}/3) ═══════")
        xi2m1, xim1 = xi**2 - 1.0, xi - 1.0
        t_xi_start = time.time()

        Delta1_xi = (Delta1_base + xi2m1 * K0_12V + xim1 * dF_inst).tocsr()
        Delta0_xi = (Delta0_base + xi2m1 * K0_ghost).tocsr()
        Delta1_free_xi = (Delta1_free + xi2m1 * K0_12V_free).tocsr()
        Delta0_free_xi = (Delta0_free + xi2m1 * K0_ghost_free).tocsr()

        print("\n    T (inst − free)...")
        _, _, T_inst_samp, _ = run_trace_projected(
            Delta1_xi, K0_12V, V_mod, N_S, rtol, maxiter, rng, dim_v,
            "T_inst", progress_every)
        _, _, T_free_samp, _ = run_trace_projected(
            Delta1_free_xi, K0_12V_free, V_free, N_S, rtol, maxiter, rng, dim_v,
            "T_free", progress_every)
        T_samp = T_inst_samp - T_free_samp

        print("\n    F (inst, dF_free = 0)...")
        _, _, F_samp, _ = run_trace_projected(
            Delta1_xi, dF_inst, V_mod, N_S, rtol, maxiter, rng, dim_v,
            "F_inst", progress_every)

        print("\n    G (ghost, inst − free)...")
        _, _, G_inst_samp, _ = run_trace_projected(
            Delta0_xi, K0_ghost, [], N_S, rtol, maxiter, rng, dim_g,
            "G_inst", progress_every)
        _, _, G_free_samp, _ = run_trace_projected(
            Delta0_free_xi, K0_ghost_free, V_ghost_free, N_S, rtol, maxiter, rng, dim_g,
            "G_free", progress_every)
        G_samp = G_inst_samp - G_free_samp

        h_samp = -xi*T_samp - 0.5*F_samp + 2*xi*G_samp
        h_mean = float(np.mean(h_samp))
        h_err = float(np.std(h_samp, ddof=1) / np.sqrt(N_S))
        t_xi_total = time.time() - t_xi_start
        print(f"\n    h(ξ={xi:.4f}) = {h_mean:+.4f} ± {h_err:.4f}  "
              f"({t_xi_total/60:.1f}min for this ξ)")

        results_by_xi[xi] = dict(
            T=float(np.mean(T_samp)),
            F=float(np.mean(F_samp)),
            G=float(np.mean(G_samp)),
            h=h_mean, h_err=h_err,
            h_samples=h_samp,
            T_samples=T_samp,
            F_samples=F_samp,
            G_samples=G_samp,
        )

        # Checkpoint: save the per-ξ samples
        ckpt_path = os.path.join(CONFIG['checkpoint_dir'],
                                  f"xi_{xi:.4f}.npz")
        np.savez(ckpt_path,
                 xi=xi, T_samples=T_samp, F_samples=F_samp,
                 G_samples=G_samp, h_samples=h_samp)
        print(f"    checkpoint saved: {ckpt_path}")

    # Simpson integration
    h0 = results_by_xi[XI_VALUES[0]]['h_samples']
    h1 = results_by_xi[XI_VALUES[1]]['h_samples']
    h2 = results_by_xi[XI_VALUES[2]]['h_samples']
    simp_samp = (eps/6.0) * (h0 + 4*h1 + h2)
    dlz_simp = float(np.mean(simp_samp)) + CONFIG['jacobian']
    dlz_err = float(np.std(simp_samp, ddof=1) / np.sqrt(N_S))
    B_exact = 1.0 - np.exp(dlz_simp)

    adpt, adpt_err = CONFIG['adaptive_prime_delta'], CONFIG['adaptive_prime_err']
    diff = dlz_simp - adpt
    combined_err = np.sqrt(dlz_err**2 + adpt_err**2)
    tension = abs(diff) / combined_err

    print("\n" + "="*78)
    print(f"CROSS-CHECK RESULTS — ROW A (N_S = {N_S})")
    print("="*78)
    for xi in XI_VALUES:
        r = results_by_xi[xi]
        print(f"  h({xi:.3f}) = {r['h']:+.4f} ± {r['h_err']:.4f}   "
              f"[T={r['T']:+.2f}, F={r['F']:+.2f}, G={r['G']:+.2f}]")
    print(f"\n  δ_ξ log ζ (exact prime, Simpson)  = {dlz_simp:+.4f} ± {dlz_err:.4f}")
    print(f"  δ_ξ log ζ (adaptive prime, H100)  = {adpt:+.4f} ± {adpt_err:.4f}")
    print(f"  Scheme difference                 = {diff:+.4f}")
    print(f"  Combined uncertainty              = {combined_err:.4f}")
    print(f"  Tension                           = {tension:.2f}σ")
    print(f"\n  Bracket (exact prime)             = {B_exact:.4f}")
    print(f"  Bracket (adaptive prime)          = {1-np.exp(adpt):.4f}")

    # Save final state
    final_path = os.path.join(CONFIG['checkpoint_dir'], "final_result.npz")
    np.savez(final_path,
             dlz_simp=dlz_simp, dlz_err=dlz_err,
             tension=tension, diff=diff,
             N_S=N_S, seed=seed,
             xi_values=XI_VALUES,
             **{f"h_samples_xi{i}": results_by_xi[xi]['h_samples']
                for i, xi in enumerate(XI_VALUES)})
    print(f"\n  Final state saved: {final_path}")

    # Verdict
    print("\n" + "="*78)
    print("VERDICT")
    print("="*78)
    if tension < 1.0:
        print(f"  PASS (strong): {tension:.2f}σ agreement.")
        print("  Adaptive prime validated by independent scheme at ~1σ precision.")
    elif tension < 2.0:
        print(f"  PASS: {tension:.2f}σ agreement.")
        print("  Two prescriptions agree within combined error.")
    elif tension < 3.0:
        print(f"  MARGINAL: {tension:.2f}σ tension.")
        print("  Scheme difference is present but within 3σ.")
        print("  Honest disclosure in paper required.")
    else:
        print(f"  FAIL: {tension:.2f}σ tension.")
        print("  Genuine scheme disagreement. Requires discussion.")

if __name__ == '__main__':
    main()