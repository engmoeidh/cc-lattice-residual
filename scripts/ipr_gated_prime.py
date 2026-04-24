#!/usr/bin/env python3
r"""
ipr_gated_prime.py
==================
IPR-gated exact-prime: project out 5 physical LT modes + 8 delocalized
artifact modes identified by:
  |lambda| < lambda_cut  AND  IPR*V < 2  AND  O_phys < 0.05  AND  O_gauge < 0.05

Test hypothesis: the refinement blow-up (-23 -> -62) is caused by
under-subtracting an extended bulk near-zero block.

Two variants:
  P_min   = I - P_phys(5) - P_art(8)         [13 modes removed]
  P_min+g = I - P_phys(5) - P_art(8) - P_gauge(3)  [16 modes removed]
"""
import sys, time
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy import sparse
sys.path.insert(0, '.')

from src.caloron import init_caloron
from src.observables import full_diagnostic
from src.spectral import build_adjoint_laplacian, build_vector_operator
from jacobian_exact import covariant_divergence, gauge_clean_tangent
from jacobian_left_triv import raw_tangent_translation_LT, raw_tangent_scale_LT
from exact_prime_F_G import build_delta_F_operator
from exact_prime_projected import projected_cg


def compute_site_norm_sq(v, V):
    return np.sum(v.reshape(V, 12)**2, axis=1)


def ipr_times_V(v, V):
    norms_sq = compute_site_norm_sq(v, V)
    return V * np.sum(norms_sq**2) / np.sum(norms_sq)**2


def run_ipr_gated(L_s, L_t, rho, x0, mass_sq=0.01, n_samples=80,
                   epsilon=0.09, n_eig=25, lambda_cut=0.06,
                   ipr_cut=2.0, ophys_cut=0.05, ogauge_cut=0.05):

    print(f"\n{'='*60}")
    print(f"IPR-GATED EXACT-PRIME: {L_s}^3 x {L_t}, rho={rho}")
    print(f"{'='*60}")

    U = init_caloron(L_s, L_t, rho=rho, x0=x0, self_dual=True, n_sub=16)
    full_diagnostic(U, f"{L_s}^3x{L_t}")
    dims = U.shape[:4]
    V = int(np.prod(dims))

    U_free = np.zeros_like(U)
    for mu in range(4):
        U_free[..., mu, :, :] = np.eye(2, dtype=complex)

    # ── 5 physical LT tangents ───────────────────────────────────────
    print(f"\n[1] Building 5 physical tangents...")
    raw = []
    for mu in range(4):
        Y = raw_tangent_translation_LT(U, L_s, L_t, rho, x0, mu, 0.1,
                                        self_dual=True, n_sub=16)
        raw.append(Y)
    Y = raw_tangent_scale_LT(U, L_s, L_t, rho, x0, 0.05,
                              self_dual=True, n_sub=16)
    raw.append(Y)

    cleaned = []
    for I in range(5):
        Z, _, info = gauge_clean_tangent(U, raw[I], mass_sq=1e-6,
                                          cg_tol=1e-10, cg_maxiter=5000)
        cleaned.append(Z)
    phys_vecs = [t.reshape(12*V) for t in cleaned]
    ortho_phys = []
    for v in phys_vecs:
        w = v.copy()
        for u in ortho_phys:
            w -= np.dot(u, w) * u
        n = np.linalg.norm(w)
        if n > 1e-10:
            ortho_phys.append(w / n)
    print(f"  Physical basis: {len(ortho_phys)} vectors")

    # ── Operators ────────────────────────────────────────────────────
    print(f"\n[2] Building operators...")
    t0 = time.time()
    Delta1_inst = build_vector_operator(U, mass_sq=mass_sq)
    Delta1_free = build_vector_operator(U_free, mass_sq=mass_sq)

    D0sq_ghost = build_adjoint_laplacian(U, mass_sq=0.0, temporal_only=True)
    D0sq_free_ghost = build_adjoint_laplacian(U_free, mass_sq=0.0, temporal_only=True)

    perm = np.zeros(12*V, dtype=int)
    for x in range(V):
        for mu in range(4):
            for a in range(3):
                perm[12*x + 3*mu + a] = 3*V*mu + 3*x + a

    D0sq_12V = sparse.block_diag([D0sq_ghost]*4, format='csr')[perm][:, perm]
    D0sq_free_12V = sparse.block_diag([D0sq_free_ghost]*4, format='csr')[perm][:, perm]

    dF_inst = build_delta_F_operator(U, dims)

    Ghost_inst = build_adjoint_laplacian(U, mass_sq=mass_sq)
    Ghost_free = build_adjoint_laplacian(U_free, mass_sq=mass_sq)
    D0sq_ghost_inst = build_adjoint_laplacian(U, mass_sq=0.0, temporal_only=True)
    D0sq_ghost_free = build_adjoint_laplacian(U_free, mass_sq=0.0, temporal_only=True)
    print(f"  Built in {time.time()-t0:.0f}s")

    # ── Find eigenmodes and classify ─────────────────────────────────
    print(f"\n[3] Finding {n_eig} lowest eigenmodes...")
    vals, vecs = eigsh(Delta1_inst, k=n_eig, which='SM')
    order = np.argsort(np.abs(vals))
    vals = vals[order]
    vecs = vecs[:, order]

    # Build gauge orientation directions for classification
    gauge_dirs = []
    sigma = [np.array([[0,1],[1,0]], dtype=complex),
             np.array([[0,-1j],[1j,0]], dtype=complex),
             np.array([[1,0],[0,-1]], dtype=complex)]
    for a in range(3):
        Y_pg = np.zeros(12*V)
        for x_flat in range(V):
            idx = np.unravel_index(x_flat, dims)
            for mu in range(4):
                U_link = U[idx[0], idx[1], idx[2], idx[3], mu]
                t_a = sigma[a] / (2j)
                Ad_ta = U_link @ t_a @ U_link.conj().T
                diff = Ad_ta - t_a
                for b in range(3):
                    cb = np.trace(sigma[b].conj().T / (-2j) @ diff).real
                    Y_pg[12*x_flat + 3*mu + b] = cb
        norm = np.linalg.norm(Y_pg)
        if norm > 1e-10:
            Y_pg /= norm
        gauge_dirs.append(Y_pg)
    gauge_ortho = []
    for v in gauge_dirs:
        w = v.copy()
        for u in gauge_ortho:
            w -= np.dot(u, w) * u
        n = np.linalg.norm(w)
        if n > 1e-10:
            gauge_ortho.append(w / n)

    # Classify each mode
    art_indices = []
    gauge_indices = []
    phys_eig_indices = []

    print(f"\n  Mode classification:")
    print(f"  {'i':>3} {'lam':>10} {'IPR*V':>8} {'O_ph':>6} {'O_ga':>6} {'class':>12}")
    for i in range(n_eig):
        v = vecs[:, i]
        lam = vals[i]
        ipr_v = ipr_times_V(v, V)
        o_phys = sum(np.dot(u, v)**2 for u in ortho_phys)
        o_gauge = sum(np.dot(u, v)**2 for u in gauge_ortho)

        if o_phys > 0.3:
            cls = "PHYS"
            phys_eig_indices.append(i)
        elif abs(lam) < lambda_cut and ipr_v < ipr_cut and o_phys < ophys_cut and o_gauge < ogauge_cut:
            cls = "ARTIFACT"
            art_indices.append(i)
        elif o_gauge > 0.1:
            cls = "GAUGE"
            gauge_indices.append(i)
        elif abs(lam) < lambda_cut:
            cls = "AMBIG"
        else:
            cls = "BULK"

        print(f"  {i:>3} {lam:>10.6f} {ipr_v:>8.2f} {o_phys:>6.3f} {o_gauge:>6.3f} {cls:>12}")

    print(f"\n  Artifact modes: {len(art_indices)} (indices: {art_indices})")
    print(f"  Gauge modes:    {len(gauge_indices)} (indices: {gauge_indices})")

    # ── Build projection bases ───────────────────────────────────────
    # P_min = I - P_phys(5) - P_art(8)
    art_vecs = [vecs[:, i] for i in art_indices]

    # Orthonormalize artifact vectors against physical basis
    basis_min = list(ortho_phys)  # start with 5 physical
    for v in art_vecs:
        w = v.copy()
        for u in basis_min:
            w -= np.dot(u, w) * u
        n = np.linalg.norm(w)
        if n > 1e-10:
            basis_min.append(w / n)
    n_min = len(basis_min)
    print(f"\n  P_min basis: {n_min} vectors (5 phys + {n_min-5} artifact)")

    # P_min+g = P_min + P_gauge(3)
    gauge_eig_vecs = [vecs[:, i] for i in gauge_indices]
    basis_ming = list(basis_min)
    for v in gauge_eig_vecs:
        w = v.copy()
        for u in basis_ming:
            w -= np.dot(u, w) * u
        n = np.linalg.norm(w)
        if n > 1e-10:
            basis_ming.append(w / n)
    n_ming = len(basis_ming)
    print(f"  P_min+g basis: {n_ming} vectors (5 phys + {n_min-5} artifact + {n_ming-n_min} gauge)")

    # Free zero modes
    free_zeros = []
    for mu in range(4):
        for a in range(3):
            v = np.zeros(12*V)
            for x in range(V):
                v[12*x + 3*mu + a] = 1.0 / np.sqrt(V)
            free_zeros.append(v)
    ghost_free_zeros = []
    for a in range(3):
        v = np.zeros(3*V)
        for x in range(V):
            v[3*x + a] = 1.0 / np.sqrt(V)
        ghost_free_zeros.append(v)

    # ── Run traces for both variants ─────────────────────────────────
    results = {}
    for label, basis in [("P_min", basis_min), ("P_min+g", basis_ming)]:
        print(f"\n{'='*60}")
        print(f"  VARIANT: {label} ({len(basis)} modes removed)")
        print(f"{'='*60}")

        # T
        print(f"  T ({n_samples} samples)...")
        t0 = time.time()
        rng = np.random.default_rng(42)
        T_samples = []
        for s in range(n_samples):
            if (s+1) % 20 == 0:
                print(f"    sample {s+1}/{n_samples} ({time.time()-t0:.0f}s)", flush=True)
            eta = rng.choice([-1.0, 1.0], size=12*V)
            r_i = D0sq_12V @ eta
            for u in basis:
                r_i -= np.dot(u, r_i) * u
            x_i, _ = projected_cg(Delta1_inst, r_i, basis, rtol=1e-8, maxiter=5000)
            r_f = D0sq_free_12V @ eta
            for u in free_zeros:
                r_f -= np.dot(u, r_f) * u
            x_f, _ = projected_cg(Delta1_free, r_f, free_zeros, rtol=1e-8, maxiter=5000)
            T_samples.append(np.dot(eta, x_i) - np.dot(eta, x_f))
        T_val = np.mean(T_samples)
        T_err = np.std(T_samples) / np.sqrt(n_samples)
        print(f"  T = {T_val:.2f} +/- {T_err:.2f} ({time.time()-t0:.0f}s)")

        # F
        print(f"  F ({n_samples} samples)...")
        t0 = time.time()
        rng2 = np.random.default_rng(123)
        F_samples = []
        for s in range(n_samples):
            if (s+1) % 20 == 0:
                print(f"    sample {s+1}/{n_samples} ({time.time()-t0:.0f}s)", flush=True)
            eta = rng2.choice([-1.0, 1.0], size=12*V)
            r_i = dF_inst @ eta
            for u in basis:
                r_i -= np.dot(u, r_i) * u
            x_i, _ = projected_cg(Delta1_inst, r_i, basis, rtol=1e-8, maxiter=5000)
            F_samples.append(np.dot(eta, x_i))
        F_val = np.mean(F_samples)
        F_err = np.std(F_samples) / np.sqrt(n_samples)
        print(f"  F = {F_val:.2f} +/- {F_err:.2f} ({time.time()-t0:.0f}s)")

        # G (same for all variants)
        if 'G_val' not in results:
            print(f"  G ({n_samples} samples)...")
            t0 = time.time()
            rng3 = np.random.default_rng(456)
            G_samples = []
            for s in range(n_samples):
                eta = rng3.choice([-1.0, 1.0], size=3*V)
                Bi = D0sq_ghost_inst @ eta
                xi, _ = projected_cg(Ghost_inst, Bi, [], rtol=1e-8, maxiter=5000)
                Bf = D0sq_ghost_free @ eta
                xf, _ = projected_cg(Ghost_free, Bf, ghost_free_zeros, rtol=1e-8, maxiter=5000)
                G_samples.append(np.dot(eta, xi) - np.dot(eta, xf))
            G_val = np.mean(G_samples)
            G_err = np.std(G_samples) / np.sqrt(n_samples)
            print(f"  G = {G_val:.2f} +/- {G_err:.2f} ({time.time()-t0:.0f}s)")
            results['G_val'] = G_val
            results['G_err'] = G_err
        else:
            G_val = results['G_val']
            G_err = results['G_err']
            print(f"  G = {G_val:.2f} +/- {G_err:.2f} (cached)")

        dlz_eps = -(T_val + 0.5*F_val) + 2*G_val
        dlz_eps_err = np.sqrt(T_err**2 + (0.5*F_err)**2 + (2*G_err)**2)
        dlz_full = epsilon * dlz_eps + 0.118
        dlz_full_err = epsilon * dlz_eps_err

        print(f"\n  {label} RESULT:")
        print(f"    T  = {T_val:.2f} +/- {T_err:.2f}")
        print(f"    F  = {F_val:.2f} +/- {F_err:.2f}")
        print(f"    G  = {G_val:.2f} +/- {G_err:.2f}")
        print(f"    dlz/eps = {dlz_eps:.2f} +/- {dlz_eps_err:.2f}")
        print(f"    dlz_full = {dlz_full:.4f} +/- {dlz_full_err:.4f}")

        results[label] = {
            'T': T_val, 'T_err': T_err,
            'F': F_val, 'F_err': F_err,
            'dlz_eps': dlz_eps, 'dlz_eps_err': dlz_eps_err,
            'dlz_full': dlz_full, 'dlz_full_err': dlz_full_err,
            'n_removed': len(basis),
        }

    return results


if __name__ == "__main__":
    epsilon = 0.09

    # Run on both lattices
    print("=" * 60)
    print("ROW 1: 12^3x8 rho=3")
    print("=" * 60)
    r12 = run_ipr_gated(12, 8, 3.0, [3.5, 5.5, 5.5, 5.5], n_samples=80)

    print("\n\n")
    print("=" * 60)
    print("ROW 2: 18^3x12 rho=4.5")
    print("=" * 60)
    r18 = run_ipr_gated(18, 12, 4.5, [5.5, 8.5, 8.5, 8.5], n_samples=80)

    # Summary
    print(f"\n\n{'='*60}")
    print("CROSS-LATTICE COMPARISON")
    print(f"{'='*60}")

    for variant in ["P_min", "P_min+g"]:
        print(f"\n  Variant: {variant}")
        print(f"  {'Row':<20} {'n_rem':>6} {'dlz/eps':>10} {'err':>8} {'dlz_full':>10} {'err':>8}")
        print(f"  {'-'*54}")
        print(f"  {'8x8 (exact-5)':<20} {'5':>6} {-23.23:>10.2f} {2.30:>8.2f} {-1.973:>10.4f} {0.207:>8.4f}")

        d12 = r12[variant]
        d18 = r18[variant]
        print(f"  {'12x8':<20} {d12['n_removed']:>6} {d12['dlz_eps']:>10.2f} {d12['dlz_eps_err']:>8.2f} "
              f"{d12['dlz_full']:>10.4f} {d12['dlz_full_err']:>8.4f}")
        print(f"  {'18x12':<20} {d18['n_removed']:>6} {d18['dlz_eps']:>10.2f} {d18['dlz_eps_err']:>8.2f} "
              f"{d18['dlz_full']:>10.4f} {d18['dlz_full_err']:>8.4f}")

        drift = abs(d12['dlz_eps'] - d18['dlz_eps'])
        sig = np.sqrt(d12['dlz_eps_err']**2 + d18['dlz_eps_err']**2)
        print(f"  Drift 12x8 vs 18x12: {drift:.2f}, sigma={sig:.2f}, tension={drift/sig:.1f}")

    print(f"\n  REFERENCE (plain exact-prime, 5 modes only):")
    print(f"  12x8:  dlz/eps = -21.87 +/- 4.45")
    print(f"  18x12: dlz/eps = -62.44 +/- 6.02  (UNSTABLE)")
