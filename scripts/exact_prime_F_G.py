#!/usr/bin/env python3
"""
exact_prime_F_G.py
==================
Complete the exact-prime computation: F (delta_F insertion) and G (ghost).
On 8^3x8 to match the T computation from exact_prime_closure.py.
"""
import sys, time
import numpy as np
from scipy.sparse.linalg import cg as conjugate_gradient
from scipy import sparse
sys.path.insert(0, '.')

from src.caloron import init_caloron
from src.observables import field_strength
from src.lattice import SIGMA
from src.spectral import (build_adjoint_laplacian, build_vector_operator,
                          field_strength_adjoint)
from jacobian_exact import (
    _adjoint_3x3_vectorized, covariant_divergence, gauge_clean_tangent,
)
from jacobian_left_triv import (
    raw_tangent_translation_LT, raw_tangent_scale_LT,
)


def build_delta_F_operator(U, dims):
    """
    Build the 2*delta_F operator for the anisotropic deformation.

    delta_F has the same structure as the 2F coupling in Delta_1,
    but with temporal planes flipped in sign:
      temporal planes (0i): coefficient +1 (sign flip from original -1)
      spatial planes (ij):  coefficient -1 (keep original)

    So: (2 delta_F)_{mu,a; nu,b} = sign(mu,nu) * (-2 eps^{abc} F_{mu,nu}^c)
    where sign = +1 for temporal planes, -1 for spatial planes.
    """
    V = int(np.prod(dims))
    N = 12 * V

    F_adj = field_strength_adjoint(U)  # (*dims, 6, 3)
    F_flat = F_adj.reshape(V, 6, 3)

    planes = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    plane_idx = {}
    for p, (mu, nu) in enumerate(planes):
        plane_idx[(mu, nu)] = (p, +1)
        plane_idx[(nu, mu)] = (p, -1)

    # Temporal vs spatial classification
    def is_temporal_plane(mu, nu):
        return (mu == 0) or (nu == 0)

    eps = np.zeros((3, 3, 3))
    eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = +1.0
    eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

    rows, cols, vals = [], [], []

    for x in range(V):
        for mu in range(4):
            for nu in range(4):
                if mu == nu:
                    continue
                p, sign = plane_idx[(mu, nu)]
                Fc = sign * F_flat[x, p, :]

                # Anisotropic sign: +1 for temporal, -1 for spatial
                aniso_sign = +1.0 if is_temporal_plane(mu, nu) else -1.0

                for a in range(3):
                    for b in range(3):
                        val = 0.0
                        for c in range(3):
                            val += -2.0 * eps[a, b, c] * Fc[c]
                        val *= aniso_sign  # apply the deformation sign
                        if abs(val) > 1e-15:
                            v_row = 12*x + 3*mu + a
                            v_col = 12*x + 3*nu + b
                            rows.append(v_row)
                            cols.append(v_col)
                            vals.append(val)

    M = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N))
    return M.tocsr()


def orthonormalize(tangents, dims):
    V = int(np.prod(dims))
    vecs = [t.reshape(12*V) for t in tangents]
    ortho = []
    for v in vecs:
        w = v.copy()
        for u in ortho:
            w -= np.dot(u, w) * u
        n = np.linalg.norm(w)
        if n > 1e-10:
            ortho.append(w / n)
    return ortho


def main():
    L_s, L_t, rho = 8, 8, 3.0
    x0 = [3.5, 3.5, 3.5, 3.5]
    mass_sq = 0.01
    n_samples = 80

    print("="*60)
    print("EXACT-PRIME: F AND G TERMS (8^3 x 8)")
    print("="*60)

    # Reference caloron
    U = init_caloron(L_s, L_t, rho=rho, x0=x0, self_dual=True, n_sub=16)
    dims = U.shape[:4]
    V = int(np.prod(dims))

    # Free field
    U_free = np.zeros_like(U)
    for mu in range(4):
        U_free[..., mu, :, :] = np.eye(2, dtype=complex)

    # Build 5 physical LT tangents
    print("\n[1] Building physical tangents...")
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
        print(f"  Mode {I}: clean ratio = {info['cleaning_ratio']:.4e}")

    ortho_12V = orthonormalize(cleaned, dims)
    print(f"  Orthonormal basis: {len(ortho_12V)} vectors")

    # Free zero modes (12 constant modes)
    free_zeros = []
    for mu in range(4):
        for a in range(3):
            v = np.zeros(12*V)
            for x in range(V):
                v[12*x + 3*mu + a] = 1.0 / np.sqrt(V)
            free_zeros.append(v)

    # Build operators
    print("\n[2] Building operators...")
    Delta1_inst = build_vector_operator(U, mass_sq=mass_sq)
    Delta1_free = build_vector_operator(U_free, mass_sq=mass_sq)

    # D_0^2 in 12V space
    D0sq_ghost = build_adjoint_laplacian(U, mass_sq=0.0, temporal_only=True)
    D0sq_free_ghost = build_adjoint_laplacian(U_free, mass_sq=0.0, temporal_only=True)

    perm = np.zeros(12*V, dtype=int)
    for x in range(V):
        for mu in range(4):
            for a in range(3):
                perm[12*x + 3*mu + a] = 3*V*mu + 3*x + a

    D0sq_12V = sparse.block_diag([D0sq_ghost]*4, format='csr')[perm][:, perm]
    D0sq_free_12V = sparse.block_diag([D0sq_free_ghost]*4, format='csr')[perm][:, perm]

    # 2*delta_F operator
    print("  Building 2*delta_F operator...")
    dF_inst = build_delta_F_operator(U, dims)
    dF_free = build_delta_F_operator(U_free, dims)
    print(f"  2dF(inst): nnz = {dF_inst.nnz}")
    print(f"  2dF(free): nnz = {dF_free.nnz}")

    # Ghost operators (3V)
    Ghost_inst = build_adjoint_laplacian(U, mass_sq=mass_sq)
    Ghost_free = build_adjoint_laplacian(U_free, mass_sq=mass_sq)
    D0sq_ghost_inst = build_adjoint_laplacian(U, mass_sq=0.0, temporal_only=True)
    D0sq_ghost_free = build_adjoint_laplacian(U_free, mass_sq=0.0, temporal_only=True)

    # Ghost zero modes (3 constant adjoint scalars)
    ghost_free_zeros = []
    for a in range(3):
        v = np.zeros(3*V)
        for x in range(V):
            v[3*x + a] = 1.0 / np.sqrt(V)
        ghost_free_zeros.append(v)

    # ================================================================
    # F TERM: Tr'[Delta_1^{-1} (2 delta_F)]  inst - free
    # ================================================================
    print(f"\n[3] Computing F (delta_F insertion) with exact 5-mode prime...")
    print(f"  {n_samples} Z2 noise samples, correlated...")

    rng = np.random.default_rng(123)
    F_corr_samples = []
    for s in range(n_samples):
        eta = rng.choice([-1.0, 1.0], size=12*V)

        # Inst
        Bi = dF_inst @ eta
        xi, _ = conjugate_gradient(Delta1_inst, Bi, rtol=1e-8, maxiter=3000, atol=0)
        vi = np.dot(eta, xi)

        # Free
        Bf = dF_free @ eta
        xf, _ = conjugate_gradient(Delta1_free, Bf, rtol=1e-8, maxiter=3000, atol=0)
        vf = np.dot(eta, xf)

        F_corr_samples.append(vi - vf)

    F_corr_mean = np.mean(F_corr_samples)
    F_corr_err = np.std(F_corr_samples) / np.sqrt(n_samples)

    # 5-mode inst correction for F
    F_inst_corr = 0.0
    for v in ortho_12V:
        Bv = dF_inst @ v
        x, _ = conjugate_gradient(Delta1_inst, Bv, rtol=1e-10, maxiter=5000, atol=0)
        F_inst_corr += np.dot(v, x)

    # 12-mode free correction for F
    F_free_corr = 0.0
    for v in free_zeros:
        Bv = dF_free @ v
        x, _ = conjugate_gradient(Delta1_free, Bv, rtol=1e-10, maxiter=5000, atol=0)
        F_free_corr += np.dot(v, x)

    F_exact = F_corr_mean - F_inst_corr + F_free_corr

    print(f"  Correlated inst-free: {F_corr_mean:.2f} +/- {F_corr_err:.2f}")
    print(f"  Inst 5-mode correction: {F_inst_corr:.4f}")
    print(f"  Free 12-mode correction: {F_free_corr:.4f}")
    print(f"  F (exact 5-mode prime) = {F_exact:.2f} +/- {F_corr_err:.2f}")
    print(f"  Previous threshold-prime F = -3.8 +/- 0.9 (8^3x8)")

    # ================================================================
    # G TERM: Tr[Ghost^{-1} D_0^2]  inst - free (no prime needed)
    # ================================================================
    print(f"\n[4] Computing G (ghost) with correlated subtraction...")
    print(f"  {n_samples} Z2 noise samples...")

    rng2 = np.random.default_rng(456)
    G_corr_samples = []
    for s in range(n_samples):
        eta = rng2.choice([-1.0, 1.0], size=3*V)

        # Inst
        Bi = D0sq_ghost_inst @ eta
        xi, _ = conjugate_gradient(Ghost_inst, Bi, rtol=1e-8, maxiter=3000, atol=0)
        vi = np.dot(eta, xi)

        # Free
        Bf = D0sq_ghost_free @ eta
        xf, _ = conjugate_gradient(Ghost_free, Bf, rtol=1e-8, maxiter=3000, atol=0)
        vf = np.dot(eta, xf)

        G_corr_samples.append(vi - vf)

    G_corr_mean = np.mean(G_corr_samples)
    G_corr_err = np.std(G_corr_samples) / np.sqrt(n_samples)

    # Ghost has no physical zero modes on instanton (irreducible on T^4).
    # Free ghost has 3 zero modes (constant), but D_0^2 annihilates them.
    # So no prime corrections needed for G.

    print(f"  G (correlated inst-free) = {G_corr_mean:.2f} +/- {G_corr_err:.2f}")
    print(f"  Previous G = 0.6 +/- 0.7 (8^3x8)")

    # ================================================================
    # ASSEMBLE
    # ================================================================
    T_exact = 21.53  # from previous run
    T_err = 3.05

    print(f"\n{'='*60}")
    print("FULL EXACT-PRIME ASSEMBLY (8^3 x 8)")
    print(f"{'='*60}")
    print(f"  T  = {T_exact:.2f} +/- {T_err:.2f}")
    print(f"  F  = {F_exact:.2f} +/- {F_corr_err:.2f}")
    print(f"  G  = {G_corr_mean:.2f} +/- {G_corr_err:.2f}")
    print(f"  J  = +0.118 (physical 5-mode)")

    dlz_eps = -(T_exact + 0.5*F_exact) + 2*G_corr_mean
    dlz_eps_err = np.sqrt(T_err**2 + (0.5*F_corr_err)**2 + (2*G_corr_err)**2)

    epsilon = 0.09
    dlz = epsilon * dlz_eps + 0.118
    dlz_err = epsilon * dlz_eps_err

    print(f"\n  dlz/eps = -(T + F/2) + 2G = {dlz_eps:.2f} +/- {dlz_eps_err:.2f}")
    print(f"  dlz_partial = eps * dlz/eps = {epsilon*dlz_eps:.3f} +/- {epsilon*dlz_eps_err:.3f}")
    print(f"  dlz_full = dlz_partial + J = {dlz:.3f} +/- {dlz_err:.3f}")
    print(f"\n  zeta(X*)/zeta(0) = exp({dlz:.3f}) = {np.exp(dlz):.4f}")
    print(f"  1 - zeta(X*)/zeta(0) = {1 - np.exp(dlz):.4f}")

    print(f"\n  OLD (threshold-prime, 8^3x8):")
    old_dlz_eps = -(27.1 + 0.5*(-3.8)) + 2*0.6
    print(f"    dlz/eps = {old_dlz_eps:.1f}")
    print(f"    dlz_partial = {0.09*old_dlz_eps:.3f}")
    print(f"    dlz_full = {0.09*old_dlz_eps + 0.118:.3f}")


if __name__ == "__main__":
    main()
