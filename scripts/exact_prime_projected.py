#!/usr/bin/env python3
"""
exact_prime_projected.py
========================
Theorem-grade projected-complement prime.

For operator A (= Delta_1 + m^2) and insertion B (= D_0^2 or 2*delta_F),
with physical collective subspace V (5 orthonormal columns):

    P = I - V V^T

The exact primed trace is:

    Tr'[A^{-1} B] = Tr[ G_P B ]

where G_P = P (P A P |_{P H})^{-1} P is the constrained Green operator.

To compute G_P B eta for a noise vector eta, solve the saddle system:

    | A   V | | x |     | P B eta |
    |       | |   |  =  |         |
    | V^T 0 | | l |     |    0    |

Then x = G_P B eta  (automatically satisfies V^T x = 0).

Stochastic trace: Tr'[A^{-1}B] = (1/N) sum_eta  eta^T x(eta)
where x solves the saddle system with RHS = (P B eta, 0).

Implementation: use the Schur complement / projected CG:
  - Project RHS: r = P B eta
  - Solve A x_0 = r  by CG
  - Project solution: x = P x_0  (first-order projected CG)
  
  This is NOT exact for the saddle system. The exact method is
  iterative projected CG:
  
  Projected CG: solve min_x ||Ax - r||  subject to  V^T x = 0.
  
  At each CG step, project the search direction and residual
  onto the complement of V. This is a standard technique
  (Golub & Ye, "An inverse free preconditioned Krylov subspace method").
"""
import sys, time
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
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
from exact_prime_F_G import build_delta_F_operator


# =====================================================================
# PROJECTED CG
# =====================================================================

def projected_cg(A, b, V_basis, rtol=1e-8, maxiter=3000):
    """
    Solve  min ||Ax - b||  subject to  V^T x = 0
    using CG in the projected subspace.

    A : sparse matrix or LinearOperator (n x n), symmetric positive on P H
    b : ndarray (n,) — assumed already projected: V^T b = 0
    V_basis : list of ndarray (n,) — orthonormal columns spanning V

    Returns
    -------
    x : ndarray (n,) — solution satisfying V^T x = 0
    info : int — 0 if converged
    """
    n = len(b)

    def project(v):
        """P v = v - V (V^T v)"""
        w = v.copy()
        for u in V_basis:
            w -= np.dot(u, v) * u
        return w

    def A_apply(v):
        if isinstance(A, LinearOperator):
            return A.matvec(v)
        else:
            return A @ v

    # Initial guess: x0 = 0 (satisfies constraint)
    x = np.zeros(n)
    r = project(b.copy())  # r = P(b - Ax) = Pb since x=0
    p = r.copy()
    rs_old = np.dot(r, r)

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-30:
        return x, 0

    for k in range(maxiter):
        Ap = project(A_apply(p))  # P A p (projected)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            break

        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) / b_norm < rtol:
            return project(x), 0  # final projection for safety

        beta = rs_new / rs_old
        p = r + beta * p
        p = project(p)  # keep search direction in complement
        rs_old = rs_new

    return project(x), 1  # did not converge


# =====================================================================
# EXACT PROJECTED TRACE
# =====================================================================

def projected_trace(A, B, V_basis, n_samples=80, rtol=1e-8,
                    maxiter=3000, rng=None):
    """
    Compute the exact projected-complement trace:

        Tr'[A^{-1} B] = Tr[ G_P B ]

    using projected CG and Hutchinson estimation.

    For each Z2 noise vector eta:
      1. Compute r = P B eta   (project out collective subspace)
      2. Solve P A P x = r  via projected CG (x in complement of V)
      3. Accumulate eta^T x

    Parameters
    ----------
    A : sparse matrix (n x n), the fluctuation operator
    B : sparse matrix (n x n), the insertion operator
    V_basis : list of ndarray (n,), orthonormal collective tangents
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = A.shape[0]

    def project(v):
        w = v.copy()
        for u in V_basis:
            w -= np.dot(u, v) * u
        return w

    samples = []
    n_fail = 0
    for s in range(n_samples):
        eta = rng.choice([-1.0, 1.0], size=n)

        # r = P B eta
        B_eta = B @ eta
        r = project(B_eta)

        # Solve P A P x = r  with V^T x = 0
        x, info = projected_cg(A, r, V_basis, rtol=rtol, maxiter=maxiter)
        if info != 0:
            n_fail += 1

        samples.append(np.dot(eta, x))

    trace = np.mean(samples)
    err = np.std(samples) / np.sqrt(n_samples)

    if n_fail > 0:
        print(f"  WARNING: {n_fail}/{n_samples} CG solves did not converge")

    return trace, err


# =====================================================================
# MAIN
# =====================================================================

def main():
    L_s, L_t, rho = 8, 8, 3.0
    x0 = [3.5, 3.5, 3.5, 3.5]
    mass_sq = 0.01
    n_samples = 80
    epsilon = 0.09

    print("=" * 60)
    print("THEOREM-GRADE PROJECTED-COMPLEMENT PRIME (8^3 x 8)")
    print("=" * 60)

    # Reference caloron
    U = init_caloron(L_s, L_t, rho=rho, x0=x0, self_dual=True, n_sub=16)
    dims = U.shape[:4]
    V = int(np.prod(dims))

    # Free field
    U_free = np.zeros_like(U)
    for mu in range(4):
        U_free[..., mu, :, :] = np.eye(2, dtype=complex)

    # Build 5 physical LT tangents
    print("\n[1] Building and cleaning 5 physical tangents...")
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

    # Orthonormalize in 12V space
    vecs = [t.reshape(12 * V) for t in cleaned]
    ortho = []
    for v in vecs:
        w = v.copy()
        for u in ortho:
            w -= np.dot(u, w) * u
        n = np.linalg.norm(w)
        if n > 1e-10:
            ortho.append(w / n)
    print(f"  Orthonormal basis: {len(ortho)} vectors in R^{12*V}")

    # Free zero modes (12 constant)
    free_zeros = []
    for mu in range(4):
        for a in range(3):
            v = np.zeros(12 * V)
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

    perm = np.zeros(12 * V, dtype=int)
    for x in range(V):
        for mu in range(4):
            for a in range(3):
                perm[12*x + 3*mu + a] = 3*V*mu + 3*x + a

    D0sq_12V = sparse.block_diag([D0sq_ghost]*4, format='csr')[perm][:, perm]
    D0sq_free_12V = sparse.block_diag([D0sq_free_ghost]*4, format='csr')[perm][:, perm]

    # 2*delta_F
    dF_inst = build_delta_F_operator(U, dims)
    dF_free = build_delta_F_operator(U_free, dims)

    # Ghost operators (3V)
    Ghost_inst = build_adjoint_laplacian(U, mass_sq=mass_sq)
    Ghost_free = build_adjoint_laplacian(U_free, mass_sq=mass_sq)
    D0sq_ghost_inst = build_adjoint_laplacian(U, mass_sq=0.0, temporal_only=True)
    D0sq_ghost_free = build_adjoint_laplacian(U_free, mass_sq=0.0, temporal_only=True)

    ghost_free_zeros = []
    for a in range(3):
        v = np.zeros(3 * V)
        for x in range(V):
            v[3*x + a] = 1.0 / np.sqrt(V)
        ghost_free_zeros.append(v)

    # ================================================================
    # T: PROJECTED-COMPLEMENT Tr'[Delta_1^{-1} D_0^2]
    # ================================================================
    print(f"\n[3] T: projected CG with 5-mode complement ({n_samples} samples)...")
    t0 = time.time()

    # INSTANTON: projected trace with 5 physical modes removed
    rng1 = np.random.default_rng(42)
    T_inst, T_inst_err = projected_trace(
        Delta1_inst, D0sq_12V, ortho,
        n_samples=n_samples, rtol=1e-8, maxiter=3000, rng=rng1)
    print(f"  T_inst (projected) = {T_inst:.2f} +/- {T_inst_err:.2f}")

    # FREE: projected trace with 12 constant modes removed
    rng2 = np.random.default_rng(42)
    T_free, T_free_err = projected_trace(
        Delta1_free, D0sq_free_12V, free_zeros,
        n_samples=n_samples, rtol=1e-8, maxiter=3000, rng=rng2)
    print(f"  T_free (projected) = {T_free:.2f} +/- {T_free_err:.2f}")

    # CORRELATED: same noise, projected on each side
    rng3 = np.random.default_rng(42)
    T_corr_samples = []
    for s in range(n_samples):
        eta = rng3.choice([-1.0, 1.0], size=12*V)

        # Inst: project, solve projected system
        r_i = D0sq_12V @ eta
        for u in ortho:
            r_i -= np.dot(u, r_i) * u
        x_i, _ = projected_cg(Delta1_inst, r_i, ortho, rtol=1e-8, maxiter=3000)

        # Free: project, solve projected system
        r_f = D0sq_free_12V @ eta
        for u in free_zeros:
            r_f -= np.dot(u, r_f) * u
        x_f, _ = projected_cg(Delta1_free, r_f, free_zeros, rtol=1e-8, maxiter=3000)

        T_corr_samples.append(np.dot(eta, x_i) - np.dot(eta, x_f))

    T_corr = np.mean(T_corr_samples)
    T_corr_err = np.std(T_corr_samples) / np.sqrt(n_samples)
    print(f"  T (correlated projected inst-free) = {T_corr:.2f} +/- {T_corr_err:.2f}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # ================================================================
    # F: PROJECTED-COMPLEMENT Tr'[Delta_1^{-1} 2*delta_F]
    # ================================================================
    print(f"\n[4] F: projected CG with 5-mode complement ({n_samples} samples)...")
    t0 = time.time()

    rng4 = np.random.default_rng(123)
    F_corr_samples = []
    for s in range(n_samples):
        eta = rng4.choice([-1.0, 1.0], size=12*V)

        # Inst
        r_i = dF_inst @ eta
        for u in ortho:
            r_i -= np.dot(u, r_i) * u
        x_i, _ = projected_cg(Delta1_inst, r_i, ortho, rtol=1e-8, maxiter=3000)

        # Free (dF_free has nnz=0, so B eta = 0)
        x_f = np.zeros(12*V)

        F_corr_samples.append(np.dot(eta, x_i) - np.dot(eta, x_f))

    F_corr = np.mean(F_corr_samples)
    F_corr_err = np.std(F_corr_samples) / np.sqrt(n_samples)
    print(f"  F (correlated projected inst-free) = {F_corr:.2f} +/- {F_corr_err:.2f}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # ================================================================
    # G: GHOST (no prime needed, same as before)
    # ================================================================
    print(f"\n[5] G: ghost correlated subtraction ({n_samples} samples)...")
    t0 = time.time()

    rng5 = np.random.default_rng(456)
    G_samples = []
    for s in range(n_samples):
        eta = rng5.choice([-1.0, 1.0], size=3*V)

        Bi = D0sq_ghost_inst @ eta
        xi, _ = projected_cg(Ghost_inst, Bi, [], rtol=1e-8, maxiter=3000)

        Bf = D0sq_ghost_free @ eta
        xf, _ = projected_cg(Ghost_free, Bf, ghost_free_zeros, rtol=1e-8, maxiter=3000)

        G_samples.append(np.dot(eta, xi) - np.dot(eta, xf))

    G_corr = np.mean(G_samples)
    G_corr_err = np.std(G_samples) / np.sqrt(n_samples)
    print(f"  G (correlated inst-free) = {G_corr:.2f} +/- {G_corr_err:.2f}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # ================================================================
    # ASSEMBLE
    # ================================================================
    print(f"\n{'='*60}")
    print("PROJECTED-COMPLEMENT EXACT-PRIME ASSEMBLY (8^3 x 8)")
    print(f"{'='*60}")
    print(f"  T  = {T_corr:.2f} +/- {T_corr_err:.2f}")
    print(f"  F  = {F_corr:.2f} +/- {F_corr_err:.2f}")
    print(f"  G  = {G_corr:.2f} +/- {G_corr_err:.2f}")
    print(f"  J  = +0.118 (physical 5-mode LT)")

    dlz_eps = -(T_corr + 0.5*F_corr) + 2*G_corr
    dlz_eps_err = np.sqrt(T_corr_err**2 + (0.5*F_corr_err)**2 + (2*G_corr_err)**2)

    dlz = epsilon * dlz_eps + 0.118
    dlz_err = epsilon * dlz_eps_err

    print(f"\n  dlz/eps = -(T + F/2) + 2G = {dlz_eps:.2f} +/- {dlz_eps_err:.2f}")
    print(f"  dlz_partial = eps * dlz/eps = {epsilon*dlz_eps:.4f} +/- {epsilon*dlz_eps_err:.4f}")
    print(f"  dlz_full = dlz_partial + J = {dlz:.4f} +/- {dlz_err:.4f}")
    print(f"\n  zeta(X*)/zeta(0) = exp({dlz:.4f}) = {np.exp(dlz):.4f}")
    print(f"  1 - zeta(X*)/zeta(0) = {1 - np.exp(dlz):.4f}")

    print(f"\n  COMPARISON TABLE:")
    print(f"  {'Scheme':<30} {'dlz/eps':>10} {'dlz_full':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Threshold-prime (old)':<30} {'~-24.0':>10} {'~-2.04':>10}")
    print(f"  {'Surrogate LT-5 subtract':<30} {'-24.18':>10} {'-2.06':>10}")
    print(f"  {'Projected-complement (this)':<30} {dlz_eps:>10.2f} {dlz:>10.4f}")


if __name__ == "__main__":
    main()
