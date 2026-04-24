#!/usr/bin/env python3
"""
verify_jacobian_infra.py
========================
Self-contained verification tests for jacobian_exact.py infrastructure.

Tests on analytically known configurations before running on the caloron.
Can be run WITHOUT the full pfmt-instanton repo.

Tests:
  1. Adjoint representation: Ad(I) = I_3, Ad orthogonal, Ad(UV) = Ad(U)Ad(V)
  2. Algebra extraction: log(exp(omega)) = omega roundtrip
  3. Covariant divergence on free field: reduces to ordinary lattice divergence
  4. Gauge cleaning on free field: removes exact gradient
  5. Orientation tangent on free field: is a pure gauge mode
"""

import numpy as np
import sys

# ── Inline minimal SU(2) helpers (no repo needed) ────────────────────

SIGMA = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]
IDENTITY = np.eye(2, dtype=complex)


def su2_exp(omega):
    """exp(i omega_a sigma_a / 2)"""
    theta = np.linalg.norm(omega)
    if theta < 1e-15:
        return IDENTITY.copy()
    n = omega / theta
    return np.cos(theta / 2) * IDENTITY + 1j * np.sin(theta / 2) * sum(n[a] * SIGMA[a] for a in range(3))


def cold_start(dims):
    """All links = identity."""
    U = np.zeros((*dims, 4, 2, 2), dtype=complex)
    for mu in range(4):
        U[..., mu, :, :] = IDENTITY
    return U


# ── Import from jacobian_exact ────────────────────────────────────────
sys.path.insert(0, '.')
from jacobian_exact import (
    link_to_algebra_site,
    link_to_algebra_field,
    _adjoint_3x3_vectorized,
    covariant_divergence,
)


def test_adjoint_identity():
    """Ad(I) should be the 3x3 identity."""
    U = IDENTITY.reshape(1, 2, 2)
    Ad = _adjoint_3x3_vectorized(U)
    err = np.linalg.norm(Ad[0] - np.eye(3))
    print(f"  Ad(I) = I_3: error = {err:.2e}", "PASS" if err < 1e-14 else "FAIL")
    return err < 1e-14


def test_adjoint_orthogonal():
    """Ad(U) should be orthogonal for any SU(2) U."""
    rng = np.random.default_rng(42)
    omega = rng.normal(size=3) * 0.5
    U = su2_exp(omega).reshape(1, 2, 2)
    Ad = _adjoint_3x3_vectorized(U)[0]
    err = np.linalg.norm(Ad @ Ad.T - np.eye(3))
    print(f"  Ad(U) orthogonal: error = {err:.2e}", "PASS" if err < 1e-13 else "FAIL")
    return err < 1e-13


def test_adjoint_product():
    """Ad(UV) = Ad(U) Ad(V)."""
    rng = np.random.default_rng(42)
    omega1 = rng.normal(size=3) * 0.5
    omega2 = rng.normal(size=3) * 0.3
    U1 = su2_exp(omega1)
    U2 = su2_exp(omega2)
    UV = U1 @ U2

    Ad1 = _adjoint_3x3_vectorized(U1.reshape(1, 2, 2))[0]
    Ad2 = _adjoint_3x3_vectorized(U2.reshape(1, 2, 2))[0]
    AdUV = _adjoint_3x3_vectorized(UV.reshape(1, 2, 2))[0]

    err = np.linalg.norm(AdUV - Ad1 @ Ad2)
    print(f"  Ad(UV) = Ad(U)Ad(V): error = {err:.2e}", "PASS" if err < 1e-13 else "FAIL")
    return err < 1e-13


def test_adjoint_action():
    """
    Cross-check: Ad(U)^{ab} v_b should match (1/2) Tr(sigma^a U (v.sigma) U^dag).
    """
    rng = np.random.default_rng(42)
    omega = rng.normal(size=3) * 0.7
    U = su2_exp(omega)
    v = rng.normal(size=3)

    # Method 1: Rodrigues
    Ad = _adjoint_3x3_vectorized(U.reshape(1, 2, 2))[0]
    w1 = Ad @ v

    # Method 2: explicit matrix
    V = sum(v[a] * SIGMA[a] for a in range(3))
    M = U @ V @ U.conj().T
    w2 = np.array([0.5 * np.trace(SIGMA[a] @ M).real for a in range(3)])

    err = np.linalg.norm(w1 - w2)
    print(f"  Ad action cross-check: error = {err:.2e}", "PASS" if err < 1e-13 else "FAIL")
    return err < 1e-13


def test_algebra_roundtrip():
    """su2_log(su2_exp(omega)) = omega for moderate omega."""
    rng = np.random.default_rng(42)
    for trial in range(5):
        omega_in = rng.normal(size=3) * 1.5  # not too large
        U = su2_exp(omega_in)
        omega_out = link_to_algebra_site(U)
        err = np.linalg.norm(omega_out - omega_in)
        if err > 1e-10:
            print(f"  Algebra roundtrip trial {trial}: error = {err:.2e} FAIL")
            return False
    print(f"  Algebra roundtrip (5 trials): max error < 1e-10 PASS")
    return True


def test_divergence_free_field():
    """
    On a free (cold) configuration, the covariant divergence reduces to
    the ordinary lattice backward difference. A constant vector field
    should have zero divergence.
    """
    dims = (4, 4, 4, 4)
    U = cold_start(dims)

    # Constant vector field: Y_mu^a(x) = delta_{mu,0} delta_{a,0}
    Y = np.zeros((*dims, 4, 3))
    Y[..., 0, 0] = 1.0  # constant in mu=0, a=0

    div_Y = covariant_divergence(U, Y)
    err = np.sqrt(np.sum(div_Y**2))
    print(f"  Divergence of constant field (free): |D.Y| = {err:.2e}",
          "PASS" if err < 1e-14 else "FAIL")
    return err < 1e-14


def test_divergence_gradient():
    """
    On a free field, D.(nabla phi) should equal -Delta phi
    (the negative Laplacian of phi). Test by constructing a known
    gradient field and checking its divergence.
    """
    dims = (4, 4, 4, 4)
    U = cold_start(dims)

    # Scalar field: phi^a(x) = sin(2*pi*x_0/N_0) * delta_{a,0}
    N0 = dims[0]
    phi = np.zeros((*dims, 3))
    x0_grid = np.arange(N0)
    phi_1d = np.sin(2 * np.pi * x0_grid / N0)
    phi[..., 0] = phi_1d[:, None, None, None]

    # Forward derivative: (nabla_mu phi)(x) = phi(x+mu) - phi(x) [free field, Ad=I]
    Y = np.zeros((*dims, 4, 3))
    for mu in range(4):
        phi_shifted = np.roll(phi, -1, axis=mu)
        Y[..., mu, :] = phi_shifted - phi

    # Divergence: D.Y = sum_mu [Y_mu(x) - Y_mu(x-mu)] [free field, Ad=I]
    div_Y = covariant_divergence(U, Y)

    # Expected: -Delta phi = sum_mu [2 phi(x) - phi(x+mu) - phi(x-mu)]
    neg_lap = np.zeros((*dims, 3))
    for mu in range(4):
        neg_lap += 2 * phi - np.roll(phi, -1, axis=mu) - np.roll(phi, +1, axis=mu)
    expected = -neg_lap  # D.(nabla phi) = -(-D^2) phi = -(neg_lap)

    err = np.sqrt(np.sum((div_Y - expected)**2))
    norm = np.sqrt(np.sum(expected**2))
    rel_err = err / (norm + 1e-30)
    print(f"  D.(nabla phi) = -Delta phi: relative error = {rel_err:.2e}",
          "PASS" if rel_err < 1e-13 else "FAIL")
    return rel_err < 1e-13


def test_algebra_field_cold():
    """On a cold (identity) configuration, A_mu should be zero everywhere."""
    dims = (4, 4, 4, 4)
    U = cold_start(dims)
    A = link_to_algebra_field(U)
    err = np.max(np.abs(A))
    print(f"  Algebra of cold start: max|A| = {err:.2e}",
          "PASS" if err < 1e-14 else "FAIL")
    return err < 1e-14


def test_algebra_field_uniform():
    """
    Uniform non-trivial links: all U_mu(x) = exp(i omega sigma_3 / 2) for some omega.
    A_mu should be constant omega everywhere.
    """
    dims = (4, 4, 4, 4)
    omega_val = 0.3
    U = cold_start(dims)
    U_link = su2_exp(np.array([0, 0, omega_val]))
    for mu in range(4):
        U[..., mu, :, :] = U_link

    A = link_to_algebra_field(U)

    # Expected: A_mu^2(x) = omega_val for all x, mu; A_mu^{0,1} = 0
    expected = np.zeros((*dims, 4, 3))
    expected[..., 2] = omega_val

    err = np.max(np.abs(A - expected))
    print(f"  Algebra of uniform field: max error = {err:.2e}",
          "PASS" if err < 1e-12 else "FAIL")
    return err < 1e-12


# =====================================================================
# RUN ALL TESTS
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Jacobian Infrastructure Verification Tests")
    print("=" * 60)

    tests = [
        ("Adjoint: Ad(I) = I_3", test_adjoint_identity),
        ("Adjoint: orthogonality", test_adjoint_orthogonal),
        ("Adjoint: product rule", test_adjoint_product),
        ("Adjoint: action cross-check", test_adjoint_action),
        ("Algebra: roundtrip", test_algebra_roundtrip),
        ("Algebra: cold start", test_algebra_field_cold),
        ("Algebra: uniform field", test_algebra_field_uniform),
        ("Divergence: constant field", test_divergence_free_field),
        ("Divergence: gradient identity", test_divergence_gradient),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            passed = test_fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            passed = False
        results.append((name, passed))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for _, p in results if p)
    n_fail = sum(1 for _, p in results if not p)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print(f"\n  {n_pass}/{len(results)} passed, {n_fail} failed.")


def test_gauge_cleaning_free():
    """
    On a free field, create a tangent Y that is a pure gradient plus a
    transverse part. Gauge cleaning should remove the gradient.
    
    Y_mu = Z_mu + nabla_mu phi
    
    where Z is divergence-free. After cleaning, we should recover Z.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import cg as conjugate_gradient
    
    dims = (4, 4, 4, 4)
    U = cold_start(dims)
    V = int(np.prod(dims))
    
    # Divergence-free part: Z_mu^a(x) = sin(2pi x_1/N) * delta_{mu,2} * delta_{a,0}
    # This is a "magnetic-like" field in the (1,2) plane. Check it's transverse.
    N = dims[1]
    Z_true = np.zeros((*dims, 4, 3))
    x1_grid = np.arange(N)
    Z_true[:, :, :, :, 2, 0] = np.sin(2 * np.pi * x1_grid / N)[None, :, None, None]
    
    # Verify Z_true is divergence-free (on free field, D.Z = sum_mu dZ_mu/dx_mu)
    # Z only has mu=2 component, and it only depends on x_1, not x_2.
    # So d_2 Z_2 = Z_2(x_2+1) - Z_2(x_2) = sin(...) - sin(...) but Z doesn't depend on x_2.
    # Actually Z_2(x) = sin(2pi x_1/N) which is independent of x_2.
    # So nabla_2^* Z_2 = Z_2(x) - Z_2(x - e_2) = sin(2pi x_1/N) - sin(2pi x_1/N) = 0. Good.
    div_Z_true = covariant_divergence(U, Z_true)
    assert np.sqrt(np.sum(div_Z_true**2)) < 1e-14, "Z_true should be divergence-free"
    
    # Longitudinal part: nabla_mu phi for some phi
    phi = np.zeros((*dims, 3))
    phi[..., 1] = np.cos(2 * np.pi * np.arange(dims[0]) / dims[0])[:, None, None, None] * 0.5
    
    grad_phi = np.zeros((*dims, 4, 3))
    for mu in range(4):
        phi_shifted = np.roll(phi, -1, axis=mu)
        grad_phi[..., mu, :] = phi_shifted - phi  # forward derivative on free field
    
    # Full tangent: Y = Z + grad_phi
    Y = Z_true + grad_phi
    
    # Gauge clean Y
    # On free field, -D^2 = ordinary lattice Laplacian
    # Build it manually for free field
    # (-D^2 phi)(x) = sum_mu [2 phi(x) - phi(x+mu) - phi(x-mu)]
    # Dimension: 3V
    
    # Instead of using build_adjoint_laplacian (needs repo), build manually
    from scipy.sparse import lil_matrix
    mass_sq = 1e-8
    dim_total = 3 * V
    Lap = lil_matrix((dim_total, dim_total))
    
    # For free field (Ad = I), the Laplacian is block-diagonal in color
    # and is just the scalar lattice Laplacian tensored with I_3.
    # Index map: v = 3 * x_flat + a
    for a_col in range(3):
        for x_flat in range(V):
            idx = 3 * x_flat + a_col
            Lap[idx, idx] += 2 * 4 + mass_sq  # diagonal: 2 * n_dim + m^2
            
            # Neighbors in each direction
            x_multi = np.unravel_index(x_flat, dims)
            for mu in range(4):
                # Forward neighbor
                x_fwd = list(x_multi)
                x_fwd[mu] = (x_fwd[mu] + 1) % dims[mu]
                idx_fwd = 3 * np.ravel_multi_index(x_fwd, dims) + a_col
                Lap[idx, idx_fwd] -= 1.0
                
                # Backward neighbor
                x_bwd = list(x_multi)
                x_bwd[mu] = (x_bwd[mu] - 1) % dims[mu]
                idx_bwd = 3 * np.ravel_multi_index(x_bwd, dims) + a_col
                Lap[idx, idx_bwd] -= 1.0
    
    Lap = Lap.tocsr()
    
    # Compute D.Y
    div_Y = covariant_divergence(U, Y)
    rhs = -div_Y.reshape(dim_total)  # CORRECT SIGN: Delta omega = -D.Y
    
    omega_flat, info = conjugate_gradient(Lap, rhs, rtol=1e-12, maxiter=3000, atol=0)
    omega = omega_flat.reshape(*dims, 3)
    
    # Compute nabla omega
    D_omega = np.zeros((*dims, 4, 3))
    for mu in range(4):
        omega_shifted = np.roll(omega, -1, axis=mu)
        D_omega[..., mu, :] = omega_shifted - omega  # free field: Ad = I
    
    # Cleaned tangent
    Z_cleaned = Y - D_omega
    
    # Compare Z_cleaned to Z_true
    err = np.sqrt(np.sum((Z_cleaned - Z_true)**2))
    norm = np.sqrt(np.sum(Z_true**2))
    rel_err = err / norm
    
    print(f"  |Z_cleaned - Z_true| / |Z_true| = {rel_err:.2e}",
          "PASS" if rel_err < 1e-4 else "FAIL")
    
    # Also check divergence of cleaned field
    div_Z_cleaned = covariant_divergence(U, Z_cleaned)
    div_ratio = np.sqrt(np.sum(div_Z_cleaned**2)) / (np.sqrt(np.sum(div_Y**2)) + 1e-30)
    print(f"  |D.Z_cleaned| / |D.Y| = {div_ratio:.2e}",
          "PASS" if div_ratio < 1e-4 else "FAIL")
    
    return rel_err < 1e-4 and div_ratio < 1e-4


if __name__ == "__main__":
    print("\n[TEST] Gauge cleaning: free field recovery")
    try:
        passed = test_gauge_cleaning_free()
        print(f"  Overall: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  EXCEPTION: {e}")
