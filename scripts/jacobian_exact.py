#!/usr/bin/env python3
"""
jacobian_exact.py
=================
Exact moduli Gram matrix Jacobian for the HS caloron branch response.

Computes:
    J_exact = (1/2) delta_xi log det G
            = epsilon * Tr[(G^(0))^{-1} H]

from background-gauge cleaned collective-coordinate tangent modes.

Collective coordinates (SU(2), k=1, trivial holonomy):
    q^I in {x0^0, x0^1, x0^2, x0^3, rho, Omega^1, Omega^2, Omega^3}
    Total: 8 moduli.

Requires the pfmt-instanton repo modules:
    src.caloron   : init_caloron
    src.lattice   : su2_log_algebra, su2_exp_algebra, su2_multiply, su2_dagger,
                    shift, lattice_dims, lattice_volume, SIGMA, IDENTITY
    src.spectral  : build_adjoint_laplacian, adjoint_matrix
    src.observables : full_diagnostic

Author: PFMT programme (Claude-assisted implementation)
Date: April 2026
Status: First implementation — requires line-by-line audit.
"""

import numpy as np
from scipy.sparse.linalg import cg as conjugate_gradient
from scipy.sparse import eye as sparse_eye
import sys
import os

# ── repo imports ──────────────────────────────────────────────────────
# Adjust path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from src.caloron import init_caloron
    from src.lattice import (su2_log_algebra, su2_exp_algebra,
                             su2_multiply, su2_dagger,
                             shift, lattice_dims, lattice_volume,
                             SIGMA, IDENTITY)
    from src.spectral import build_adjoint_laplacian, adjoint_matrix
    from src.observables import full_diagnostic
    HAVE_REPO = True
except ImportError:
    print("WARNING: repo modules not found. Standalone helper mode.")
    HAVE_REPO = False


# =====================================================================
# §1. ALGEBRA EXTRACTION: link -> Lie algebra potential
# =====================================================================

def link_to_algebra_site(U_link):
    """
    Extract algebra coefficients from a single SU(2) link.

    Given U = exp(i omega_a sigma_a / 2), returns omega = (omega_1, omega_2, omega_3).

    Uses the SU(2) parametrisation:
        U = a_0 I + i a_k sigma_k
        a_0 = cos(theta/2),  (a_1, a_2, a_3) = sin(theta/2) * n_hat
        omega_a = theta * n_a = theta * a_a / sin(theta/2)

    Matrix layout (matching repo convention):
        U[0,0] = a_0 + i a_3
        U[0,1] = a_2 + i a_1
        U[1,0] = -a_2 + i a_1
        U[1,1] = a_0 - i a_3

    Parameters
    ----------
    U_link : ndarray, shape (2, 2), complex
        Single SU(2) link matrix.

    Returns
    -------
    omega : ndarray, shape (3,), real
        Algebra coefficients: U = exp(i omega_a sigma_a / 2).
    """
    a0 = U_link[0, 0].real
    a1 = U_link[0, 1].imag
    a2 = U_link[0, 1].real
    a3 = U_link[0, 0].imag

    cos_half = np.clip(a0, -1.0, 1.0)
    theta = 2.0 * np.arccos(cos_half)
    sin_half = np.sin(theta / 2.0)

    if abs(sin_half) > 1e-12:
        factor = theta / sin_half
    else:
        # Taylor: theta / sin(theta/2) -> 2 + theta^2/12 + ...
        factor = 2.0

    return np.array([a1 * factor, a2 * factor, a3 * factor])


def link_to_algebra_field(U):
    """
    Extract the full algebra-valued gauge potential from link variables.

    A_mu^a(x) defined by U_mu(x) = exp(i A_mu^a sigma_a / 2).

    Parameters
    ----------
    U : ndarray, shape (N0, N1, N2, N3, 4, 2, 2), complex
        Link configuration.

    Returns
    -------
    A : ndarray, shape (N0, N1, N2, N3, 4, 3), real
        Algebra-valued gauge potential.
    """
    dims = U.shape[:4]
    A = np.zeros((*dims, 4, 3))

    # Vectorized extraction using the SU(2) parametrisation
    # U[..., 0, 0] = a_0 + i a_3
    # U[..., 0, 1] = a_2 + i a_1
    for mu in range(4):
        u00 = U[..., mu, 0, 0]  # shape (*dims)
        u01 = U[..., mu, 0, 1]

        a0 = u00.real
        a1 = u01.imag
        a2 = u01.real
        a3 = u00.imag

        cos_half = np.clip(a0, -1.0, 1.0)
        theta = 2.0 * np.arccos(cos_half)
        sin_half = np.sin(theta / 2.0)

        # Safe division with Taylor limit
        safe = np.abs(sin_half) > 1e-12
        factor = np.where(safe, theta / np.where(safe, sin_half, 1.0), 2.0)

        A[..., mu, 0] = a1 * factor
        A[..., mu, 1] = a2 * factor
        A[..., mu, 2] = a3 * factor

    return A


# =====================================================================
# §2. RAW TANGENT GENERATORS
# =====================================================================

def raw_tangent_translation(L_s, L_t, rho, x0, mu_dir, delta,
                            self_dual=True, n_sub=16, eps_FD=1e-5):
    """
    Raw tangent for translation modulus x_0^{mu_dir}.

    Y_{mu}^a(x) = [A_mu^a(x; x0 + delta e_{mu_dir}) - A_mu^a(x; x0 - delta e_{mu_dir})] / (2 delta)

    Parameters
    ----------
    L_s, L_t : int
        Spatial, temporal extents.
    rho : float
        Instanton size parameter.
    x0 : array-like, length 4
        Caloron center (t, x, y, z).
    mu_dir : int, 0..3
        Translation direction.
    delta : float
        Finite-difference step in lattice units.
    self_dual : bool
        SD (True) or ASD (False).
    n_sub, eps_FD : int, float
        Caloron discretisation parameters.

    Returns
    -------
    Y : ndarray, shape (N0, N1, N2, N3, 4, 3)
        Raw (un-cleaned) tangent field.
    """
    x0 = np.array(x0, dtype=float)

    x0_plus = x0.copy()
    x0_plus[mu_dir] += delta
    x0_minus = x0.copy()
    x0_minus[mu_dir] -= delta

    U_plus = init_caloron(L_s, L_t, rho=rho, x0=list(x0_plus),
                          self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)
    U_minus = init_caloron(L_s, L_t, rho=rho, x0=list(x0_minus),
                           self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)

    A_plus = link_to_algebra_field(U_plus)
    A_minus = link_to_algebra_field(U_minus)

    Y = (A_plus - A_minus) / (2.0 * delta)
    return Y


def raw_tangent_scale(L_s, L_t, rho, x0, delta_rho,
                      self_dual=True, n_sub=16, eps_FD=1e-5):
    """
    Raw tangent for scale modulus rho.

    Y_{mu}^a(x) = [A_mu^a(x; rho + delta_rho) - A_mu^a(x; rho - delta_rho)] / (2 delta_rho)

    Parameters
    ----------
    delta_rho : float
        Finite-difference step for rho.

    Returns
    -------
    Y : ndarray, shape (N0, N1, N2, N3, 4, 3)
    """
    x0 = list(x0)

    U_plus = init_caloron(L_s, L_t, rho=rho + delta_rho, x0=x0,
                          self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)
    U_minus = init_caloron(L_s, L_t, rho=rho - delta_rho, x0=x0,
                           self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)

    A_plus = link_to_algebra_field(U_plus)
    A_minus = link_to_algebra_field(U_minus)

    Y = (A_plus - A_minus) / (2.0 * delta_rho)
    return Y


def apply_global_gauge_rotation(U, g):
    """
    Apply a global SU(2) gauge transformation: U_mu(x) -> g U_mu(x) g^dagger.

    Parameters
    ----------
    U : ndarray, shape (N0, N1, N2, N3, 4, 2, 2)
    g : ndarray, shape (2, 2)

    Returns
    -------
    U_rot : same shape as U
    """
    g_dag = g.conj().T
    U_rot = np.zeros_like(U)
    for mu in range(4):
        # U_rot[..., mu, :, :] = g @ U[..., mu, :, :] @ g_dag
        # Vectorized over spatial indices
        U_mu = U[..., mu, :, :]  # shape (*dims, 2, 2)
        U_rot[..., mu, :, :] = np.einsum('ij,...jk,kl->...il', g, U_mu, g_dag)
    return U_rot


def raw_tangent_orientation(U_ref, a_dir, delta_theta):
    """
    Raw tangent for gauge orientation modulus Omega^{a_dir}.

    Applies global gauge rotations g_pm = exp(+/- i delta_theta sigma_a / 2)
    and finite-differences the algebra-valued potential.

    Parameters
    ----------
    U_ref : ndarray, shape (N0, N1, N2, N3, 4, 2, 2)
        Reference caloron configuration.
    a_dir : int, 0..2
        Color direction (0=sigma_1, 1=sigma_2, 2=sigma_3).
    delta_theta : float
        Finite-difference rotation angle.

    Returns
    -------
    Y : ndarray, shape (N0, N1, N2, N3, 4, 3)
    """
    # Build rotation matrices
    # sigma_a for a_dir in {0,1,2}
    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),       # sigma_1
        np.array([[0, -1j], [1j, 0]], dtype=complex),    # sigma_2
        np.array([[1, 0], [0, -1]], dtype=complex),      # sigma_3
    ]

    # g_+ = exp(+i delta_theta sigma_a / 2)
    # For SU(2): exp(i theta n.sigma/2) = cos(theta/2) I + i sin(theta/2) n.sigma
    # Here n = e_a (unit vector along color direction a_dir)
    ct = np.cos(delta_theta / 2.0)
    st = np.sin(delta_theta / 2.0)
    I2 = np.eye(2, dtype=complex)

    g_plus = ct * I2 + 1j * st * sigma[a_dir]
    g_minus = ct * I2 - 1j * st * sigma[a_dir]

    U_plus = apply_global_gauge_rotation(U_ref, g_plus)
    U_minus = apply_global_gauge_rotation(U_ref, g_minus)

    A_plus = link_to_algebra_field(U_plus)
    A_minus = link_to_algebra_field(U_minus)

    Y = (A_plus - A_minus) / (2.0 * delta_theta)
    return Y


# =====================================================================
# §3. COVARIANT DIVERGENCE AND GAUGE CLEANING
# =====================================================================

def compute_adjoint_transport(U, mu):
    """
    Compute the 3x3 adjoint representation matrix Ad(U_mu(x)) at every site.

    Parameters
    ----------
    U : ndarray, shape (N0, N1, N2, N3, 4, 2, 2)
    mu : int, 0..3

    Returns
    -------
    Ad_U : ndarray, shape (N0, N1, N2, N3, 3, 3)
        Ad(U_mu(x))^{ab} = (1/2) Tr(sigma^a U_mu(x) sigma^b U_mu(x)^dag)
    """
    dims = U.shape[:4]
    Ad_U = np.zeros((*dims, 3, 3))

    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]

    U_mu = U[..., mu, :, :]        # (*dims, 2, 2)
    U_mu_dag = U_mu.conj()
    # U_mu_dag[..., i, j] = conj(U_mu[..., j, i])
    U_mu_dag = np.swapaxes(U_mu_dag, -2, -1)

    for a in range(3):
        for b in range(3):
            # Ad^{ab} = (1/2) Tr(sigma^a U sigma^b U^dag)
            # Compute sigma^b U^dag first, then U (sigma^b U^dag), then sigma^a (U sigma^b U^dag)
            # Actually: Tr(sigma^a [U sigma^b U^dag])
            # Step 1: M = U sigma^b U^dag
            # Vectorized: M[..., i, j] = sum_k sum_l U[..., i, k] sigma^b[k, l] U_dag[..., l, j]
            sig_b = sigma[b]
            # U sigma^b: shape (*dims, 2, 2)
            U_sig = np.einsum('...ij,jk->...ik', U_mu, sig_b)
            # (U sigma^b) U^dag
            M = np.einsum('...ij,...kj->...ik', U_sig, U_mu.conj())
            # Tr(sigma^a M) = sum_i sigma^a[i,j] M[j,i] = sum_ij sigma^a[i,j] M[j,i]
            sig_a = sigma[a]
            tr = np.einsum('ij,...ji->', sig_a, M)
            Ad_U[..., a, b] = 0.5 * tr.real

    return Ad_U


def covariant_divergence(U, Y):
    """
    Lattice backward covariant divergence of a vector-valued adjoint field.

    (D . Y)^a(x) = sum_mu [ Y_mu^a(x) - Ad(U_mu(x - mu_hat)^dag)^{ab} Y_mu^b(x - mu_hat) ]

    This is the standard lattice backward divergence in the adjoint representation.
    It is the adjoint of the forward covariant derivative.

    Parameters
    ----------
    U : ndarray, shape (N0, N1, N2, N3, 4, 2, 2)
    Y : ndarray, shape (N0, N1, N2, N3, 4, 3)
        Vector-valued adjoint field (4 Lorentz components, 3 color components).

    Returns
    -------
    div_Y : ndarray, shape (N0, N1, N2, N3, 3)
        Scalar-valued adjoint field.

    Note
    ----
    Convention: the backward divergence uses Ad(U(x-mu)^dag).
    For orthogonal Ad, Ad(U^dag) = Ad(U)^T = Ad(U)^{-1}.
    So Ad(U(x-mu)^dag)^{ab} = Ad(U(x-mu))^{ba}.
    """
    dims = U.shape[:4]
    div_Y = np.zeros((*dims, 3))

    for mu in range(4):
        Y_mu = Y[..., mu, :]  # (*dims, 3)

        # Transport: Ad(U_mu(x - mu_hat))^T applied to Y_mu(x - mu_hat)
        # First shift Y_mu backward: Y_mu(x - mu_hat)
        Y_mu_shifted = np.roll(Y_mu, +1, axis=mu)  # Y_mu at x - mu_hat

        # Ad(U_mu(x - mu_hat)): need U_mu at x - mu_hat
        U_mu_shifted = np.roll(U[..., mu, :, :], +1, axis=mu)  # U_mu at x - mu_hat

        # Compute Ad(U_mu(x-mu_hat))^T . Y_mu(x-mu_hat)
        # Ad(U^dag)^{ab} = Ad(U)^{ba}, so the transport is sum_b Ad(U)^{ba} Y^b
        # = (Ad(U)^T . Y)^a = sum_b Ad(U)^{ab}_T Y^b ... 
        # Let me be more careful.
        #
        # The backward covariant derivative of a scalar phi is:
        #   (nabla_mu^* phi)^a(x) = phi^a(x) - Ad(U_mu(x-mu))^{ab} phi^b(x-mu)
        #
        # Wait — need to check the sign/direction convention against the
        # existing adjoint Laplacian in spectral.py.
        #
        # From the continuation document:
        #   (-D^2 phi)(x) = sum_mu [2 phi(x) - Ad(U_mu(x)) phi(x+mu) - Ad(U_mu(x-mu)^dag) phi(x-mu)]
        #
        # Since Ad(U^dag) = Ad(U)^{-1} = Ad(U)^T for orthogonal Ad:
        #   (-D^2 phi)(x) = sum_mu [2 phi(x) - Ad(U_mu(x)) phi(x+mu) - Ad(U_mu(x-mu))^T phi(x-mu)]
        #
        # The forward derivative is:
        #   (nabla_mu phi)(x) = Ad(U_mu(x)) phi(x+mu) - phi(x)
        # The backward derivative is:
        #   (nabla_mu^* phi)(x) = phi(x) - Ad(U_mu(x-mu))^T phi(x-mu)
        #
        # Then -D^2 = sum_mu nabla_mu^* nabla_mu = sum_mu (nabla_mu^* forward).
        # Check: nabla_mu^* nabla_mu phi(x)
        #   = nabla_mu^* [Ad(U(x)) phi(x+mu) - phi(x)]
        #   = [Ad(U(x)) phi(x+mu) - phi(x)] - Ad(U(x-mu))^T [Ad(U(x-mu)) phi(x) - phi(x-mu)]
        #   Hmm, that doesn't immediately simplify. Let me just use:
        #
        # The divergence is the negative adjoint of the gradient.
        # For the background-gauge condition D_mu Z_mu = 0, the
        # lattice transcription is the backward divergence:
        #
        #   (D . Y)(x) = sum_mu [Y_mu(x) - Ad(U_mu(x-mu))^T Y_mu(x-mu)]
        #
        # This is exactly the lattice transcription of partial_mu A_mu + [A_mu, Y_mu]
        # in the continuum, discretised as a backward difference.

        # Compute Ad(U_mu(x-mu_hat))
        # I'll compute this explicitly rather than using the full compute_adjoint_transport
        # for efficiency with the shifted links.
        Ad_shifted = _adjoint_3x3_vectorized(U_mu_shifted)  # (*dims, 3, 3)

        # Transport: Ad(U)^T . Y_shifted = sum_b Ad(U)^{ba} Y^b
        transported = np.einsum('...ba,...b->...a', Ad_shifted, Y_mu_shifted)

        div_Y += Y_mu - transported

    return div_Y


def _adjoint_3x3_vectorized(U_2x2):
    """
    Compute 3x3 adjoint representation of SU(2) matrices, vectorized.

    Uses the Rodrigues formula:
        Ad(U)^{ab} = (2 a_0^2 - 1) delta_{ab} + 2 a_a a_b + 2 a_0 eps_{abc} a_c

    where U = a_0 I + i (a_1 sigma_1 + a_2 sigma_2 + a_3 sigma_3).

    Color indices run 0,1,2 corresponding to sigma_1, sigma_2, sigma_3.

    Parameters
    ----------
    U_2x2 : ndarray, shape (..., 2, 2), complex

    Returns
    -------
    Ad : ndarray, shape (..., 3, 3), real
    """
    # Extract quaternion components from the SU(2) matrix layout:
    #   U[0,0] = a_0 + i a_3,   U[0,1] = a_2 + i a_1
    a0 = U_2x2[..., 0, 0].real
    a1 = U_2x2[..., 0, 1].imag
    a2 = U_2x2[..., 0, 1].real
    a3 = U_2x2[..., 0, 0].imag

    aa = np.stack([a1, a2, a3], axis=-1)  # (..., 3)

    # Symmetric part: (2 a_0^2 - 1) delta_{ab} + 2 a_a a_b
    c1 = 2.0 * a0**2 - 1.0
    shape = U_2x2.shape[:-2]
    Ad = np.zeros((*shape, 3, 3))
    for i in range(3):
        Ad[..., i, i] = c1
    Ad += 2.0 * np.einsum('...a,...b->...ab', aa, aa)

    # Antisymmetric part: 2 a_0 eps_{abc} a_c
    # eps_{abc} with a,b,c in {0,1,2}: eps_{012} = +1 (standard).
    # Ad^{01} += 2 a_0 * eps_{012} * aa[2] = +2 a_0 * aa[2]
    # Ad^{02} += 2 a_0 * eps_{021} * aa[1] = -2 a_0 * aa[1]
    # Ad^{10} += 2 a_0 * eps_{102} * aa[2] = -2 a_0 * aa[2]
    # Ad^{12} += 2 a_0 * eps_{120} * aa[0] = +2 a_0 * aa[0]
    # Ad^{20} += 2 a_0 * eps_{201} * aa[1] = +2 a_0 * aa[1]
    # Ad^{21} += 2 a_0 * eps_{210} * aa[0] = -2 a_0 * aa[0]
    Ad[..., 0, 1] += 2.0 * a0 * aa[..., 2]
    Ad[..., 0, 2] += -2.0 * a0 * aa[..., 1]
    Ad[..., 1, 0] += -2.0 * a0 * aa[..., 2]
    Ad[..., 1, 2] += 2.0 * a0 * aa[..., 0]
    Ad[..., 2, 0] += 2.0 * a0 * aa[..., 1]
    Ad[..., 2, 1] += -2.0 * a0 * aa[..., 0]

    return Ad


def gauge_clean_tangent(U, Y, mass_sq=1e-6, cg_tol=1e-10, cg_maxiter=5000):
    """
    Background-gauge clean a raw tangent field.

    Solves  (-D^2 + m^2) omega = - D . Y
    then computes  Z = Y - (nabla omega)

    where (nabla_mu omega)(x) = Ad(U_mu(x)) omega(x+mu) - omega(x)
    is the forward covariant derivative.

    Derivation: D.Z = D.Y + (-D^2)omega = 0  =>  (-D^2)omega = -D.Y.

    Parameters
    ----------
    U : ndarray, shape (N0, N1, N2, N3, 4, 2, 2)
        Background gauge field.
    Y : ndarray, shape (N0, N1, N2, N3, 4, 3)
        Raw tangent field.
    mass_sq : float
        Small IR regulator for the Laplacian (prevents exact zero mode of -D^2).
    cg_tol : float
        CG convergence tolerance.
    cg_maxiter : int
        Maximum CG iterations.

    Returns
    -------
    Z : ndarray, shape (N0, N1, N2, N3, 4, 3)
        Gauge-cleaned tangent field satisfying D . Z ≈ 0.
    omega : ndarray, shape (N0, N1, N2, N3, 3)
        Gauge-cleaning scalar (for diagnostics).
    info : dict
        Convergence information.
    """
    dims = U.shape[:4]
    V = int(np.prod(dims))

    # Step 1: compute D . Y
    div_Y = covariant_divergence(U, Y)  # (*dims, 3)

    # Step 2: build the adjoint Laplacian (-D^2 + m^2)
    # Using the repo function, which returns a sparse CSR matrix of dim 3V x 3V
    Lap = build_adjoint_laplacian(U, mass_sq=mass_sq)

    # Step 3: flatten div_Y to vector and solve
    # SIGN: the gauge-cleaning equation is
    #   D.Z = D.Y + Delta omega = 0
    #   => Delta omega = -D.Y
    #   => (Delta + m^2) omega = -D.Y
    # so the RHS carries a MINUS sign.
    rhs = -div_Y.reshape(3 * V)

    # Solve (Delta + m^2) omega = -div_Y
    omega_flat, cg_info = conjugate_gradient(Lap, rhs, rtol=cg_tol, maxiter=cg_maxiter,
                                              atol=0)
    converged = (cg_info == 0)

    omega = omega_flat.reshape(*dims, 3)

    # Compute residual
    residual = Lap @ omega_flat - rhs
    res_norm = np.linalg.norm(residual) / (np.linalg.norm(rhs) + 1e-30)

    # Step 4: compute D omega (forward covariant gradient of scalar)
    # (D_mu omega)^a(x) = Ad(U_mu(x))^{ab} omega^b(x + mu_hat) - omega^a(x)
    D_omega = np.zeros((*dims, 4, 3))
    for mu in range(4):
        omega_shifted = np.roll(omega, -1, axis=mu)  # omega at x + mu_hat
        U_mu = U[..., mu, :, :]
        Ad_U = _adjoint_3x3_vectorized(U_mu)  # (*dims, 3, 3)
        transported = np.einsum('...ab,...b->...a', Ad_U, omega_shifted)
        D_omega[..., mu, :] = transported - omega

    # Step 5: Z = Y - D omega
    Z = Y - D_omega

    # Diagnostic: check D . Z ≈ 0
    div_Z = covariant_divergence(U, Z)
    div_Z_norm = np.sqrt(np.sum(div_Z**2))
    div_Y_norm = np.sqrt(np.sum(div_Y**2))

    info = {
        'cg_converged': converged,
        'cg_info': cg_info,
        'residual_relative': res_norm,
        'div_Y_norm': div_Y_norm,
        'div_Z_norm': div_Z_norm,
        'cleaning_ratio': div_Z_norm / (div_Y_norm + 1e-30),
    }

    return Z, omega, info


# =====================================================================
# §4. GRAM MATRIX AND JACOBIAN
# =====================================================================

def inner_product_isotropic(Z1, Z2):
    """
    Isotropic inner product of two gauge-field tangent vectors.

    <Z1, Z2> = sum_x sum_mu sum_a Z1_{mu}^a(x) Z2_{mu}^a(x)

    Parameters
    ----------
    Z1, Z2 : ndarray, shape (N0, N1, N2, N3, 4, 3)

    Returns
    -------
    float
    """
    return np.sum(Z1 * Z2)


def temporal_overlap(Z1, Z2):
    """
    Temporal-component overlap of two gauge-field tangent vectors.

    H = sum_x sum_a Z1_{0}^a(x) Z2_{0}^a(x)

    Only the mu=0 (temporal) component contributes.

    Parameters
    ----------
    Z1, Z2 : ndarray, shape (N0, N1, N2, N3, 4, 3)

    Returns
    -------
    float
    """
    return np.sum(Z1[..., 0, :] * Z2[..., 0, :])


def build_gram_and_H(tangents):
    """
    Build the isotropic Gram matrix G^{(0)} and temporal overlap matrix H
    from a list of gauge-cleaned tangent fields.

    G^{(0)}_{IJ} = <Z_I, Z_J>
    H_{IJ} = sum_x Z_{I,0}(x) . Z_{J,0}(x)

    Parameters
    ----------
    tangents : list of ndarray
        Each element has shape (N0, N1, N2, N3, 4, 3).

    Returns
    -------
    G0 : ndarray, shape (n_mod, n_mod)
        Isotropic Gram matrix.
    H : ndarray, shape (n_mod, n_mod)
        Temporal overlap matrix.
    """
    n_mod = len(tangents)
    G0 = np.zeros((n_mod, n_mod))
    H = np.zeros((n_mod, n_mod))

    for I in range(n_mod):
        for J in range(I, n_mod):
            g_ij = inner_product_isotropic(tangents[I], tangents[J])
            h_ij = temporal_overlap(tangents[I], tangents[J])
            G0[I, J] = g_ij
            G0[J, I] = g_ij
            H[I, J] = h_ij
            H[J, I] = h_ij

    return G0, H


def compute_exact_jacobian(G0, H, epsilon):
    """
    Compute the exact first-order moduli Jacobian.

    J_exact = (1/2) delta_xi log det G = epsilon * Tr[(G^(0))^{-1} H]

    Parameters
    ----------
    G0 : ndarray, shape (n_mod, n_mod)
        Isotropic Gram matrix.
    H : ndarray, shape (n_mod, n_mod)
        Temporal overlap matrix.
    epsilon : float
        Disformal anisotropy parameter (xi - 1).

    Returns
    -------
    J_exact : float
        The Jacobian contribution to delta_xi log zeta.
    trace_GinvH : float
        The basis-invariant trace Tr[(G^(0))^{-1} H].
    info : dict
        Diagnostic information.
    """
    # Condition number check
    eigvals_G = np.linalg.eigvalsh(G0)
    cond_G = eigvals_G[-1] / (eigvals_G[0] + 1e-30)

    # Solve G^{(0)} X = H  =>  X = (G^(0))^{-1} H
    # Then Tr[X] = Tr[(G^(0))^{-1} H]
    G_inv_H = np.linalg.solve(G0, H)
    trace_GinvH = np.trace(G_inv_H)

    # The Jacobian term
    J_exact = epsilon * trace_GinvH

    # Cross-check: if G0 were the identity, this would be epsilon * Tr(H) = epsilon * sum_I f_0^(I)
    # where f_0^(I) = H_{II} / G0_{II}
    f0_diagonal = np.array([H[I, I] / G0[I, I] for I in range(len(G0))])
    naive_sum = epsilon * np.sum(f0_diagonal)

    info = {
        'trace_GinvH': trace_GinvH,
        'G0_eigenvalues': eigvals_G,
        'G0_condition': cond_G,
        'G0_diagonal': np.diag(G0),
        'H_diagonal': np.diag(H),
        'f0_diagonal': f0_diagonal,
        'naive_diagonal_sum': naive_sum,
        'off_diagonal_correction': J_exact - naive_sum,
    }

    return J_exact, trace_GinvH, info


# =====================================================================
# §5. CROSS-CHECKS
# =====================================================================

def check_translation_vs_fieldstrength(U, Z_trans, mu_dir):
    """
    Continuum cross-check: for translation tangent in direction nu,
    the background-gauge cleaned tangent should satisfy

        Z^{(nu)}_mu  proportional to  F_{mu,nu}

    Check the correlation coefficient.

    Parameters
    ----------
    U : ndarray, shape (N0, N1, N2, N3, 4, 2, 2)
    Z_trans : ndarray, shape (N0, N1, N2, N3, 4, 3)
        Gauge-cleaned translation tangent for direction mu_dir.
    mu_dir : int
        The translation direction.

    Returns
    -------
    corr : float
        Correlation coefficient between Z and F_{.,mu_dir}.
    """
    from src.observables import field_strength

    dims = U.shape[:4]
    # Build F_{mu, mu_dir} for each mu
    F_ref = np.zeros((*dims, 4, 3))
    for mu in range(4):
        if mu == mu_dir:
            continue
        # F_mu_nu^a = Tr(sigma^a F_mu_nu) where F is the clover
        F_munu = field_strength(U, mu, mu_dir)  # (*dims, 2, 2)
        # Extract adjoint components
        sigma = [
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        ]
        for a in range(3):
            # F^a = Tr(sigma^a F) — but F from observables is already
            # F = (C - C^dag)/(8i), which is traceless Hermitian.
            # In adjoint: F^a = 2 Tr(T^a F) = Tr(sigma^a F)
            F_ref[..., mu, a] = np.einsum('ij,...ji->', sigma[a], F_munu).real

    # Compute correlation
    z_flat = Z_trans.flatten()
    f_flat = F_ref.flatten()

    z_norm = np.linalg.norm(z_flat)
    f_norm = np.linalg.norm(f_flat)

    if z_norm < 1e-15 or f_norm < 1e-15:
        return 0.0

    corr = np.dot(z_flat, f_flat) / (z_norm * f_norm)
    return corr


def check_low_mode_overlap(Z_I, U, threshold=0.06):
    """
    Check whether the collective tangent Z_I is captured by the
    spectrally-thresholded low-mode subspace.

    Computes |P_low Z_I|^2 / |Z_I|^2 where P_low projects onto
    the near-zero modes of the vector operator Delta_1.

    Parameters
    ----------
    Z_I : ndarray, shape (N0, N1, N2, N3, 4, 3)
    U : ndarray
    threshold : float

    Returns
    -------
    overlap : float
        Should be close to 1 if spectral proxy was valid.
    """
    from src.spectral import build_vector_operator
    from scipy.sparse.linalg import eigsh

    dims = U.shape[:4]
    V = int(np.prod(dims))

    # Build vector operator
    Delta1 = build_vector_operator(U, mass_sq=1e-4)

    # Find low-lying eigenmodes
    n_low = 20  # find enough to cover the near-zero sector
    vals, vecs = eigsh(Delta1, k=n_low, which='SM')

    # Select modes below threshold
    mask = np.abs(vals) < threshold
    low_vecs = vecs[:, mask]  # (12V, n_low_selected)

    # Flatten Z_I to 12V vector
    z_flat = Z_I.reshape(12 * V)

    # Project: P_low z = sum_i (v_i . z) v_i
    if low_vecs.shape[1] == 0:
        return 0.0

    coeffs = low_vecs.T @ z_flat  # (n_selected,)
    z_proj = low_vecs @ coeffs

    overlap = np.dot(z_proj, z_proj) / (np.dot(z_flat, z_flat) + 1e-30)
    return overlap


# =====================================================================
# §6. FINITE-DIFFERENCE CONVERGENCE CHECK
# =====================================================================

def check_fd_convergence(L_s, L_t, rho, x0, modulus_type, modulus_index,
                         deltas, **kwargs):
    """
    Check that the raw tangent has converged in the FD step size.

    Computes Y for multiple delta values and checks that |Y(delta) - Y(delta/2)|
    decreases as O(delta^2) (symmetric FD).

    Parameters
    ----------
    modulus_type : str
        'translation', 'scale', or 'orientation'
    modulus_index : int
        Direction/color index.
    deltas : list of float
        Step sizes to test.

    Returns
    -------
    norms : list of float
        |Y(delta)| for each delta.
    diffs : list of float
        |Y(delta) - Y(delta_smallest)| for each delta.
    """
    x0 = list(x0)
    Ys = []

    for delta in deltas:
        if modulus_type == 'translation':
            Y = raw_tangent_translation(L_s, L_t, rho, x0, modulus_index,
                                        delta, **kwargs)
        elif modulus_type == 'scale':
            Y = raw_tangent_scale(L_s, L_t, rho, x0, delta, **kwargs)
        elif modulus_type == 'orientation':
            U_ref = init_caloron(L_s, L_t, rho=rho, x0=x0, **kwargs)
            Y = raw_tangent_orientation(U_ref, modulus_index, delta)
        else:
            raise ValueError(f"Unknown modulus_type: {modulus_type}")
        Ys.append(Y)

    norms = [np.sqrt(np.sum(Y**2)) for Y in Ys]

    # Differences relative to finest (last)
    Y_finest = Ys[-1]
    diffs = [np.sqrt(np.sum((Y - Y_finest)**2)) for Y in Ys]

    return norms, diffs


# =====================================================================
# §7. MASTER DRIVER
# =====================================================================

def run_exact_jacobian(L_s, L_t, rho, x0=None, epsilon=0.09,
                       delta_trans=0.1, delta_rho=0.05, delta_theta=0.05,
                       self_dual=True, n_sub=16, eps_FD=1e-5,
                       mass_sq_clean=1e-6, cg_tol=1e-10, cg_maxiter=5000,
                       verbose=True):
    """
    Master driver: compute the exact moduli Gram matrix Jacobian.

    Returns the full result and all diagnostics.

    Parameters
    ----------
    L_s, L_t : int
        Lattice extents.
    rho : float
        Caloron size.
    x0 : list or None
        Center (default: geometric center).
    epsilon : float
        Disformal anisotropy (xi - 1).
    delta_trans, delta_rho, delta_theta : float
        Finite-difference step sizes.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    result : dict with keys:
        'J_exact': float — the Jacobian term (1/2 delta_xi log det G)
        'trace_GinvH': float — basis-invariant trace
        'G0': ndarray (8,8) — isotropic Gram matrix
        'H': ndarray (8,8) — temporal overlap matrix
        'tangents': list of 8 tangent fields
        'cleaning_info': list of 8 dicts
        'jacobian_info': dict
    """
    if x0 is None:
        x0 = [L_t / 2.0, L_s / 2.0, L_s / 2.0, L_s / 2.0]

    if verbose:
        print("=" * 70)
        print(f"EXACT JACOBIAN COMPUTATION")
        print(f"Lattice: {L_s}^3 x {L_t}, rho = {rho}")
        print(f"Center: {x0}")
        print(f"epsilon = {epsilon}")
        print(f"FD steps: trans={delta_trans}, rho={delta_rho}, theta={delta_theta}")
        print("=" * 70)

    # ── Generate reference caloron ────────────────────────────────────
    if verbose:
        print("\n[1/7] Generating reference caloron...")
    U_ref = init_caloron(L_s, L_t, rho=rho, x0=list(x0),
                         self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)
    if verbose:
        full_diagnostic(U_ref, f"Reference {L_s}^3x{L_t}")

    # ── Generate raw tangents ─────────────────────────────────────────
    if verbose:
        print("\n[2/7] Computing raw tangents (8 moduli, 16 caloron calls)...")

    raw_tangents = []
    labels = []

    # Translations: q^0 = x0^0, q^1 = x0^1, q^2 = x0^2, q^3 = x0^3
    for mu_dir in range(4):
        if verbose:
            print(f"  Translation mu={mu_dir}...", end=" ", flush=True)
        Y = raw_tangent_translation(L_s, L_t, rho, x0, mu_dir, delta_trans,
                                    self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)
        raw_tangents.append(Y)
        labels.append(f"trans_{mu_dir}")
        if verbose:
            print(f"|Y| = {np.sqrt(np.sum(Y**2)):.6f}")

    # Scale: q^4 = rho
    if verbose:
        print(f"  Scale rho...", end=" ", flush=True)
    Y = raw_tangent_scale(L_s, L_t, rho, x0, delta_rho,
                          self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)
    raw_tangents.append(Y)
    labels.append("scale")
    if verbose:
        print(f"|Y| = {np.sqrt(np.sum(Y**2)):.6f}")

    # Orientations: q^5 = Omega^0, q^6 = Omega^1, q^7 = Omega^2
    for a_dir in range(3):
        if verbose:
            print(f"  Orientation a={a_dir}...", end=" ", flush=True)
        Y = raw_tangent_orientation(U_ref, a_dir, delta_theta)
        raw_tangents.append(Y)
        labels.append(f"orient_{a_dir}")
        if verbose:
            print(f"|Y| = {np.sqrt(np.sum(Y**2)):.6f}")

    # ── Gauge cleaning ────────────────────────────────────────────────
    if verbose:
        print(f"\n[3/7] Gauge-cleaning tangents (8 CG solves)...")

    cleaned_tangents = []
    cleaning_info = []

    for I in range(8):
        if verbose:
            print(f"  Cleaning {labels[I]}...", end=" ", flush=True)
        Z, omega, info = gauge_clean_tangent(U_ref, raw_tangents[I],
                                             mass_sq=mass_sq_clean,
                                             cg_tol=cg_tol,
                                             cg_maxiter=cg_maxiter)
        cleaned_tangents.append(Z)
        cleaning_info.append(info)
        if verbose:
            status = "OK" if info['cg_converged'] else "FAIL"
            print(f"CG {status}, |D.Y| = {info['div_Y_norm']:.4e}, "
                  f"|D.Z| = {info['div_Z_norm']:.4e}, "
                  f"ratio = {info['cleaning_ratio']:.4e}")

    # ── Build Gram matrix and temporal overlap ────────────────────────
    if verbose:
        print(f"\n[4/7] Building Gram matrix G^(0) and temporal overlap H...")

    G0, H = build_gram_and_H(cleaned_tangents)

    if verbose:
        print(f"\n  G^(0) diagonal: {np.diag(G0)}")
        print(f"  H diagonal:     {np.diag(H)}")
        print(f"  f_0 diagonal:   {np.diag(H) / np.diag(G0)}")
        print(f"  G^(0) condition number: {np.linalg.cond(G0):.4e}")

    # ── Compute exact Jacobian ────────────────────────────────────────
    if verbose:
        print(f"\n[5/7] Computing exact Jacobian...")

    J_exact, trace_GinvH, jac_info = compute_exact_jacobian(G0, H, epsilon)

    if verbose:
        print(f"\n  Tr[(G^(0))^{{-1}} H] = {trace_GinvH:.6f}")
        print(f"  J_exact = epsilon * Tr[G^-1 H] = {epsilon} * {trace_GinvH:.6f} = {J_exact:.6f}")
        print(f"  Naive diagonal estimate = {jac_info['naive_diagonal_sum']:.6f}")
        print(f"  Off-diagonal correction = {jac_info['off_diagonal_correction']:.6f}")

    # ── Translation vs field-strength cross-check ─────────────────────
    if verbose:
        print(f"\n[6/7] Cross-check: translation tangent vs F_{{mu,nu}}...")
    try:
        for nu in range(4):
            corr = check_translation_vs_fieldstrength(U_ref, cleaned_tangents[nu], nu)
            if verbose:
                print(f"  Z_trans[{nu}] vs F_{{.,{nu}}}: correlation = {corr:.4f}")
    except Exception as e:
        if verbose:
            print(f"  Cross-check skipped: {e}")

    # ── Final summary ─────────────────────────────────────────────────
    if verbose:
        print(f"\n[7/7] Final result")
        print("=" * 70)
        print(f"  delta_xi log zeta_partial  = -2.23 +/- 0.11  [BRICK]")
        print(f"  J_exact (this computation) = {J_exact:+.4f}")
        print(f"  delta_xi log zeta_full     = {-2.23 + J_exact:+.4f}")
        print(f"  zeta(X*)/zeta(0)           = {np.exp(-2.23 + J_exact):.4f}")
        print(f"  1 - zeta(X*)/zeta(0)       = {1.0 - np.exp(-2.23 + J_exact):.4f}")
        print("=" * 70)

    result = {
        'J_exact': J_exact,
        'trace_GinvH': trace_GinvH,
        'G0': G0,
        'H': H,
        'labels': labels,
        'tangents': cleaned_tangents,
        'raw_tangents': raw_tangents,
        'cleaning_info': cleaning_info,
        'jacobian_info': jac_info,
        'lattice': (L_s, L_t, rho),
        'epsilon': epsilon,
        'fd_steps': (delta_trans, delta_rho, delta_theta),
    }

    return result


# =====================================================================
# §8. ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    print("PFMT Lattice Instanton Programme — Exact Jacobian Computation")
    print("=" * 70)

    # ── Primary: best near-Bogomolny row ──────────────────────────────
    print("\n>>> PRIMARY: 16^3 x 8, rho = 3.0")
    result_16 = run_exact_jacobian(
        L_s=16, L_t=8, rho=3.0,
        epsilon=0.09,
        delta_trans=0.1,
        delta_rho=0.05,
        delta_theta=0.05,
        verbose=True,
    )

    # ── Cross-check: secondary row ────────────────────────────────────
    print("\n\n>>> CROSS-CHECK: 12^3 x 8, rho = 3.0")
    result_12 = run_exact_jacobian(
        L_s=12, L_t=8, rho=3.0,
        epsilon=0.09,
        delta_trans=0.1,
        delta_rho=0.05,
        delta_theta=0.05,
        verbose=True,
    )

    # ── Comparison ────────────────────────────────────────────────────
    print("\n\n>>> COMPARISON")
    print(f"  16^3x8: Tr[G^-1 H] = {result_16['trace_GinvH']:.4f}, "
          f"J = {result_16['J_exact']:+.4f}")
    print(f"  12^3x8: Tr[G^-1 H] = {result_12['trace_GinvH']:.4f}, "
          f"J = {result_12['J_exact']:+.4f}")
    diff = abs(result_16['trace_GinvH'] - result_12['trace_GinvH'])
    avg = 0.5 * (result_16['trace_GinvH'] + result_12['trace_GinvH'])
    print(f"  Relative difference: {diff / (abs(avg) + 1e-30):.4f}")
