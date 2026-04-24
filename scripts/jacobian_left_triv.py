#!/usr/bin/env python3
"""
jacobian_left_triv.py
=====================
Rebuild the moduli Gram matrix in left-trivialized link variables,
matching the fluctuation convention of build_vector_operator / build_adjoint_laplacian.

Also: pure-gauge orientation test and low-mode overlap matrix.
"""
import sys, os
import numpy as np
from scipy.sparse.linalg import cg as conjugate_gradient, eigsh
sys.path.insert(0, '.')
from src.caloron import init_caloron
from src.lattice import SIGMA, IDENTITY
from src.observables import field_strength, full_diagnostic
from src.spectral import build_adjoint_laplacian, build_vector_operator
from jacobian_exact import (
    _adjoint_3x3_vectorized, covariant_divergence, gauge_clean_tangent,
    build_gram_and_H, compute_exact_jacobian, link_to_algebra_site
)

# =====================================================================
# §1. LEFT-TRIVIALIZED ALGEBRA EXTRACTION
# =====================================================================

def log_su2(M):
    """
    Extract algebra coefficients from SU(2) matrix M.
    M = exp(i omega_a sigma_a / 2)  =>  returns omega = (omega_1, omega_2, omega_3).
    
    Vectorized over leading dimensions.
    """
    a0 = M[..., 0, 0].real
    a1 = M[..., 0, 1].imag
    a2 = M[..., 0, 1].real
    a3 = M[..., 0, 0].imag
    
    cos_half = np.clip(a0, -1.0, 1.0)
    theta = 2.0 * np.arccos(cos_half)
    sin_half = np.sin(theta / 2.0)
    
    safe = np.abs(sin_half) > 1e-12
    factor = np.where(safe, theta / np.where(safe, sin_half, 1.0), 2.0)
    
    omega = np.stack([a1 * factor, a2 * factor, a3 * factor], axis=-1)
    return omega  # (..., 3)


def left_triv_tangent_field(U_shifted, U_ref):
    """
    Compute the left-trivialized algebra field:
        a_mu^a(x) = Log( U_shifted_mu(x) . U_ref_mu(x)^dag )
    
    This is the fluctuation variable used by the vector/ghost operators.
    
    Parameters
    ----------
    U_shifted, U_ref : (N0,N1,N2,N3,4,2,2) complex
    
    Returns
    -------
    a : (N0,N1,N2,N3,4,3) real
    """
    dims = U_ref.shape[:4]
    a = np.zeros((*dims, 4, 3))
    
    for mu in range(4):
        # M = U_shifted . U_ref^dag, per site
        Us = U_shifted[..., mu, :, :]    # (..., 2, 2)
        Ur_dag = U_ref[..., mu, :, :].conj()
        Ur_dag = np.swapaxes(Ur_dag, -2, -1)  # dagger
        
        M = np.einsum('...ij,...jk->...ik', Us, Ur_dag)
        a[..., mu, :] = log_su2(M)
    
    return a


# =====================================================================
# §2. RAW TANGENTS IN LEFT-TRIVIALIZED VARIABLES
# =====================================================================

def raw_tangent_translation_LT(U_ref, L_s, L_t, rho, x0, mu_dir, delta,
                                self_dual=True, n_sub=16, eps_FD=1e-5):
    """Left-trivialized translation tangent."""
    x0 = np.array(x0, dtype=float)
    x0p = x0.copy(); x0p[mu_dir] += delta
    x0m = x0.copy(); x0m[mu_dir] -= delta
    
    Up = init_caloron(L_s, L_t, rho=rho, x0=list(x0p),
                      self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)
    Um = init_caloron(L_s, L_t, rho=rho, x0=list(x0m),
                      self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)
    
    a_plus = left_triv_tangent_field(Up, U_ref)
    a_minus = left_triv_tangent_field(Um, U_ref)
    
    return (a_plus - a_minus) / (2.0 * delta)


def raw_tangent_scale_LT(U_ref, L_s, L_t, rho, x0, delta_rho,
                          self_dual=True, n_sub=16, eps_FD=1e-5):
    """Left-trivialized scale tangent."""
    Up = init_caloron(L_s, L_t, rho=rho+delta_rho, x0=list(x0),
                      self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)
    Um = init_caloron(L_s, L_t, rho=rho-delta_rho, x0=list(x0),
                      self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)
    
    a_plus = left_triv_tangent_field(Up, U_ref)
    a_minus = left_triv_tangent_field(Um, U_ref)
    
    return (a_plus - a_minus) / (2.0 * delta_rho)


def raw_tangent_orientation_LT(U_ref, a_dir, delta_theta):
    """Left-trivialized orientation tangent."""
    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    I2 = np.eye(2, dtype=complex)
    ct = np.cos(delta_theta / 2.0)
    st = np.sin(delta_theta / 2.0)
    
    g_plus  = ct * I2 + 1j * st * sigma[a_dir]
    g_minus = ct * I2 - 1j * st * sigma[a_dir]
    g_plus_dag  = g_plus.conj().T
    g_minus_dag = g_minus.conj().T
    
    # U_rotated = g U_ref g^dag
    dims = U_ref.shape[:4]
    Up = np.zeros_like(U_ref)
    Um = np.zeros_like(U_ref)
    for mu in range(4):
        Umu = U_ref[..., mu, :, :]
        Up[..., mu, :, :] = np.einsum('ij,...jk,kl->...il', g_plus, Umu, g_plus_dag)
        Um[..., mu, :, :] = np.einsum('ij,...jk,kl->...il', g_minus, Umu, g_minus_dag)
    
    a_plus = left_triv_tangent_field(Up, U_ref)
    a_minus = left_triv_tangent_field(Um, U_ref)
    
    return (a_plus - a_minus) / (2.0 * delta_theta)


# =====================================================================
# §3. PURE-GAUGE DIRECTION TEST
# =====================================================================

def pure_gauge_direction(U_ref, a_dir):
    """
    Build the exact pure-gauge direction in left-trivialized variables.
    
    Under constant gauge transform g(x) = g for all x:
        U_mu(x) -> g U_mu(x) g^dag
    
    In left-trivialized variables at U_ref:
        a_mu(x) = Log( g U_mu(x) g^dag . U_mu(x)^dag )
    
    For infinitesimal rotation by angle epsilon around color axis a_dir:
        Y^{pg}_mu(x) = d/d(epsilon) Log(g(eps) U U^dag g(eps)^dag)|_{eps=0}
                      = Ad(U_mu(x)) . e_a - e_a
    
    This is the EXACT pure-gauge direction, not a finite-difference approximation.
    """
    dims = U_ref.shape[:4]
    Y_pg = np.zeros((*dims, 4, 3))
    
    e_a = np.zeros(3)
    e_a[a_dir] = 1.0
    
    for mu in range(4):
        Ad_U = _adjoint_3x3_vectorized(U_ref[..., mu, :, :])  # (..., 3, 3)
        # Ad(U) . e_a - e_a
        transported = np.einsum('...ab,b->...a', Ad_U, e_a)
        Y_pg[..., mu, :] = transported - e_a
    
    return Y_pg


# =====================================================================
# §4. LOW-MODE OVERLAP MATRIX
# =====================================================================

def compute_low_mode_overlaps(U_ref, tangents, labels, threshold=0.06):
    """
    Compute overlap of each tangent with the spectrally-thresholded
    low-mode subspace of Delta_1.
    """
    dims = U_ref.shape[:4]
    V = int(np.prod(dims))
    
    print("  Building vector operator...")
    Delta1 = build_vector_operator(U_ref, mass_sq=1e-4)
    
    print("  Finding low eigenmodes...")
    n_find = 20
    vals, vecs = eigsh(Delta1, k=n_find, which='SM')
    
    # Sort by absolute value
    order = np.argsort(np.abs(vals))
    vals = vals[order]
    vecs = vecs[:, order]
    
    mask = np.abs(vals) < threshold
    n_low = np.sum(mask)
    print(f"  Found {n_low} modes below |lambda| < {threshold}")
    print(f"  Eigenvalues: {vals[:n_low]}")
    
    low_vecs = vecs[:, mask]  # (12V, n_low)
    
    # For each tangent, compute overlap with low subspace
    n_tang = len(tangents)
    overlaps = np.zeros(n_tang)
    overlap_matrix = np.zeros((n_low, n_tang))
    
    for I in range(n_tang):
        z_flat = tangents[I].reshape(12 * V)
        z_norm_sq = np.dot(z_flat, z_flat)
        
        # Project onto low subspace
        coeffs = low_vecs.T @ z_flat  # (n_low,)
        proj_norm_sq = np.dot(coeffs, coeffs)
        
        overlaps[I] = proj_norm_sq / (z_norm_sq + 1e-30)
        overlap_matrix[:, I] = coeffs / (np.sqrt(z_norm_sq) + 1e-30)
        
        print(f"  {labels[I]:12s}: |P_low Z|^2/|Z|^2 = {overlaps[I]:.4f}")
    
    return overlaps, overlap_matrix, vals[:n_low], low_vecs


# =====================================================================
# §5. MASTER DRIVER
# =====================================================================

def main():
    L_s, L_t, rho = 16, 8, 3.0
    x0 = [3.5, 7.5, 7.5, 7.5]
    epsilon = 0.09
    delta_trans, delta_rho, delta_theta = 0.1, 0.05, 0.05
    
    print("=" * 70)
    print("LEFT-TRIVIALIZED JACOBIAN + STRUCTURAL CLOSURE")
    print(f"Lattice: {L_s}^3 x {L_t}, rho={rho}, x0={x0}")
    print("=" * 70)
    
    # ── Reference caloron ─────────────────────────────────────────────
    print("\n[1/6] Reference caloron...")
    U_ref = init_caloron(L_s, L_t, rho=rho, x0=x0, self_dual=True, n_sub=16)
    full_diagnostic(U_ref, f"Reference {L_s}^3x{L_t}")
    
    # ── Raw tangents in LT variables ──────────────────────────────────
    print("\n[2/6] Computing LEFT-TRIVIALIZED raw tangents...")
    raw_tangents = []
    labels = []
    
    for mu_dir in range(4):
        print(f"  Translation mu={mu_dir}...", end=" ", flush=True)
        Y = raw_tangent_translation_LT(U_ref, L_s, L_t, rho, x0, mu_dir,
                                        delta_trans, self_dual=True, n_sub=16)
        raw_tangents.append(Y)
        labels.append(f"trans_{mu_dir}")
        print(f"|Y| = {np.sqrt(np.sum(Y**2)):.6f}")
    
    print(f"  Scale rho...", end=" ", flush=True)
    Y = raw_tangent_scale_LT(U_ref, L_s, L_t, rho, x0, delta_rho,
                              self_dual=True, n_sub=16)
    raw_tangents.append(Y)
    labels.append("scale")
    print(f"|Y| = {np.sqrt(np.sum(Y**2)):.6f}")
    
    for a_dir in range(3):
        print(f"  Orientation a={a_dir}...", end=" ", flush=True)
        Y = raw_tangent_orientation_LT(U_ref, a_dir, delta_theta)
        raw_tangents.append(Y)
        labels.append(f"orient_{a_dir}")
        print(f"|Y| = {np.sqrt(np.sum(Y**2)):.6f}")
    
    # ── Gauge cleaning ────────────────────────────────────────────────
    print(f"\n[3/6] Gauge-cleaning tangents...")
    cleaned = []
    for I in range(8):
        print(f"  Cleaning {labels[I]}...", end=" ", flush=True)
        Z, omega, info = gauge_clean_tangent(U_ref, raw_tangents[I],
                                              mass_sq=1e-6, cg_tol=1e-10,
                                              cg_maxiter=5000)
        cleaned.append(Z)
        s = "OK" if info['cg_converged'] else "FAIL"
        print(f"CG {s}, |D.Y|={info['div_Y_norm']:.4e}, "
              f"|D.Z|={info['div_Z_norm']:.4e}, "
              f"ratio={info['cleaning_ratio']:.4e}")
    
    # ── Gram matrix and Jacobian ──────────────────────────────────────
    print(f"\n[4/6] Building Gram matrix and Jacobian...")
    G0, H = build_gram_and_H(cleaned)
    J_exact, trace_GinvH, jac_info = compute_exact_jacobian(G0, H, epsilon)
    
    print(f"\n  G^(0) diagonal: {np.diag(G0)}")
    print(f"  H diagonal:     {np.diag(H)}")
    print(f"  f_0 diagonal:   {jac_info['f0_diagonal']}")
    print(f"  G^(0) condition: {jac_info['G0_condition']:.4e}")
    print(f"\n  Tr[(G^(0))^{{-1}} H] = {trace_GinvH:.6f}")
    print(f"  J_exact = {epsilon} * {trace_GinvH:.6f} = {J_exact:.6f}")
    
    # ── F_{mu,nu} cross-check ─────────────────────────────────────────
    print(f"\n[5/6] F_{{mu,nu}} cross-check (LEFT-TRIVIALIZED tangents)...")
    dims = U_ref.shape[:4]
    for nu in range(4):
        F_ref = np.zeros((*dims, 4, 3))
        for mu in range(4):
            if mu == nu: continue
            Fmn = field_strength(U_ref, mu, nu)
            for a in range(3):
                F_ref[..., mu, a] = np.einsum('ij,...ji->...', SIGMA[a], Fmn).real
        z_flat = cleaned[nu].flatten()
        f_flat = F_ref.flatten()
        corr = np.dot(z_flat, f_flat) / (np.linalg.norm(z_flat) * np.linalg.norm(f_flat) + 1e-30)
        print(f"  Z_trans[{nu}] vs F_{{.,{nu}}}: corr = {corr:+.6f}")
    
    # ── Pure-gauge test ───────────────────────────────────────────────
    print(f"\n[5b/6] Pure-gauge direction test...")
    for a in range(3):
        Ypg = pure_gauge_direction(U_ref, a)
        # Check: is this exactly in the span of the orientation tangents?
        # Compute overlap with orientation tangent
        corr = np.dot(Ypg.flatten(), cleaned[5+a].flatten()) / (
            np.linalg.norm(Ypg.flatten()) * np.linalg.norm(cleaned[5+a].flatten()) + 1e-30)
        
        # Check D.Y_pg (should be nonzero — it's NOT transverse)
        div_pg = covariant_divergence(U_ref, Ypg)
        div_norm = np.sqrt(np.sum(div_pg**2))
        pg_norm = np.sqrt(np.sum(Ypg**2))
        
        print(f"  PG dir a={a}: |Y_pg|={pg_norm:.4f}, |D.Y_pg|={div_norm:.4f}, "
              f"corr(Y_pg, Z_orient_{a})={corr:+.4f}")
    
    # ── Low-mode overlap (if feasible) ────────────────────────────────
    print(f"\n[6/6] Low-mode overlap matrix...")
    try:
        overlaps, Omat, low_vals, low_vecs = compute_low_mode_overlaps(
            U_ref, cleaned, labels, threshold=0.06)
    except Exception as e:
        print(f"  Skipped: {e}")
    
    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON: LOG-LINK vs LEFT-TRIVIALIZED")
    print(f"  Log-link  Tr[G^-1 H] = 2.066905  (from previous run)")
    print(f"  Left-triv Tr[G^-1 H] = {trace_GinvH:.6f}")
    print(f"  Difference: {abs(trace_GinvH - 2.066905):.6f}")
    print(f"  Relative:   {abs(trace_GinvH - 2.066905)/2.066905:.6f}")
    print(f"\n  Log-link  J = +0.186021")
    print(f"  Left-triv J = {J_exact:+.6f}")
    print(f"\n  delta_xi log zeta_full = {-2.23 + J_exact:+.4f} +/- 0.11")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
