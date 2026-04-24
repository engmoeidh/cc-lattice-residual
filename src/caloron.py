"""
PFMT CC Programme — Harrington-Shepard Caloron Seed

The HS caloron is a self-dual SU(2) instanton on R³ × S¹(β).
It is the natural instanton for the PFMT programme because the
sonic/disformal background selects a Euclidean time direction.

Key advantages over BPST on T⁴:
    - Q = 1 exactly (integer topological charge on R³ × S¹)
    - Naturally periodic in Euclidean time (no image sum needed)
    - Power-law spatial decay: A ~ 1/r₃² (regular gauge)
      Still much better torus compatibility than BPST (1/r₃² vs 1/r₃)
    
    NOTE: the HS prepotential has a Coulombic 1/r₃ tail (not exponential).
    The gauge field ∂ ln f falls as 1/r₃².  The earlier claim of
    "exponential spatial decay" was incorrect.

The HS caloron is given by the 't Hooft ansatz:
    A_μ^a(x) = η̄^a_{μν} ∂_ν ln f(x)     [regular gauge, SD, Q > 0]
    A_μ^a(x) = η^a_{μν} ∂_ν ln f(x)      [regular gauge, ASD, Q < 0]

where the prepotential is the sum over temporal images:
    f(x) = 1 + (πρ²)/(β r₃) × sinh(ξ) / [cosh(ξ) - cos(τ)]

with:
    r₃ = |x_spatial - x₀_spatial|   (3D spatial distance)
    ξ  = 2π r₃ / β                  (scaled spatial distance)
    τ  = 2π (x₀ - x₀⁰) / β         (scaled Euclidean time)
    β  = Euclidean time period

The derivatives ∂_μ ln f are computed via central finite differences
to avoid algebraic errors in the complicated hyperbolic expressions.

GAUGE CONVENTION (CONFIRMED by Test 6):
    Regular gauge has A ~ 1/r₃² at large r₃ (power law).

    Regular gauge SD uses η̄  (anti-self-dual 't Hooft symbols) → Q > 0
    Regular gauge ASD uses η  (self-dual 't Hooft symbols)      → Q < 0

    This is the SAME as singular gauge (both use η̄ for SD).
    The 't Hooft ansatz for instantons uses η̄ regardless of gauge choice.
"""

import numpy as np
from typing import Optional
from src.lattice import (
    su2_identity, su2_multiply, su2_project, su2_exp_algebra,
    SIGMA
)
from src.instanton import ETA_SD, ETA_ASD


# ===================================================================
#  CALORON PREPOTENTIAL
# ===================================================================

def caloron_f(x: np.ndarray, x0: np.ndarray,
               rho: float, beta: float,
               L_spatial: float) -> np.ndarray:
    """HS caloron prepotential f(x).

    f = 1 + (πρ²)/(β r₃) × sinh(ξ) / [cosh(ξ) - cos(τ)]

    Parameters
    ----------
    x : (..., 4) coordinates.  x[..., 0] = Euclidean time.
    x0 : (4,) instanton centre.
    rho : instanton size.
    beta : Euclidean time period (= L_t in lattice units).
    L_spatial : spatial box size (for minimum-image distance).

    Returns
    -------
    f : (...) prepotential values.
    """
    dx = x - x0

    # Spatial minimum-image distance (3D, indices 1,2,3)
    for i in range(1, 4):
        dx[..., i] = dx[..., i] - L_spatial * np.round(dx[..., i] / L_spatial)

    # Temporal coordinate: no minimum-image needed, the caloron sum handles it.
    # But we do wrap into [-β/2, β/2] for the cos(τ) term.
    dx[..., 0] = dx[..., 0] - beta * np.round(dx[..., 0] / beta)

    # 3D spatial distance
    r3 = np.sqrt(dx[..., 1]**2 + dx[..., 2]**2 + dx[..., 3]**2)

    # Scaled variables
    xi = 2.0 * np.pi * r3 / beta
    tau = 2.0 * np.pi * dx[..., 0] / beta

    # Handle r3 → 0 limit:
    # At small r3:  sinh(ξ)/[r₃(cosh(ξ)-cos(τ))]
    #   → (2π/β)/[(2π/β)r₃ (1 + ξ²/2 - cos(τ))]  ... diverges like 1/r₃
    # But (πρ²/β) × (1/r₃) × (ξ/...) → finite as r₃ → 0 if ξ ~ r₃
    # sinh(ξ)/ξ → 1, so the limit is:
    # g → (πρ²/β) × (1/r₃) × r₃ × (2π/β) / (1 - cos(τ))
    #   = 2π²ρ² / [β²(1 - cos(τ))]
    # This is finite for τ ≠ 0.  At τ = 0 (instanton centre): g → ∞.

    r3_safe = np.maximum(r3, 1e-12)
    xi_safe = 2.0 * np.pi * r3_safe / beta

    denom = np.cosh(xi_safe) - np.cos(tau)
    # Guard against denom = 0 (at the instanton centre: r3=0, x0=centre)
    denom_safe = np.maximum(np.abs(denom), 1e-20) * np.sign(denom + 1e-30)

    h = np.sinh(xi_safe) / denom_safe

    g = (np.pi * rho**2 / beta) * h / r3_safe

    return 1.0 + g


# ===================================================================
#  GAUGE FIELD VIA FINITE-DIFFERENCE ∂_μ ln f
# ===================================================================

def caloron_dlogf(x: np.ndarray, x0: np.ndarray,
                    rho: float, beta: float,
                    L_spatial: float,
                    eps: float = 1e-5) -> np.ndarray:
    """Central finite-difference derivatives of ln f.

    ∂_μ ln f ≈ [ln f(x + ε ê_μ) - ln f(x - ε ê_μ)] / (2ε)

    Parameters
    ----------
    x : (..., 4)
    Returns
    -------
    dlnf : (..., 4)   partial derivatives.
    """
    dlnf = np.zeros((*x.shape[:-1], 4), dtype=np.float64)

    for mu in range(4):
        e_mu = np.zeros(4)
        e_mu[mu] = eps

        f_plus = caloron_f(x + e_mu, x0, rho, beta, L_spatial)
        f_minus = caloron_f(x - e_mu, x0, rho, beta, L_spatial)

        # Guard against f ≤ 0 (shouldn't happen if rho > 0)
        f_plus = np.maximum(f_plus, 1e-30)
        f_minus = np.maximum(f_minus, 1e-30)

        dlnf[..., mu] = (np.log(f_plus) - np.log(f_minus)) / (2.0 * eps)

    return dlnf


def caloron_gauge_field(x: np.ndarray, x0: np.ndarray,
                          rho: float, beta: float,
                          L_spatial: float,
                          self_dual: bool = True,
                          eps_FD: float = 1e-5) -> np.ndarray:
    """HS caloron gauge field in regular gauge.

    A_μ(x) = A_μ^a(x) σ_a/2

    SD (Q > 0):   A_μ^a = η̄^a_{μν} ∂_ν ln f     (uses η̄ = ETA_ASD)
    ASD (Q < 0):  A_μ^a = η^a_{μν} ∂_ν ln f      (uses η = ETA_SD)

    Parameters
    ----------
    x : (..., 4)
    x0 : (4,) centre
    rho : instanton size
    beta : Euclidean time period
    L_spatial : spatial box size
    self_dual : True for Q > 0

    Returns
    -------
    A : (..., 4, 2, 2) Hermitian gauge field matrices
    """
    dlnf = caloron_dlogf(x, x0, rho, beta, L_spatial, eps=eps_FD)

    # Regular gauge: SD uses η̄ (ETA_ASD), ASD uses η (ETA_SD)
    # This is CONFIRMED by Test 6: the original assignment gave wrong Q signs.
    # The gauge transformation reg↔sing does NOT simply swap η↔η̄ for the caloron.
    eta = ETA_ASD if self_dual else ETA_SD

    batch = x.shape[:-1]
    A = np.zeros((*batch, 4, 2, 2), dtype=np.complex128)

    for mu in range(4):
        for a in range(3):
            coeff = np.zeros(batch, dtype=np.float64)
            for nu in range(4):
                if abs(eta[a, mu, nu]) > 0.5:
                    coeff += eta[a, mu, nu] * dlnf[..., nu]
            A[..., mu, :, :] += coeff[..., None, None] * (SIGMA[a] / 2.0)

    return A


# ===================================================================
#  PATH-ORDERED LINK (reuses same infrastructure as BPST)
# ===================================================================

def caloron_link(x_start: np.ndarray, mu: int,
                  x0: np.ndarray, rho: float,
                  beta: float, L_spatial: float,
                  self_dual: bool = True,
                  n_sub: int = 16,
                  eps_FD: float = 1e-5) -> np.ndarray:
    """Path-ordered exponential for caloron gauge field.

    U_μ(x) = P exp(i ∫₀¹ A_μ(x + t ê_μ) dt)

    Discretised as n_sub sub-link product.
    """
    dt = 1.0 / n_sub
    batch = x_start.shape[:-1]
    e_mu = np.zeros(4)
    e_mu[mu] = 1.0

    U = su2_identity(batch)

    for k in range(n_sub):
        t_mid = (k + 0.5) * dt
        x_mid = x_start + t_mid * e_mu

        A_mid = caloron_gauge_field(x_mid, x0, rho, beta, L_spatial,
                                     self_dual=self_dual, eps_FD=eps_FD)
        A_mu_mat = A_mid[..., mu, :, :]

        # Extract algebra: omega_a = dt × Tr(sigma_a A_mu)
        omega = np.zeros((*batch, 3), dtype=np.float64)
        for a in range(3):
            tr = np.einsum('ij,...ji->...', SIGMA[a], A_mu_mat)
            omega[..., a] = dt * np.real(tr)

        U = su2_multiply(su2_exp_algebra(omega), U)

    return su2_project(U)


# ===================================================================
#  FULL LATTICE INITIALIZATION
# ===================================================================

def init_caloron(L_spatial: int, L_temporal: int,
                  rho: float,
                  x0: Optional[np.ndarray] = None,
                  self_dual: bool = True,
                  n_sub: int = 16,
                  eps_FD: float = 1e-5) -> np.ndarray:
    """Initialize L³ × L_t lattice as HS caloron.

    Parameters
    ----------
    L_spatial : spatial lattice extent
    L_temporal : temporal lattice extent (= β in lattice units)
    rho : instanton size in lattice units
    x0 : (4,) centre. Default: half-integer offset.
    self_dual : True for Q > 0
    n_sub : sub-links for path integration
    eps_FD : finite-difference step for ∂ ln f
    """
    L_t = L_temporal
    L_s = L_spatial

    if x0 is None:
        # Half-integer offset in all directions
        x0 = np.array([L_t / 2.0 - 0.5,
                        L_s / 2.0 - 0.5,
                        L_s / 2.0 - 0.5,
                        L_s / 2.0 - 0.5])

    beta = float(L_t)

    # Coordinate grid: (L_t, L_s, L_s, L_s, 4)
    t_coords = np.arange(L_t, dtype=np.float64)
    s_coords = np.arange(L_s, dtype=np.float64)
    grid = np.stack(np.meshgrid(t_coords, s_coords, s_coords, s_coords,
                                 indexing='ij'), axis=-1)

    duality_str = "SD" if self_dual else "ASD"
    U = np.zeros((L_t, L_s, L_s, L_s, 4, 2, 2), dtype=np.complex128)

    for mu in range(4):
        print(f"  Caloron links mu={mu} ({duality_str}, rho={rho:.2f}, "
              f"L={L_s}³×{L_t}) ... ", end="", flush=True)

        U[..., mu, :, :] = caloron_link(
            grid, mu, x0, rho, beta, float(L_s),
            self_dual=self_dual, n_sub=n_sub, eps_FD=eps_FD)

        print("done.")

    return U


# ===================================================================
#  SYMMETRIC LATTICE WRAPPER
# ===================================================================

def init_caloron_symmetric(L: int, rho: float,
                             x0: Optional[np.ndarray] = None,
                             self_dual: bool = True,
                             n_sub: int = 16,
                             eps_FD: float = 1e-5) -> np.ndarray:
    """Initialize L⁴ symmetric lattice as caloron (L_t = L_s = L).

    Convenience wrapper for quick tests.
    """
    return init_caloron(L, L, rho, x0, self_dual, n_sub, eps_FD)


# ===================================================================
#  SELF-TEST
# ===================================================================

def _self_test():
    from src.lattice import unitarity_check, average_plaquette
    from src.observables import topological_charge, full_diagnostic

    # 1. Prepotential sanity
    print("Test 1: Caloron prepotential")
    x0 = np.array([4.0, 4.0, 4.0, 4.0])
    x_test = np.array([[4.0, 4.0, 4.0, 5.0]])  # r3 = 1 from centre
    f = caloron_f(x_test, x0, rho=2.0, beta=8.0, L_spatial=8.0)
    print(f"  f(r3=1) = {f[0]:.4f}  (expect > 1)")
    assert f[0] > 1.0

    x_far = np.array([[4.0, 4.0, 4.0, 7.5]])  # r3 = 3.5
    f_far = caloron_f(x_far, x0, rho=2.0, beta=8.0, L_spatial=8.0)
    print(f"  f(r3=3.5) = {f_far[0]:.4f}  (expect close to 1)")
    assert f_far[0] < f[0], "f should decrease with distance"
    print("  [PASS]")

    # 2. Gauge field Hermiticity
    print("\nTest 2: Gauge field Hermiticity")
    x_batch = np.random.uniform(1, 7, size=(20, 4))
    A = caloron_gauge_field(x_batch, x0, 2.0, 8.0, 8.0, self_dual=True)
    for mu in range(4):
        diff = A[:, mu] - np.conj(np.swapaxes(A[:, mu], -2, -1))
        assert np.max(np.abs(diff)) < 1e-12
    print("  [PASS] Gauge field is Hermitian.")

    # 3. Caloron on 8⁴ — test Q
    print("\nTest 3: Caloron on 8⁴ (rho=2.5)")
    U = init_caloron_symmetric(8, rho=2.5, n_sub=16)
    diag = full_diagnostic(U, "Caloron 8⁴")

    print(f"\n  Q = {diag['Q']:.4f}")
    if abs(diag['Q']) > 0.5:
        print(f"  >>> Q > 0.5: caloron has significant topology!")
    if abs(diag['Q'] - 1.0) < 0.3:
        print(f"  >>> Q ≈ 1: CALORON WORKS!")

    # 4. Compare with BPST on same lattice
    print("\nTest 4: BPST comparison on 8⁴ (rho=2.5)")
    from src.instanton import init_bpst_instanton
    U_bpst = init_bpst_instanton(8, rho=2.5, n_sub=16)
    Q_bpst = topological_charge(U_bpst)
    print(f"  BPST Q = {Q_bpst:.4f}")
    print(f"  Caloron Q = {diag['Q']:.4f}")
    if abs(diag['Q']) > abs(Q_bpst):
        print(f"  >>> Caloron gives BETTER Q than BPST!")

    # 5. Caloron on anisotropic 12³ × 8
    print("\nTest 5: Caloron on 12³ × 8 (rho=2.5)")
    U_aniso = init_caloron(12, 8, rho=2.5, n_sub=16)
    diag_aniso = full_diagnostic(U_aniso, "Caloron 12³×8")
    print(f"  Q = {diag_aniso['Q']:.4f}")

    # 6. SD vs ASD
    print("\nTest 6: SD vs ASD duality")
    U_asd = init_caloron_symmetric(6, rho=2.0, self_dual=False, n_sub=16)
    Q_asd = topological_charge(U_asd)
    U_sd = init_caloron_symmetric(6, rho=2.0, self_dual=True, n_sub=16)
    Q_sd = topological_charge(U_sd)
    print(f"  SD:  Q = {Q_sd:+.4f}")
    print(f"  ASD: Q = {Q_asd:+.4f}")
    if Q_sd > 0 and Q_asd < 0:
        print("  [PASS] Correct SD/ASD signs.")
    elif Q_sd < 0 and Q_asd > 0:
        print("  [NOTE] Signs swapped — need to swap η ↔ η̄ convention.")
        print("         The code should swap the eta assignment and retest.")
    else:
        print(f"  [CHECK] Unexpected: both same sign or zero.")


if __name__ == "__main__":
    _self_test()
