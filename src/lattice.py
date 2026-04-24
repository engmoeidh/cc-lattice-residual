"""
PFMT CC Programme — SU(2) Lattice Gauge Field Algebra

All operations are exact and fully vectorized over the lattice volume.

Conventions
-----------
Link field:  U.shape = (N0, N1, N2, N3, 4, 2, 2), dtype = complex128
    U[x0, x1, x2, x3, mu]  is the 2x2 SU(2) matrix on the link
    from site x in direction mu.

    N0 = temporal extent (L_t), N1,N2,N3 = spatial extents (L_x,L_y,L_z).
    For isotropic lattices: N0 = N1 = N2 = N3 = L.

Directions:  mu = 0, 1, 2, 3  (Euclidean time = 0)

Shift:  shift(U, mu, +1)  returns U(x + hat{mu})
        implemented via np.roll with periodic BC.
        Works correctly for anisotropic extents.

SU(2) parametrisation:
    U = a0 * I + i * a_k * sigma_k
    with  a0^2 + a1^2 + a2^2 + a3^2 = 1
"""

import numpy as np
from typing import Tuple, Union


# ---------------------------------------------------------------------------
#  Anisotropic lattice helpers
# ---------------------------------------------------------------------------

def _parse_dims(dims: Union[int, Tuple[int, ...]]) -> Tuple[int, int, int, int]:
    """Convert dims specification to (N0, N1, N2, N3).

    Accepts:
        int L          -> (L, L, L, L)
        (L,)           -> (L, L, L, L)
        (Lt, Ls)       -> (Lt, Ls, Ls, Ls)
        (N0, N1, N2, N3) -> as-is
    """
    if isinstance(dims, (int, np.integer)):
        return (int(dims),) * 4
    dims = tuple(int(d) for d in dims)
    if len(dims) == 1:
        return dims * 4
    if len(dims) == 2:
        return (dims[0], dims[1], dims[1], dims[1])
    if len(dims) == 4:
        return dims
    raise ValueError(f"dims must be int, (L,), (Lt,Ls), or (N0,N1,N2,N3), got {dims}")


def lattice_dims(U: np.ndarray) -> Tuple[int, int, int, int]:
    """Extract (N0, N1, N2, N3) from a link field."""
    return tuple(U.shape[:4])


def lattice_volume(U: np.ndarray) -> int:
    """Total number of lattice sites."""
    return int(np.prod(U.shape[:4]))

# ---------------------------------------------------------------------------
#  Pauli matrices (fixed, do not modify)
# ---------------------------------------------------------------------------
SIGMA_1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
IDENTITY = np.eye(2, dtype=np.complex128)
SIGMA = np.array([SIGMA_1, SIGMA_2, SIGMA_3])  # shape (3, 2, 2)


# ---------------------------------------------------------------------------
#  Basic SU(2) element operations
# ---------------------------------------------------------------------------

def su2_identity(shape: Tuple[int, ...] = ()) -> np.ndarray:
    """Return identity SU(2) matrices with given batch shape.

    Parameters
    ----------
    shape : tuple
        Batch dimensions.  E.g. (L,L,L,L) for one per site,
        or (L,L,L,L,4) for the full link field.

    Returns
    -------
    U : ndarray, shape (*shape, 2, 2), complex128
    """
    U = np.zeros((*shape, 2, 2), dtype=np.complex128)
    U[..., 0, 0] = 1.0
    U[..., 1, 1] = 1.0
    return U


def su2_dagger(U: np.ndarray) -> np.ndarray:
    """Hermitian conjugate U^dagger.

    For SU(2):  U^dag = [[alpha*, -beta], [beta*, alpha]]
    Equivalent to  U.conj() with last two axes transposed.
    """
    return np.conj(np.swapaxes(U, -2, -1))


def su2_multiply(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Matrix product U . V over the last two (matrix) indices.

    Uses np.matmul which broadcasts over all leading dimensions.
    """
    return np.matmul(U, V)


def su2_trace(U: np.ndarray) -> np.ndarray:
    """Trace over the 2x2 matrix indices.

    Returns
    -------
    tr : ndarray, shape U.shape[:-2], complex128
    """
    return U[..., 0, 0] + U[..., 1, 1]


def su2_det(U: np.ndarray) -> np.ndarray:
    """Determinant of 2x2 matrices.

    det = U[0,0]*U[1,1] - U[0,1]*U[1,0]
    """
    return U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]


def su2_project(M: np.ndarray) -> np.ndarray:
    """Project a general 2x2 complex matrix to the nearest SU(2) element.

    Method: extract the SU(2)-compatible part using the quaternion
    structure, then normalise.

    For M = [[m00, m01], [m10, m11]], the SU(2) projection is:
        alpha = (m00 + m11*) / 2
        beta  = (m01 - m10*) / 2
        r     = sqrt(|alpha|^2 + |beta|^2)
        U     = [[ alpha/r,  beta/r ],
                 [-beta*/r, alpha*/r ]]

    This is the unique nearest SU(2) element in Frobenius norm
    (Cabibbo-Marinari projection for N=2).

    Parameters
    ----------
    M : ndarray, shape (..., 2, 2), complex

    Returns
    -------
    U : ndarray, shape (..., 2, 2), complex128
        Exact SU(2) element (det = 1, U^dag U = I to machine precision).
    """
    alpha = 0.5 * (M[..., 0, 0] + np.conj(M[..., 1, 1]))
    beta = 0.5 * (M[..., 0, 1] - np.conj(M[..., 1, 0]))

    r = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
    # Guard against zero (degenerate input)
    r = np.maximum(r, 1e-15)

    alpha_n = alpha / r
    beta_n = beta / r

    U = np.zeros_like(M)
    U[..., 0, 0] = alpha_n
    U[..., 0, 1] = beta_n
    U[..., 1, 0] = -np.conj(beta_n)
    U[..., 1, 1] = np.conj(alpha_n)
    return U


def su2_random_uniform(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Generate random SU(2) elements with uniform Haar measure.

    Method:  sample (a0, a1, a2, a3) from 4D Gaussian,
    project to S^3.  This gives exact uniform Haar measure.

    Parameters
    ----------
    shape : tuple
        Batch dimensions.  The returned array has shape (*shape, 2, 2).
    rng : numpy Generator
        Random number generator for reproducibility.

    Returns
    -------
    U : ndarray, shape (*shape, 2, 2), complex128
        SU(2) elements with det = 1, unitary.
    """
    # 4-component quaternion vector
    q = rng.standard_normal((*shape, 4))
    r = np.sqrt(np.sum(q**2, axis=-1, keepdims=True))
    r = np.maximum(r, 1e-15)
    q = q / r  # now on S^3

    a0 = q[..., 0]
    a1 = q[..., 1]
    a2 = q[..., 2]
    a3 = q[..., 3]

    # U = a0*I + i*(a1*s1 + a2*s2 + a3*s3)
    # = [[a0 + i*a3,  a2 + i*a1],
    #    [-a2 + i*a1, a0 - i*a3]]
    U = np.zeros((*shape, 2, 2), dtype=np.complex128)
    U[..., 0, 0] = a0 + 1j * a3
    U[..., 0, 1] = a2 + 1j * a1
    U[..., 1, 0] = -a2 + 1j * a1
    U[..., 1, 1] = a0 - 1j * a3
    return U


def su2_random_near_identity(shape: Tuple[int, ...], epsilon: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Generate random SU(2) elements near the identity.

    U = exp(i * epsilon * n_k * sigma_k / 2)

    where n_k are Gaussian with unit variance.
    Used for small perturbations in cooling/flow.

    Parameters
    ----------
    shape : tuple
        Batch dimensions.
    epsilon : float
        Spread parameter.  epsilon -> 0 gives identity.
    rng : numpy Generator

    Returns
    -------
    U : ndarray, shape (*shape, 2, 2), complex128
    """
    # Algebra coefficients
    omega = epsilon * rng.standard_normal((*shape, 3))
    return su2_exp_algebra(omega)


def su2_exp_algebra(omega: np.ndarray) -> np.ndarray:
    """Exponential map from su(2) algebra to SU(2) group.

    Given omega = (omega_1, omega_2, omega_3),
    computes  U = exp(i * omega_k * sigma_k / 2).

    Exact closed form:
        |omega| = sqrt(omega_1^2 + omega_2^2 + omega_3^2)
        U = cos(|omega|/2) * I  +  i * sin(|omega|/2) / |omega| * omega_k * sigma_k

    Parameters
    ----------
    omega : ndarray, shape (..., 3)
        Algebra coefficients (real).

    Returns
    -------
    U : ndarray, shape (..., 2, 2), complex128
    """
    norm = np.sqrt(np.sum(omega**2, axis=-1))  # shape (...)
    half_norm = norm / 2.0

    # Avoid division by zero: Taylor expand sin(x)/x near x=0
    # sin(x)/x = 1 - x^2/6 + ...  accurate to ~1e-16 for |x| < 0.01
    safe = norm > 1e-10
    sinc_half = np.where(safe,
                         np.sin(half_norm) / np.where(safe, half_norm, 1.0),
                         1.0 - half_norm**2 / 6.0)
    cos_half = np.cos(half_norm)

    # omega_hat_k = omega_k * sinc(|omega|/2) / 2
    # But we defined sinc_half = sin(|omega|/2) / (|omega|/2)
    # so sin(|omega|/2) / |omega| = sinc_half / 2
    s_over_norm = np.where(safe,
                           np.sin(half_norm) / np.where(safe, norm, 1.0),
                           0.5 - norm**2 / 48.0)

    a0 = cos_half
    a1 = s_over_norm * omega[..., 0]
    a2 = s_over_norm * omega[..., 1]
    a3 = s_over_norm * omega[..., 2]

    U = np.zeros((*omega.shape[:-1], 2, 2), dtype=np.complex128)
    U[..., 0, 0] = a0 + 1j * a3
    U[..., 0, 1] = a2 + 1j * a1
    U[..., 1, 0] = -a2 + 1j * a1
    U[..., 1, 1] = a0 - 1j * a3
    return U


def su2_log_algebra(U: np.ndarray) -> np.ndarray:
    """Logarithm map from SU(2) to su(2) algebra.

    Inverse of su2_exp_algebra.
    Returns omega such that U = exp(i * omega_k * sigma_k / 2).

    Parameters
    ----------
    U : ndarray, shape (..., 2, 2), complex128
        SU(2) elements.

    Returns
    -------
    omega : ndarray, shape (..., 3), float64
    """
    # Extract quaternion components
    a0 = 0.5 * np.real(U[..., 0, 0] + U[..., 1, 1])
    a1 = 0.5 * np.imag(U[..., 0, 1] + U[..., 1, 0])
    a2 = 0.5 * np.real(U[..., 0, 1] - U[..., 1, 0])
    a3 = 0.5 * np.imag(U[..., 0, 0] - U[..., 1, 1])

    # |a_vec| = sin(|omega|/2), a0 = cos(|omega|/2)
    a_vec_norm = np.sqrt(a1**2 + a2**2 + a3**2)
    half_norm = np.arctan2(a_vec_norm, a0)  # = |omega|/2, in [0, pi]

    # omega_k = a_k * |omega| / sin(|omega|/2)
    # = a_k * 2 * half_norm / sin(half_norm)
    safe = a_vec_norm > 1e-10
    factor = np.where(safe,
                      2.0 * half_norm / np.where(safe, a_vec_norm, 1.0),
                      2.0)  # Taylor: 2*x/sin(x) -> 2 as x->0

    omega = np.stack([factor * a1, factor * a2, factor * a3], axis=-1)
    return omega


# ---------------------------------------------------------------------------
#  Lattice operations
# ---------------------------------------------------------------------------

def init_cold(dims: Union[int, Tuple[int, ...]] = 8) -> np.ndarray:
    """Cold start: all links = identity.

    Parameters
    ----------
    dims : int or tuple
        int L -> (L,L,L,L).  (Lt, Ls) -> (Lt,Ls,Ls,Ls).  (N0,N1,N2,N3) -> as-is.

    Returns
    -------
    U : ndarray, shape (N0, N1, N2, N3, 4, 2, 2), complex128
    """
    d = _parse_dims(dims)
    return su2_identity((*d, 4))


def init_hot(dims: Union[int, Tuple[int, ...]], rng: np.random.Generator) -> np.ndarray:
    """Hot start: all links = random SU(2) (Haar uniform).

    Parameters
    ----------
    dims : int or tuple (same as init_cold)
    rng : numpy Generator

    Returns
    -------
    U : ndarray, shape (N0, N1, N2, N3, 4, 2, 2), complex128
    """
    d = _parse_dims(dims)
    return su2_random_uniform((*d, 4), rng)


def shift(U: np.ndarray, mu: int, direction: int = 1) -> np.ndarray:
    """Periodic shift of a lattice field in direction mu.

    shift(U, mu, +1)  returns  U(x + hat{mu})
    shift(U, mu, -1)  returns  U(x - hat{mu})

    Parameters
    ----------
    U : ndarray
        Any lattice field with the first 4 axes being (x0, x1, x2, x3).
    mu : int
        Direction (0, 1, 2, 3).
    direction : int
        +1 for forward, -1 for backward.
    """
    return np.roll(U, -direction, axis=mu)


# ---------------------------------------------------------------------------
#  Plaquette
# ---------------------------------------------------------------------------

def plaquette(U: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """Compute the plaquette  P_{mu,nu}(x) at every lattice site.

    P_{mu,nu}(x) = U_mu(x) . U_nu(x+mu) . U_mu^dag(x+nu) . U_nu^dag(x)

    Parameters
    ----------
    U : ndarray, shape (L,L,L,L, 4, 2, 2)
    mu, nu : int
        Directions (mu != nu).

    Returns
    -------
    P : ndarray, shape (L,L,L,L, 2, 2), complex128
    """
    assert mu != nu
    U_mu = U[..., mu, :, :]                           # U_mu(x)
    U_nu_shifted = shift(U[..., nu, :, :], mu, +1)    # U_nu(x+mu)
    U_mu_shifted = shift(U[..., mu, :, :], nu, +1)    # U_mu(x+nu)
    U_nu = U[..., nu, :, :]                           # U_nu(x)

    # P = U_mu . U_nu(x+mu) . U_mu^dag(x+nu) . U_nu^dag(x)
    P = su2_multiply(U_mu, U_nu_shifted)
    P = su2_multiply(P, su2_dagger(U_mu_shifted))
    P = su2_multiply(P, su2_dagger(U_nu))
    return P


def plaquette_sum_trace(U: np.ndarray) -> float:
    """Sum of Re Tr(P_{mu,nu}) over all sites and planes.

    Returns
    -------
    S : float
        sum_{x, mu<nu} Re Tr(P_{mu,nu}(x))
    """
    total = 0.0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            P = plaquette(U, mu, nu)
            total += np.sum(np.real(su2_trace(P)))
    return total


# ---------------------------------------------------------------------------
#  Rectangle (1x2) loops for Symanzik improvement
# ---------------------------------------------------------------------------

def rectangle_munu(U: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """Compute the 1x2 rectangle loop (2 steps in mu, 1 step in nu).

    R^{(1x2)}_{mu,nu}(x)
       = U_mu(x) . U_mu(x+mu) . U_nu(x+2*mu) . U_mu^dag(x+mu+nu) . U_mu^dag(x+nu) . U_nu^dag(x)

    Parameters
    ----------
    U : ndarray, shape (L,L,L,L, 4, 2, 2)
    mu, nu : int

    Returns
    -------
    R : ndarray, shape (L,L,L,L, 2, 2), complex128
    """
    assert mu != nu
    U_mu_x = U[..., mu, :, :]
    U_mu_xpmu = shift(U[..., mu, :, :], mu, +1)
    U_nu_xp2mu = shift(shift(U[..., nu, :, :], mu, +1), mu, +1)
    U_mu_dag_xpmupnu = su2_dagger(shift(shift(U[..., mu, :, :], mu, +1), nu, +1))
    U_mu_dag_xpnu = su2_dagger(shift(U[..., mu, :, :], nu, +1))
    U_nu_dag_x = su2_dagger(U[..., nu, :, :])

    R = su2_multiply(U_mu_x, U_mu_xpmu)
    R = su2_multiply(R, U_nu_xp2mu)
    R = su2_multiply(R, U_mu_dag_xpmupnu)
    R = su2_multiply(R, U_mu_dag_xpnu)
    R = su2_multiply(R, U_nu_dag_x)
    return R


def rectangle_sum_trace(U: np.ndarray) -> float:
    """Sum of Re Tr over all 1x2 AND 2x1 rectangles.

    For each ordered pair (mu, nu) with mu != nu, there is one
    1x2 rectangle (2 in mu, 1 in nu).
    For the pair (nu, mu), we get the 2x1 orientation.

    Total:  sum over all 12 ordered (mu, nu) pairs with mu != nu.

    Returns
    -------
    S : float
        sum_{x, mu!=nu} Re Tr(R^{1x2}_{mu,nu}(x))
    """
    total = 0.0
    for mu in range(4):
        for nu in range(4):
            if mu == nu:
                continue
            R = rectangle_munu(U, mu, nu)
            total += np.sum(np.real(su2_trace(R)))
    return total


# ---------------------------------------------------------------------------
#  Gauge actions
# ---------------------------------------------------------------------------

def wilson_action(U: np.ndarray, beta: float) -> float:
    """Wilson plaquette action.

    S_W = (beta / nc) * sum_{x, mu<nu} Re Tr(I - P_{mu,nu}(x))
        = (beta / nc) * [ nc * 6 * V - plaquette_sum_trace ]

    where V = N0*N1*N2*N3 and 6 = number of planes.
    """
    nc = 2
    V = lattice_volume(U)
    n_planes = 6   # (4 choose 2)
    plaq_sum = plaquette_sum_trace(U)
    return (beta / nc) * (nc * n_planes * V - plaq_sum)


def symanzik_action(U: np.ndarray, beta: float,
                     c0: float = 5.0/3.0, c1: float = -1.0/12.0) -> float:
    """Symanzik tree-level O(a^2) improved action.

    S_sym = (beta / nc) * [ c0 * sum_{plaq} Re Tr(I - P)
                          + c1 * sum_{rect} Re Tr(I - R) ]

    Parameters
    ----------
    U : ndarray, shape (L,L,L,L, 4, 2, 2)
    beta : float
    c0, c1 : float
        Symanzik coefficients with c0 + 8*c1 = 1.

    Returns
    -------
    S : float
    """
    nc = 2
    V = lattice_volume(U)

    plaq_sum = plaquette_sum_trace(U)
    rect_sum = rectangle_sum_trace(U)

    # Number of plaquettes per site: 6 (mu<nu)
    # Number of 1x2 rectangles per site: 12 (all mu!=nu pairs)
    n_plaq = 6 * V
    n_rect = 12 * V

    S = (beta / nc) * (c0 * (nc * n_plaq - plaq_sum)
                       + c1 * (nc * n_rect - rect_sum))
    return S


# ---------------------------------------------------------------------------
#  Staples (for link updates and force computation)
# ---------------------------------------------------------------------------

def plaquette_staple(U: np.ndarray, mu: int) -> np.ndarray:
    """Sum of plaquette staples for link U_mu(x).

    For each nu != mu, the staple is:
        S^+_{mu,nu}(x) = U_nu(x+mu) . U_mu^dag(x+nu) . U_nu^dag(x)
      + S^-_{mu,nu}(x) = U_nu^dag(x+mu-nu) . U_mu^dag(x-nu) . U_nu(x-nu)

    Returns the sum over all nu != mu.

    Parameters
    ----------
    U : ndarray, shape (L,L,L,L, 4, 2, 2)
    mu : int

    Returns
    -------
    staple : ndarray, shape (*dims, 2, 2), complex128
    """
    dims = U.shape[:4]
    staple = np.zeros((*dims, 2, 2), dtype=np.complex128)

    for nu in range(4):
        if nu == mu:
            continue

        # Forward staple:  U_nu(x+mu) . U_mu^dag(x+nu) . U_nu^dag(x)
        A = shift(U[..., nu, :, :], mu, +1)
        B = su2_dagger(shift(U[..., mu, :, :], nu, +1))
        C = su2_dagger(U[..., nu, :, :])
        staple += su2_multiply(su2_multiply(A, B), C)

        # Backward staple:  U_nu^dag(x+mu-nu) . U_mu^dag(x-nu) . U_nu(x-nu)
        A = su2_dagger(shift(shift(U[..., nu, :, :], mu, +1), nu, -1))
        B = su2_dagger(shift(U[..., mu, :, :], nu, -1))
        C = shift(U[..., nu, :, :], nu, -1)
        staple += su2_multiply(su2_multiply(A, B), C)

    return staple


def rectangle_staple(U: np.ndarray, mu: int) -> np.ndarray:
    """Sum of rectangle (1x2 and 2x1) staples for link U_mu(x).

    For each nu != mu, there are 6 rectangle staple contributions
    (3 from 1x2 in the mu-nu plane that pass through the link U_mu(x),
    plus their backward reflections).

    Returns the total rectangle staple sum.

    Parameters
    ----------
    U : ndarray, shape (L,L,L,L, 4, 2, 2)
    mu : int

    Returns
    -------
    staple : ndarray, shape (*dims, 2, 2), complex128
    """
    dims = U.shape[:4]
    staple = np.zeros((*dims, 2, 2), dtype=np.complex128)

    for nu in range(4):
        if nu == mu:
            continue

        # === Rectangle with 2 steps in mu, 1 in nu ===
        # Type A (forward-forward):
        # U_mu(x+mu) . U_nu(x+2mu) . U_mu^dag(x+mu+nu) . U_mu^dag(x+nu) . U_nu^dag(x)
        A = shift(U[..., mu, :, :], mu, +1)
        B = shift(shift(U[..., nu, :, :], mu, +1), mu, +1)
        C = su2_dagger(shift(shift(U[..., mu, :, :], mu, +1), nu, +1))
        D = su2_dagger(shift(U[..., mu, :, :], nu, +1))
        E = su2_dagger(U[..., nu, :, :])
        staple += su2_multiply(su2_multiply(su2_multiply(su2_multiply(A, B), C), D), E)

        # Type A (forward-backward):
        # U_mu(x+mu) . U_nu^dag(x+2mu-nu) . U_mu^dag(x+mu-nu) . U_mu^dag(x-nu) . U_nu(x-nu)
        A = shift(U[..., mu, :, :], mu, +1)
        B = su2_dagger(shift(shift(shift(U[..., nu, :, :], mu, +1), mu, +1), nu, -1))
        C = su2_dagger(shift(shift(U[..., mu, :, :], mu, +1), nu, -1))
        D = su2_dagger(shift(U[..., mu, :, :], nu, -1))
        E = shift(U[..., nu, :, :], nu, -1)
        staple += su2_multiply(su2_multiply(su2_multiply(su2_multiply(A, B), C), D), E)

        # Type B (backward-forward):
        # U_nu(x+mu) . U_mu^dag(x+nu) . U_mu^dag(x-mu+nu) . U_nu^dag(x-mu) . U_mu(x-mu)
        A = shift(U[..., nu, :, :], mu, +1)
        B = su2_dagger(shift(U[..., mu, :, :], nu, +1))
        C = su2_dagger(shift(shift(U[..., mu, :, :], mu, -1), nu, +1))
        D = su2_dagger(shift(U[..., nu, :, :], mu, -1))
        E = shift(U[..., mu, :, :], mu, -1)
        staple += su2_multiply(su2_multiply(su2_multiply(su2_multiply(A, B), C), D), E)

        # Type B (backward-backward):
        # U_nu^dag(x+mu-nu) . U_mu^dag(x-nu) . U_mu^dag(x-mu-nu) . U_nu(x-mu-nu) . U_mu(x-mu)
        A = su2_dagger(shift(shift(U[..., nu, :, :], mu, +1), nu, -1))
        B = su2_dagger(shift(U[..., mu, :, :], nu, -1))
        C = su2_dagger(shift(shift(U[..., mu, :, :], mu, -1), nu, -1))
        D = shift(shift(U[..., nu, :, :], mu, -1), nu, -1)
        E = shift(U[..., mu, :, :], mu, -1)
        staple += su2_multiply(su2_multiply(su2_multiply(su2_multiply(A, B), C), D), E)

        # === Rectangle with 1 step in mu, 2 in nu ===
        # Type C (forward):
        # U_nu(x+mu) . U_nu(x+mu+nu) . U_mu^dag(x+2nu) . U_nu^dag(x+nu) . U_nu^dag(x)
        A = shift(U[..., nu, :, :], mu, +1)
        B = shift(shift(U[..., nu, :, :], mu, +1), nu, +1)
        C = su2_dagger(shift(shift(U[..., mu, :, :], nu, +1), nu, +1))
        D = su2_dagger(shift(U[..., nu, :, :], nu, +1))
        E = su2_dagger(U[..., nu, :, :])
        staple += su2_multiply(su2_multiply(su2_multiply(su2_multiply(A, B), C), D), E)

        # Type C (backward):
        # U_nu^dag(x+mu-nu) . U_nu^dag(x+mu-2nu) . U_mu^dag(x-2nu) . U_nu(x-2nu) . U_nu(x-nu)
        A = su2_dagger(shift(shift(U[..., nu, :, :], mu, +1), nu, -1))
        B = su2_dagger(shift(shift(shift(U[..., nu, :, :], mu, +1), nu, -1), nu, -1))
        C = su2_dagger(shift(shift(U[..., mu, :, :], nu, -1), nu, -1))
        D = shift(shift(U[..., nu, :, :], nu, -1), nu, -1)
        E = shift(U[..., nu, :, :], nu, -1)
        staple += su2_multiply(su2_multiply(su2_multiply(su2_multiply(A, B), C), D), E)

    return staple


def total_staple(U: np.ndarray, mu: int, c0: float, c1: float) -> np.ndarray:
    """Total staple for Symanzik-improved action.

    Sigma_mu(x) = c0 * plaquette_staple + c1 * rectangle_staple

    The force / update direction for link U_mu(x) is:
        U_mu(x) . Sigma_mu(x)  (project to su(2) for the algebra element)

    Parameters
    ----------
    U : ndarray, shape (L,L,L,L, 4, 2, 2)
    mu : int
    c0, c1 : float

    Returns
    -------
    sigma : ndarray, shape (L,L,L,L, 2, 2), complex128
    """
    sigma = c0 * plaquette_staple(U, mu)
    if abs(c1) > 1e-15:
        sigma += c1 * rectangle_staple(U, mu)
    return sigma


# ---------------------------------------------------------------------------
#  Diagnostic utilities
# ---------------------------------------------------------------------------

def average_plaquette(U: np.ndarray) -> float:
    """Average plaquette:  <(1/nc) Re Tr P>  over all sites and planes.

    Range: [0, 1].  Cold start -> 1.  Hot start -> 0.
    Works for anisotropic lattices.
    """
    nc = 2
    V = lattice_volume(U)
    n_planes = 6
    return plaquette_sum_trace(U) / (nc * n_planes * V)


def unitarity_check(U: np.ndarray) -> float:
    """Maximum deviation from unitarity:  max |U^dag U - I|.

    Returns the largest Frobenius norm of (U^dag U - I)
    over all links.
    """
    UdU = su2_multiply(su2_dagger(U), U)
    I = su2_identity(U.shape[:-2])
    diff = UdU - I
    frob2 = np.sum(np.abs(diff)**2, axis=(-2, -1))
    return np.sqrt(np.max(frob2))


def reunitarise(U: np.ndarray) -> np.ndarray:
    """Project all links back to exact SU(2).

    Call periodically to prevent floating-point drift.
    """
    shape = U.shape[:-2]  # (..., 2, 2)
    return su2_project(U.reshape(-1, 2, 2)).reshape(*shape, 2, 2)
