"""
PFMT CC Programme — Instanton Initialization (v3)

Singular-gauge BPST on periodic torus with path-ordered links.

DUALITY CONVENTION (from 't Hooft):
    Self-dual F:     singular gauge uses  eta_bar  in A_mu
    Anti-self-dual F: singular gauge uses  eta      in A_mu

    The gauge transformation between regular and singular gauge
    converts eta <-> eta_bar in the potential.

    v1-v2 BUG: ASD was implemented by daggering all links.
    That does not change the duality of F (Re Tr P is invariant).
    v3 FIX: ASD uses the opposite 't Hooft symbol in the potential.

NOTE: BPST-on-torus is only an approximate seed, not an exact
torus saddle.  It must be followed by a proper minimiser to
find the actual lattice saddle point.
"""

import numpy as np
from typing import Optional
from src.lattice import (
    su2_identity, su2_multiply, su2_project, su2_exp_algebra,
    su2_random_uniform, SIGMA, IDENTITY
)


# ---------------------------------------------------------------------------
#  't Hooft symbols
# ---------------------------------------------------------------------------

def _levi_civita_3(i: int, j: int, k: int) -> float:
    if i == j or j == k or i == k:
        return 0.0
    arr = [i, j, k]
    parity = sum(1 for a in range(3) for b in range(a+1, 3) if arr[a] > arr[b])
    return 1.0 if parity % 2 == 0 else -1.0


def thooft_eta() -> np.ndarray:
    """Self-dual: *eta = +eta.

    eta^{a_phys}_{b,c} = eps_{a_phys,b,c},
    eta^{a_phys}_{0,b} = +delta_{a_phys,b},
    eta^{a_phys}_{b,0} = -delta_{a_phys,b}.
    """
    eta = np.zeros((3, 4, 4), dtype=np.float64)
    for a in range(3):
        ap = a + 1
        for b in range(1, 4):
            for c in range(1, 4):
                eta[a, b, c] = _levi_civita_3(ap, b, c)
        for b in range(1, 4):
            if ap == b:
                eta[a, 0, b] = 1.0
                eta[a, b, 0] = -1.0
    return eta


def thooft_eta_bar() -> np.ndarray:
    """Anti-self-dual: *eta_bar = -eta_bar.

    Same spatial block as eta, opposite time-space sign.
    """
    ebar = np.zeros((3, 4, 4), dtype=np.float64)
    for a in range(3):
        ap = a + 1
        for b in range(1, 4):
            for c in range(1, 4):
                ebar[a, b, c] = _levi_civita_3(ap, b, c)
        for b in range(1, 4):
            if ap == b:
                ebar[a, 0, b] = -1.0
                ebar[a, b, 0] = 1.0
    return ebar


ETA_SD = thooft_eta()
ETA_ASD = thooft_eta_bar()


# ---------------------------------------------------------------------------
#  Singular-gauge BPST
# ---------------------------------------------------------------------------

def bpst_singular(x: np.ndarray, x0: np.ndarray, rho: float,
                   L: float, self_dual: bool = True) -> np.ndarray:
    """Singular-gauge BPST gauge potential.

    Self-dual F:   A_mu^a = eta_bar^a_{mu,nu} * 2 rho^2 dx_nu / (r^2(r^2+rho^2))
    Anti-self-dual F: A_mu^a = eta^a_{mu,nu} * 2 rho^2 dx_nu / (r^2(r^2+rho^2))

    Parameters
    ----------
    x : (..., 4), coordinates
    x0 : (4,), instanton centre
    rho : float, instanton size
    L : float, box size (periodic minimum-image)
    self_dual : bool
        True -> SD field strength (uses eta_bar in A)
        False -> ASD field strength (uses eta in A)

    Returns
    -------
    A : (..., 4, 2, 2), complex128
    """
    # DUALITY: SD uses eta_bar, ASD uses eta (in singular gauge)
    eta = ETA_ASD if self_dual else ETA_SD

    dx = x - x0
    dx = dx - L * np.round(dx / L)
    r2 = np.sum(dx**2, axis=-1)
    r2_safe = np.maximum(r2, 1e-20)
    f = 2.0 * rho**2 / (r2_safe * (r2_safe + rho**2))

    batch = x.shape[:-1]
    A = np.zeros((*batch, 4, 2, 2), dtype=np.complex128)
    for mu in range(4):
        for a in range(3):
            coeff = np.zeros(batch, dtype=np.float64)
            for nu in range(4):
                if abs(eta[a, mu, nu]) > 0.5:
                    coeff += eta[a, mu, nu] * f * dx[..., nu]
            A[..., mu, :, :] += coeff[..., None, None] * (SIGMA[a] / 2.0)
    return A


# ---------------------------------------------------------------------------
#  Path-ordered link
# ---------------------------------------------------------------------------

def path_ordered_link(x_start: np.ndarray, mu: int, x0: np.ndarray,
                       rho: float, L: float,
                       self_dual: bool = True, n_sub: int = 16
                       ) -> np.ndarray:
    """U_mu(x) = P exp(i int_0^1 A_mu(x + t hat_mu) dt).

    Discretised as n_sub sub-link product with exact SU(2) exponentials.
    """
    dt = 1.0 / n_sub
    batch = x_start.shape[:-1]
    e_mu = np.zeros(4, dtype=np.float64)
    e_mu[mu] = 1.0

    U = su2_identity(batch)
    for k in range(n_sub):
        t_mid = (k + 0.5) * dt
        x_mid = x_start + t_mid * e_mu
        A_mid = bpst_singular(x_mid, x0, rho, L, self_dual)
        A_mu_mat = A_mid[..., mu, :, :]

        omega = np.zeros((*batch, 3), dtype=np.float64)
        for a in range(3):
            tr = np.einsum('ij,...ji->...', SIGMA[a], A_mu_mat)
            omega[..., a] = dt * np.real(tr)

        U = su2_multiply(su2_exp_algebra(omega), U)

    return su2_project(U)


# ---------------------------------------------------------------------------
#  Full lattice initialization
# ---------------------------------------------------------------------------

def init_bpst_instanton(L: int, rho: float,
                         x0: Optional[np.ndarray] = None,
                         self_dual: bool = True,
                         n_sub: int = 16) -> np.ndarray:
    """Initialize L^4 lattice as singular-gauge BPST instanton.

    Parameters
    ----------
    L : int
    rho : float, instanton size in lattice units
    x0 : (4,) centre. Default: half-integer offset to avoid singularity.
    self_dual : bool
        True: Q > 0 (self-dual F)
        False: Q < 0 (anti-self-dual F)
    n_sub : int, sub-links for path integration
    """
    if x0 is None:
        x0 = np.array([L / 2.0 - 0.5] * 4)

    coords = np.arange(L, dtype=np.float64)
    grid = np.stack(np.meshgrid(coords, coords, coords, coords,
                                 indexing='ij'), axis=-1)

    duality_str = "SD" if self_dual else "ASD"
    U = np.zeros((L, L, L, L, 4, 2, 2), dtype=np.complex128)
    for mu in range(4):
        print(f"  Building links mu={mu} (singular, {duality_str}, "
              f"rho={rho:.2f}) ... ", end="", flush=True)
        U[..., mu, :, :] = path_ordered_link(
            grid, mu, x0, rho, float(L),
            self_dual=self_dual, n_sub=n_sub)
        print("done.")
    return U


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------

def _self_test():
    from itertools import permutations

    # Verify self-duality of eta symbols
    eps = np.zeros((4,4,4,4))
    for p in permutations(range(4)):
        plist = list(p)
        s = 0
        for i in range(4):
            while plist[i] != i:
                j = plist[i]
                plist[i], plist[j] = plist[j], plist[i]
                s += 1
        eps[p] = 1.0 if s % 2 == 0 else -1.0

    for a in range(3):
        for mu in range(4):
            for nu in range(4):
                d = sum(0.5*eps[mu,nu,r,s]*ETA_SD[a,r,s]
                        for r in range(4) for s in range(4))
                assert abs(d - ETA_SD[a,mu,nu]) < 1e-14
                d2 = sum(0.5*eps[mu,nu,r,s]*ETA_ASD[a,r,s]
                         for r in range(4) for s in range(4))
                assert abs(d2 + ETA_ASD[a,mu,nu]) < 1e-14
    print("[PASS] 't Hooft symbols verified.")

    # Test both dualities on 6^4
    from src.observables import topological_charge
    from src.lattice import unitarity_check

    for sd in [True, False]:
        tag = "SD" if sd else "ASD"
        U = init_bpst_instanton(6, rho=2.5, self_dual=sd, n_sub=16)
        Q = topological_charge(U)
        un = unitarity_check(U)
        print(f"  {tag}: Q={Q:+.4f}, unitarity={un:.2e}")
        if sd:
            assert Q > 0, f"SD instanton should have Q > 0, got {Q}"
        else:
            assert Q < 0, f"ASD instanton should have Q < 0, got {Q}"
    print("[PASS] SD/ASD duality correct.")


if __name__ == "__main__":
    _self_test()
