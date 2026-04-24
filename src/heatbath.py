"""
PFMT CC Programme — SU(2) Heatbath Monte Carlo

The Creutz heatbath for SU(2) is EXACT: no Metropolis accept/reject
needed.  The full conditional distribution for each link is:

    P(U) ∝ exp( (beta/2) Re Tr[U . Sigma] )

where Sigma is the staple sum.  Decompose Sigma = alpha * V_hat
(alpha > 0, V_hat in SU(2)).  Then the relative orientation
W = U . V_hat^dag has quaternion scalar part a0 distributed as:

    P(a0) ∝ sqrt(1 - a0^2) * exp(k * a0),   k = beta * alpha

Sampling uses inverse-CDF for exp(k a0) with rejection for the
sqrt(1 - a0^2) factor.

Pipeline for instanton finding:
    1. Hot start
    2. Thermalise with heatbath (50-200 sweeps at beta ~ 2.3-2.5)
    3. Measure Q_clover
    4. If |Q - 1| < 0.3, found a Q=1 configuration -> flow it
    5. Otherwise, do more sweeps and try again

Reference: Creutz, Phys. Rev. D 21 (1980) 2308
           Kennedy & Pendleton, Phys. Lett. B 156 (1985) 393
"""

import numpy as np
import time
from typing import Tuple, Optional, List, Dict
from src.lattice import (
    su2_multiply, su2_dagger, su2_project, su2_trace,
    su2_identity, su2_random_uniform, su2_det,
    plaquette_staple, average_plaquette, reunitarise
)
from src.observables import topological_charge


# ===================================================================
#  CREUTZ HEATBATH FOR SU(2)
# ===================================================================

def _sample_a0(k: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a0 from P(a0) ∝ sqrt(1-a0^2) exp(k*a0) on [-1, 1].

    Method: inverse CDF for exp(k*a0), then rejection for sqrt(1-a0^2).

    For exp(k*a0) on [-1,1]:
        CDF: F(a0) = [exp(k*a0) - exp(-k)] / [exp(k) - exp(-k)]
        Inverse: a0 = (1/k) * log(exp(-k) + u * 2*sinh(k))

    Rejection: accept with prob sqrt(1 - a0^2).
    Average acceptance ~ pi/4 ~ 0.79 for small k, higher for large k.

    Parameters
    ----------
    k : ndarray, shape (...)
        Effective coupling beta * alpha at each site.
        Must be > 0.
    rng : numpy Generator

    Returns
    -------
    a0 : ndarray, shape (...), float64
        Sampled quaternion scalar parts in [-1, 1].
    """
    shape = k.shape
    result = np.zeros(shape, dtype=np.float64)
    remaining = np.ones(shape, dtype=bool)

    # Handle k ~ 0 separately (uniform on S^3 -> a0 ~ sqrt(1-a0^2))
    small_k = k < 1e-10
    if np.any(small_k):
        # For k=0: P(a0) ∝ sqrt(1-a0^2), use rejection from uniform
        n_small = np.sum(small_k)
        while True:
            candidates = 2.0 * rng.random(n_small) - 1.0
            accept = rng.random(n_small) < np.sqrt(1 - candidates**2)
            if np.all(accept):
                result[small_k] = candidates
                remaining[small_k] = False
                break
            # Partial accept
            idx_small = np.where(small_k)[0]
            for i, idx in enumerate(idx_small):
                if accept[i]:
                    result.flat[idx] = candidates[i]
                    remaining.flat[idx] = False

    # Main loop for k > 0
    max_iter = 100
    for _ in range(max_iter):
        if not np.any(remaining):
            break

        n_rem = np.sum(remaining)
        k_rem = k[remaining]

        # Sample from exp(k * a0) using inverse CDF
        u = rng.random(n_rem)
        # a0 = (1/k) * log(exp(-k) + u * 2*sinh(k))
        # For numerical stability: log(exp(-k) + u*(exp(k)-exp(-k)))
        # = log(exp(-k) * (1 + u*(exp(2k)-1)))
        # = -k + log(1 + u*(exp(2k)-1))
        # For large k: log(1 + u*exp(2k)) ≈ 2k + log(u) for u > exp(-2k)
        exp_neg_2k = np.exp(-2.0 * k_rem)
        arg = 1.0 + u * (1.0 / exp_neg_2k - 1.0)  # = 1 + u*(exp(2k)-1)
        # But 1/exp_neg_2k can overflow. Use log-sum-exp:
        # a0 = -1 + (1/k) * log1p(u * (exp(2k) - 1))
        # For moderate k (< 500): direct computation
        safe_k = np.minimum(k_rem, 500.0)
        a0_candidates = -1.0 + np.log1p(u * np.expm1(2.0 * safe_k)) / safe_k

        # Clip to [-1, 1] for safety
        a0_candidates = np.clip(a0_candidates, -1.0, 1.0)

        # Rejection: accept with prob sqrt(1 - a0^2)
        accept_prob = np.sqrt(1.0 - a0_candidates**2)
        accept = rng.random(n_rem) < accept_prob

        # Place accepted values
        idx_remaining = np.where(remaining)[0]
        for i, idx in enumerate(idx_remaining):
            if accept[i]:
                result.flat[idx] = a0_candidates[i]
                remaining.flat[idx] = False

    return result


def heatbath_sweep(U: np.ndarray, beta: float,
                    rng: np.random.Generator) -> np.ndarray:
    """One full heatbath sweep over all links.

    Updates each link (x, mu) from the exact conditional distribution.
    Sequential over directions, vectorised over sites within each
    even/odd checkerboard sublattice.

    Parameters
    ----------
    U : ndarray, shape (L,L,L,L, 4, 2, 2)
    beta : float
        Gauge coupling (beta = 4/g^2 for SU(2)).
    rng : numpy Generator

    Returns
    -------
    U_new : ndarray, shape (N0,N1,N2,N3, 4, 2, 2)
    """
    dims = U.shape[:4]
    U_new = U.copy()

    for mu in range(4):
        # Use checkerboard to avoid conflicts:
        # even sites (sum of coords is even) and odd sites updated separately
        for parity in [0, 1]:
            # Build mask for sites with given parity
            grids = [np.arange(d) for d in dims]
            grid = np.meshgrid(*grids, indexing='ij')
            parity_mask = (grid[0] + grid[1] + grid[2] + grid[3]) % 2 == parity

            # Get staple (Wilson only for heatbath — c1=0)
            sigma = plaquette_staple(U_new, mu)  # (*dims, 2, 2)

            # Extract at sites with this parity
            sigma_p = sigma[parity_mask]  # (N_sites, 2, 2)
            N_sites = sigma_p.shape[0]

            # Decompose sigma = alpha * V_hat
            # alpha = sqrt(|det(sigma)|), but for sum of SU(2) elements,
            # det can be negative.  Use: alpha = sqrt(Re det(sigma))
            # since for Hermitian-like staples, det is real and positive.
            # More robust: alpha = sqrt(|a_0|^2 + |a_1|^2 + ...) where
            # sigma = a_0 I + i a_k sigma_k
            # Actually for SU(2): Sigma = sum of SU(2) elements, so
            # Sigma = c * V where c > 0 and V in SU(2).
            # c = sqrt(det Sigma) when det > 0.
            det_sigma = su2_det(sigma_p)  # complex
            alpha = np.sqrt(np.maximum(np.real(det_sigma), 1e-30))

            # V_hat = Sigma / alpha = proj(Sigma)
            V_hat = su2_project(sigma_p)

            # Effective coupling
            k = beta * alpha  # shape (N_sites,)

            # Sample a0 from P(a0) ∝ sqrt(1-a0^2) exp(k a0)
            a0 = _sample_a0(k, rng)

            # Sample (a1, a2, a3) uniformly on sphere of radius sqrt(1-a0^2)
            r = np.sqrt(np.maximum(1.0 - a0**2, 0.0))
            # Random direction on S^2
            cos_theta = 2.0 * rng.random(N_sites) - 1.0
            sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
            phi = 2.0 * np.pi * rng.random(N_sites)

            a1 = r * sin_theta * np.cos(phi)
            a2 = r * sin_theta * np.sin(phi)
            a3 = r * cos_theta

            # Construct W = a0 I + i a_k sigma_k
            W = np.zeros((N_sites, 2, 2), dtype=np.complex128)
            W[:, 0, 0] = a0 + 1j * a3
            W[:, 0, 1] = a2 + 1j * a1
            W[:, 1, 0] = -a2 + 1j * a1
            W[:, 1, 1] = a0 - 1j * a3

            # New link: U = W . V_hat^dag
            U_sites = su2_multiply(W, su2_dagger(V_hat))

            # Place back
            U_new[..., mu, :, :][parity_mask] = U_sites

    return U_new


# ===================================================================
#  MONTE CARLO THERMALISATION + Q SCANNING
# ===================================================================

def thermalise_and_find_Q(dims, beta: float,
                           rng: np.random.Generator,
                           target_Q: int = 1,
                           Q_tolerance: float = 0.3,
                           n_thermalise: int = 100,
                           n_measure_sweeps: int = 500,
                           measure_interval: int = 5,
                           verbose: bool = True
                           ) -> Tuple[Optional[np.ndarray], List[Dict]]:
    """Thermalise and scan for a configuration with |Q| = target_Q.

    Parameters
    ----------
    dims : int or tuple
        Lattice dimensions. int L -> (L,L,L,L). (Lt,Ls) -> (Lt,Ls,Ls,Ls).
    beta : float
    rng, target_Q, Q_tolerance, n_thermalise, n_measure_sweeps,
    measure_interval, verbose : as before
    """
    from src.lattice import _parse_dims
    d = _parse_dims(dims)

    if verbose:
        if d[0]==d[1]==d[2]==d[3]:
            lat_str = f"{d[0]}^4"
        elif d[1]==d[2]==d[3]:
            lat_str = f"{d[1]}^3x{d[0]}"
        else:
            lat_str = "x".join(str(x) for x in d)
        print(f"Monte Carlo: {lat_str}, beta={beta}")
        print(f"  Target Q={target_Q}, tolerance={Q_tolerance}")
        print(f"  Thermalising for {n_thermalise} sweeps...")

    # Hot start
    U = su2_random_uniform((*d, 4), rng)

    t0 = time.time()

    # Thermalise
    for sw in range(1, n_thermalise + 1):
        U = heatbath_sweep(U, beta, rng)
        if verbose and sw % 20 == 0:
            plaq = average_plaquette(U)
            print(f"    thermalise sw={sw}: <P>={plaq:.6f}  "
                  f"[{time.time()-t0:.1f}s]")

    if verbose:
        plaq = average_plaquette(U)
        print(f"  Thermalised: <P>={plaq:.6f}")
        print(f"  Scanning for Q={target_Q}...")

    # Scan
    history = []
    for sw in range(1, n_measure_sweeps + 1):
        U = heatbath_sweep(U, beta, rng)

        if sw % measure_interval == 0:
            Q = topological_charge(U)
            plaq = average_plaquette(U)
            Q_int = int(np.round(Q))
            Q_dev = abs(Q - Q_int)

            record = {
                'sweep': n_thermalise + sw,
                'Q': Q, 'Q_int': Q_int, 'Q_dev': Q_dev,
                'plaq': plaq,
            }
            history.append(record)

            if verbose:
                print(f"    scan sw={sw}: Q={Q:+.4f}  <P>={plaq:.6f}  "
                      f"[{time.time()-t0:.1f}s]")

            if abs(Q_int) == abs(target_Q) and Q_dev < Q_tolerance:
                if verbose:
                    print(f"  *** FOUND Q={Q_int} configuration! "
                          f"Q_clover={Q:.4f}")
                return U, history

    if verbose:
        print(f"  Did not find Q={target_Q} in {n_measure_sweeps} sweeps.")
        Q_vals = [h['Q_int'] for h in history]
        from collections import Counter
        print(f"  Q distribution: {dict(Counter(Q_vals))}")

    return None, history


# ===================================================================
#  FULL PIPELINE: MC -> FLOW -> MEASURE
# ===================================================================

def mc_instanton_pipeline(L: int = 16, beta: float = 2.4,
                           rng: Optional[np.random.Generator] = None,
                           n_attempts: int = 5,
                           verbose: bool = True
                           ) -> Tuple[Optional[np.ndarray], Dict]:
    """Full pipeline to find and stabilise a lattice instanton.

    1. Thermalise with heatbath
    2. Find Q=1 configuration
    3. Apply over-improved gradient flow
    4. Check for stable instanton

    Parameters
    ----------
    L : int
    beta : float
    rng : numpy Generator (default: seeded at 42)
    n_attempts : int
        Number of independent thermalisation attempts.
    verbose : bool

    Returns
    -------
    U_final : ndarray or None
    info : dict with diagnostics
    """
    if rng is None:
        rng = np.random.default_rng(42)

    from src.cooling import run_gradient_flow
    from src.observables import full_diagnostic

    for attempt in range(1, n_attempts + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"ATTEMPT {attempt}/{n_attempts}")
            print(f"{'='*60}")

        U_mc, hist = thermalise_and_find_Q(
            L, beta, rng,
            target_Q=1, Q_tolerance=0.3,
            n_thermalise=100,
            n_measure_sweeps=500,
            measure_interval=5,
            verbose=verbose
        )

        if U_mc is None:
            continue

        # Found Q=1. Now flow it.
        if verbose:
            print(f"\n  Applying over-improved gradient flow...")
            full_diagnostic(U_mc, "pre-flow")

        U_flowed, flow_hist = run_gradient_flow(
            U_mc, c0=3.0, c1=-0.25,
            dt=0.005, n_steps=400, measure_interval=10,
            Q_target=1, Q_tolerance=0.6,
            integrator="rk3", verbose=verbose
        )

        diag = full_diagnostic(U_flowed, "post-flow")

        if diag['Q_deviation'] < 0.1 and abs(diag['Q_int']) == 1:
            if verbose:
                print(f"\n  *** INSTANTON STABILISED ***")
                print(f"  I_lat = {diag['I_lat']:.4f}")
            return U_flowed, diag

    if verbose:
        print(f"\nFailed to stabilise instanton in {n_attempts} attempts.")
    return None, {}


# ===================================================================
#  SELF-TEST
# ===================================================================

def _self_test():
    """Test heatbath sampling and thermalisation."""
    rng = np.random.default_rng(42)

    # 1. Test _sample_a0 distribution
    print("Test 1: a0 sampling distribution")
    k_test = np.full(10000, 3.0)
    a0 = _sample_a0(k_test, rng)
    print(f"  <a0> = {np.mean(a0):.4f}  (expect positive for k>0)")
    print(f"  min={np.min(a0):.4f}, max={np.max(a0):.4f}")
    assert np.all(a0 >= -1) and np.all(a0 <= 1)
    assert np.mean(a0) > 0  # biased toward +1 for k > 0
    print("  [PASS]")

    # 2. Test heatbath thermalisation on 4^4
    print("\nTest 2: Heatbath thermalisation (4^4, beta=2.4)")
    U = su2_random_uniform((4, 4, 4, 4, 4), rng)
    plaq_init = average_plaquette(U)
    print(f"  Initial <P> = {plaq_init:.4f}  (hot start, expect ~0)")

    for sw in range(1, 51):
        U = heatbath_sweep(U, beta=2.4, rng=rng)
        if sw % 10 == 0:
            plaq = average_plaquette(U)
            print(f"  sw={sw}: <P>={plaq:.6f}")

    plaq_final = average_plaquette(U)
    print(f"  Final <P> = {plaq_final:.6f}")
    # At beta=2.4, <P> ~ 0.62-0.65 for SU(2)
    assert plaq_final > 0.5, f"Plaquette too low: {plaq_final}"
    assert plaq_final < 0.8, f"Plaquette too high: {plaq_final}"
    print("  [PASS] Plaquette in expected range.")

    # 3. Q measurement on thermalised config
    print("\nTest 3: Q on thermalised 4^4")
    Q = topological_charge(U)
    print(f"  Q = {Q:.4f}")
    print("  [PASS] (Q can be anything on 4^4)")


if __name__ == "__main__":
    _self_test()
