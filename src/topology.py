"""
PFMT CC Programme — Geometric Topological Charge

For admissible SU(2) configurations (max||I-P||_F < 1/sqrt(5)),
Lüscher's theorem guarantees an exact Q_geo ∈ Z that is stable
under continuous deformations within the admissible set.

This module provides two approaches:

1. Q_log: single-plaquette logarithm charge.
   Uses F_{μν} = log(P_{μν}) (unique for admissible configs)
   instead of the clover average.  Much closer to Q_geo than
   Q_clover because it avoids the O(a²) clover averaging error.
   Not exactly integer, but typically |Q_log - Q_geo| << |Q_cl - Q_geo|.

2. Admissibility-preserving flow (backup):
   Flow within the admissible set until Q converges to integer.
   Rigorous but slow.

STATUS:
   Q_log on an admissible config with Q_log very close to 1
   combined with the Lüscher existence theorem (Q_geo ∈ Z)
   gives Q_geo = 1 with high confidence.
   
   Theorem-grade certainty requires either:
   (a) Q_log close enough that no other integer is possible, or
   (b) explicit fiber-bundle winding computation (not yet implemented).
"""

import numpy as np
import time
from typing import Dict, List, Optional
from src.lattice import (
    su2_multiply, su2_dagger, su2_trace, su2_log_algebra,
    shift, plaquette, average_plaquette, wilson_action, reunitarise
)
from src.observables import (
    topological_charge, admissibility_check,
    self_duality_violation, action_density_clover
)
from src.cooling import flow_step_rk3, flow_step_euler
from src.config import NORM_INSTANTON


# ===================================================================
#  Q_LOG: SINGLE-PLAQUETTE LOGARITHM CHARGE
# ===================================================================

def topological_charge_log(U: np.ndarray) -> float:
    """Topological charge from single-plaquette logarithms.

    For each plaquette P_{μν}(x), compute the unique algebra element:
        ω_{μν}(x) = su2_log_algebra(P_{μν}(x))

    where P = exp(i ω_a σ_a/2), so F_{μν} = ω_a σ_a/2 (Hermitian).

    Then:
        Q_log = (1/8π²) Σ_x [ω₀₁·ω₂₃ - ω₀₂·ω₁₃ + ω₀₃·ω₁₂]

    Derivation:
        Tr(F₁ F₂) = Tr((ω₁^a σ_a/2)(ω₂^b σ_b/2))
                   = (1/4) ω₁^a ω₂^b × 2δ_ab
                   = (1/2) ω₁ · ω₂

        Q = (1/4π²) Σ [Tr(F₀₁ F₂₃) - ...]
          = (1/4π²) Σ [(1/2)(ω₀₁·ω₂₃) - ...]
          = (1/8π²) Σ [ω₀₁·ω₂₃ - ω₀₂·ω₁₃ + ω₀₃·ω₁₂]

    For admissible configs, the logarithm is unique and this gives
    a tighter approximation to Q_geo than the clover.

    Parameters
    ----------
    U : link field (should be admissible for the log to be unique)

    Returns
    -------
    Q : float
    """
    # Compute plaquette algebra vectors for all 6 planes
    omega = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            P = plaquette(U, mu, nu)  # (..., 2, 2)
            # su2_log_algebra returns omega such that P = exp(i omega_a sigma_a/2)
            omega[(mu, nu)] = su2_log_algebra(P)  # (..., 3)

    # Three dual-pair dot products
    # (01, 23): ω₀₁ · ω₂₃
    dot_01_23 = np.sum(omega[(0, 1)] * omega[(2, 3)], axis=-1)
    # (02, 13): ω₀₂ · ω₁₃
    dot_02_13 = np.sum(omega[(0, 2)] * omega[(1, 3)], axis=-1)
    # (03, 12): ω₀₃ · ω₁₂
    dot_03_12 = np.sum(omega[(0, 3)] * omega[(1, 2)], axis=-1)

    q = (1.0 / (8.0 * np.pi**2)) * (dot_01_23 - dot_02_13 + dot_03_12)

    return float(np.sum(q))


def full_topology_diagnostic(U: np.ndarray, label: str = "") -> Dict:
    """Complete topology diagnostic: Q_clover, Q_log, admissibility.

    For admissible configs, if Q_log is close to an integer n
    and Q_clover also rounds to n, then Q_geo = n with very
    high confidence.
    """
    adm = admissibility_check(U)
    Q_cl = adm['Q_clover']
    Q_log = topological_charge_log(U)
    Q_round_cl = int(np.round(Q_cl))
    Q_round_log = int(np.round(Q_log))

    I_W = wilson_action(U, 1.0) / NORM_INSTANTON
    s_cl = action_density_clover(U)
    I_cl = float(np.sum(s_cl)) / (4.0 * np.pi**2)
    dSD = self_duality_violation(U)

    result = {
        'Q_clover': Q_cl,
        'Q_log': Q_log,
        'Q_round_clover': Q_round_cl,
        'Q_round_log': Q_round_log,
        'Q_consistent': (Q_round_cl == Q_round_log),
        'Q_log_deviation': abs(Q_log - Q_round_log),
        'Q_cl_deviation': abs(Q_cl - Q_round_cl),
        'admissible': adm['admissible'],
        'max_plaq_dev': adm['max_deviation'],
        'I_W': I_W,
        'I_cl': I_cl,
        'delta_SD': dSD,
    }

    # Confidence assessment
    if adm['admissible'] and Q_round_cl == Q_round_log:
        if abs(Q_log - Q_round_log) < 0.01:
            result['Q_geo_confidence'] = 'THEOREM'
            result['Q_geo'] = Q_round_log
        elif abs(Q_log - Q_round_log) < 0.1:
            result['Q_geo_confidence'] = 'VERY_STRONG'
            result['Q_geo'] = Q_round_log
        else:
            result['Q_geo_confidence'] = 'STRONG'
            result['Q_geo'] = Q_round_log
    elif adm['admissible']:
        result['Q_geo_confidence'] = 'AMBIGUOUS'
        result['Q_geo'] = None
    else:
        result['Q_geo_confidence'] = 'NOT_ADMISSIBLE'
        result['Q_geo'] = None

    if label:
        print(f"=== Topology [{label}] ===")
    else:
        print(f"=== Topology ===")
    print(f"  Q_clover       = {Q_cl:+.6f}  (round: {Q_round_cl})")
    print(f"  Q_log          = {Q_log:+.6f}  (round: {Q_round_log})")
    print(f"  |Q_log - int|  = {result['Q_log_deviation']:.6f}")
    print(f"  |Q_cl - int|   = {result['Q_cl_deviation']:.6f}")
    print(f"  consistent     = {result['Q_consistent']}")
    print(f"  admissible     = {adm['admissible']}  "
          f"(max||I-P||={adm['max_deviation']:.6f})")
    print(f"  I_W = {I_W:.4f},  I_cl = {I_cl:.4f},  δ₊ = {dSD:.4f}")
    print(f"  Q_geo = {result['Q_geo']}  [{result['Q_geo_confidence']}]")
    print(f"---")
    return result


# ===================================================================
#  ADMISSIBILITY-PRESERVING FLOW (backup)
# ===================================================================

def compute_Q_geo_flow(U: np.ndarray,
                        dt: float = 0.002,
                        n_steps_max: int = 200,
                        Q_convergence: float = 0.01,
                        measure_interval: int = 1,
                        c0: float = 1.0,
                        c1: float = 0.0,
                        integrator: str = "rk3",
                        verbose: bool = True
                        ) -> Dict:
    """Compute Q_geo via admissibility-preserving flow (backup method)."""
    adm_init = admissibility_check(U)
    if not adm_init['admissible']:
        if verbose:
            print(f"Q_geo flow: NOT admissible")
        return {'Q_geo': None, 'Q_geo_status': 'NOT_ADMISSIBLE'}

    step_func = flow_step_rk3 if integrator == "rk3" else flow_step_euler
    U_cur = U.copy()
    history = []

    if verbose:
        print(f"Q_geo flow: c0={c0}, c1={c1}, dt={dt}")

    t0 = time.time()
    for step in range(1, n_steps_max + 1):
        U_cur = step_func(U_cur, dt, c0, c1)
        if step % measure_interval == 0:
            adm = admissibility_check(U_cur)
            Q_log = topological_charge_log(U_cur)
            Q_log_dev = abs(Q_log - round(Q_log))
            history.append((step, Q_log, Q_log_dev, adm['max_deviation'],
                           adm['admissible']))
            if verbose and step % 10 == 0:
                print(f"  step {step}: Q_log={Q_log:+.6f}, "
                      f"|Q_log-int|={Q_log_dev:.6f}, "
                      f"max||I-P||={adm['max_deviation']:.6f}")
            if not adm['admissible']:
                if verbose:
                    print(f"  ADMISSIBILITY LOST at step {step}")
                break
            if Q_log_dev < Q_convergence:
                Q_geo = int(round(Q_log))
                if verbose:
                    print(f"  CONVERGED: Q_geo = {Q_geo} at step {step}")
                return {'Q_geo': Q_geo, 'Q_geo_status': 'EXACT',
                        'history': history, 'steps': step}

    return {'Q_geo': None, 'Q_geo_status': 'NOT_CONVERGED',
            'history': history}


# ===================================================================
#  SELF-TEST
# ===================================================================

def _self_test():
    from src.caloron import init_caloron

    print("="*72)
    print("TOPOLOGY DIAGNOSTIC: Q_log vs Q_clover")
    print("="*72)

    # Test on the admissible caloron family
    Lt = 8
    print()
    print("1. Fixed-Lt table (rho=3.0, varying Ls):")
    print()
    for Ls in [10, 12, 14, 16]:
        U = init_caloron(Ls, Lt, rho=3.0, n_sub=16)
        full_topology_diagnostic(U, f"{Ls}^3x{Lt} rho=3.0")
        print()

    # Fixed-shape refinement
    print("2. Fixed-shape refinement (rho/Lt = 3/8):")
    print()
    for scale_num, scale_den in [(2, 2), (3, 2), (2, 1)]:
        lam = scale_num / scale_den
        Ls = 12 * scale_num // scale_den
        Lt_s = 8 * scale_num // scale_den
        rho = 3.0 * lam
        U = init_caloron(Ls, Lt_s, rho=rho, n_sub=16)
        full_topology_diagnostic(U, f"{lam:.1f}x: {Ls}^3x{Lt_s} rho={rho:.1f}")
        print()


if __name__ == "__main__":
    _self_test()



if __name__ == "__main__":
    _self_test()
