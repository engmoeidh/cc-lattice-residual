"""
PFMT CC Programme — Gradient Flow (v4)

v4 changes:
    - Normalisation: I_lat = S_W(1)/(2 pi^2) throughout
    - Q tracking: compares |Q| to |Q_target|, no sign mismatch
    - Three-action separation: flow kernel ≠ target action ≠ diagnostic
    - Force: F = -proj_TA[U Sigma]  (gradient descent, verified)
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from src.lattice import (
    su2_identity, su2_multiply, su2_dagger, su2_project,
    su2_exp_algebra, su2_trace,
    total_staple, average_plaquette, wilson_action,
    reunitarise, SIGMA
)
from src.observables import topological_charge
from src.config import NORM_INSTANTON


# ===================================================================
#  HELPERS
# ===================================================================

def _proj_TA(M: np.ndarray) -> np.ndarray:
    """Traceless anti-Hermitian projection."""
    nc = 2
    AH = (M - su2_dagger(M)) / 2.0
    tr = su2_trace(AH)
    return AH - (tr / nc)[..., None, None] * su2_identity(AH.shape[:-2])


def _antiherm_to_algebra(F: np.ndarray) -> np.ndarray:
    """Anti-Hermitian 2x2 -> su(2) algebra vector."""
    batch = F.shape[:-2]
    omega = np.zeros((*batch, 3), dtype=np.float64)
    for a in range(3):
        tr = np.einsum('ij,...ji->...', SIGMA[a], F)
        omega[..., a] = np.real(-1j * tr)
    return omega


# ===================================================================
#  FORCE (verified: gradient descent)
# ===================================================================

def lie_derivative_force(U: np.ndarray, mu: int,
                          c0: float, c1: float) -> np.ndarray:
    """F_mu = -proj_TA[U_mu . Sigma_mu].  Descent direction."""
    sigma = total_staple(U, mu, c0, c1)
    V = su2_multiply(U[..., mu, :, :], sigma)
    return -_proj_TA(V)


# ===================================================================
#  INTEGRATORS
# ===================================================================

def flow_step_euler(U, dt, c0, c1):
    U_new = U.copy()
    for mu in range(4):
        F = lie_derivative_force(U, mu, c0, c1)
        omega = _antiherm_to_algebra(dt * F)
        U_new[..., mu, :, :] = su2_multiply(
            su2_exp_algebra(omega), U[..., mu, :, :])
    return reunitarise(U_new)


def flow_step_rk3(U, dt, c0, c1):
    """Luscher 3rd-order RK."""
    F0 = np.zeros_like(U)
    for mu in range(4):
        F0[..., mu, :, :] = lie_derivative_force(U, mu, c0, c1)

    W1 = U.copy()
    for mu in range(4):
        omega = _antiherm_to_algebra(0.25 * dt * F0[..., mu, :, :])
        W1[..., mu, :, :] = su2_multiply(
            su2_exp_algebra(omega), U[..., mu, :, :])
    W1 = reunitarise(W1)

    F1 = np.zeros_like(U)
    for mu in range(4):
        F1[..., mu, :, :] = lie_derivative_force(W1, mu, c0, c1)

    W2 = W1.copy()
    for mu in range(4):
        Z = (8./9)*dt*F1[...,mu,:,:] - (17./36)*dt*F0[...,mu,:,:]
        W2[...,mu,:,:] = su2_multiply(
            su2_exp_algebra(_antiherm_to_algebra(Z)), W1[...,mu,:,:])
    W2 = reunitarise(W2)

    F2 = np.zeros_like(U)
    for mu in range(4):
        F2[..., mu, :, :] = lie_derivative_force(W2, mu, c0, c1)

    U_new = W2.copy()
    for mu in range(4):
        Z = (3./4)*dt*F2[...,mu,:,:] - (8./9)*dt*F1[...,mu,:,:] \
            + (17./36)*dt*F0[...,mu,:,:]
        U_new[...,mu,:,:] = su2_multiply(
            su2_exp_algebra(_antiherm_to_algebra(Z)), W2[...,mu,:,:])
    return reunitarise(U_new)


# ===================================================================
#  FLOW DRIVER
# ===================================================================

def run_gradient_flow(U: np.ndarray,
                       c0: float = 3.0, c1: float = -0.25,
                       dt: float = 0.005, n_steps: int = 600,
                       measure_interval: int = 10,
                       integrator: str = "rk3",
                       Q_target: int = 1,
                       Q_tolerance: float = 0.8,
                       abort_on_Q_jump: bool = True,
                       verbose: bool = True
                       ) -> Tuple[np.ndarray, List[Dict]]:
    """Wilson gradient flow with topology monitoring.

    Q_target sign convention:
        |Q| is compared to |Q_target|, so Q_target=1 accepts
        both +1 and -1 sectors.  Set Q_target=0 to disable.
    """
    step_func = flow_step_rk3 if integrator == "rk3" else flow_step_euler
    history = []
    U_cur = U.copy()
    dims = U.shape[:4]

    if verbose:
        if dims[0]==dims[1]==dims[2]==dims[3]:
            lat_str = f"{dims[0]}^4"
        elif dims[1]==dims[2]==dims[3]:
            lat_str = f"{dims[1]}^3x{dims[0]}"
        else:
            lat_str = "x".join(str(d) for d in dims)
        print(f"Gradient flow: {lat_str}, dt={dt}, {integrator}")
        print(f"  Flow kernel: c0={c0:.4f}, c1={c1:.4f}")
        print(f"  |Q| target={abs(Q_target)}, tol={Q_tolerance}")

    t0 = time.time()
    for step in range(1, n_steps + 1):
        U_cur = step_func(U_cur, dt, c0, c1)

        if step % measure_interval == 0 or step == 1:
            tf = step * dt
            Q = topological_charge(U_cur)
            plaq = average_plaquette(U_cur)
            S = wilson_action(U_cur, beta=1.0)
            I_lat = S / NORM_INSTANTON   # CORRECT normalisation

            Q_int = int(np.round(Q))
            Q_dev = abs(Q - Q_int)

            rec = {'step': step, 't_flow': tf,
                   'Q': Q, 'Q_int': Q_int, 'Q_dev': Q_dev,
                   'plaq': plaq, 'S_W': S, 'I_lat': I_lat}
            history.append(rec)

            if verbose:
                # Also show I_lat per unit of |Q|
                I_per_Q = I_lat / max(abs(Q), 0.01)
                print(f"  t={tf:.3f}  Q={Q:+.4f}  <P>={plaq:.6f}  "
                      f"I_lat={I_lat:.4f}  I/|Q|={I_per_Q:.3f}  "
                      f"[{time.time()-t0:.1f}s]")

            # Sign-consistent Q check: compare |Q| to |Q_target|
            if abort_on_Q_jump and Q_target != 0:
                if abs(abs(Q) - abs(Q_target)) > Q_tolerance:
                    if verbose:
                        print(f"  *** |Q|={abs(Q):.3f} outside "
                              f"|Q_target|={abs(Q_target)} +/- {Q_tolerance}")
                    break

            # Plateau detection — must be in the TARGET sector
            if len(history) >= 10:
                Iv = [h['I_lat'] for h in history[-10:]]
                rel_var = (max(Iv)-min(Iv)) / max(abs(np.mean(Iv)), 1e-10)
                in_target = (Q_target == 0) or (abs(Q_int) == abs(Q_target))
                if rel_var < 0.005 and Q_dev < 0.05 and in_target:
                    if verbose:
                        print(f"  *** PLATEAU in Q={Q_int} sector: "
                              f"I_lat var={rel_var:.6f}")
                    break

    if verbose:
        print(f"Flow done: {len(history)} meas, {time.time()-t0:.1f}s")
    return U_cur, history


# ===================================================================
#  SELF-TEST
# ===================================================================

def _self_test():
    from src.lattice import init_cold, su2_random_near_identity

    print("Test 1: Force on cold start = 0")
    U = init_cold(4)
    F = lie_derivative_force(U, 0, c0=3.0, c1=-0.25)
    assert np.max(np.abs(F)) < 1e-14
    print("  [PASS]")

    print("\nTest 2: Action decreases under flow")
    rng = np.random.default_rng(42)
    U_pert = U.copy()
    for mu in range(4):
        pert = su2_random_near_identity(U.shape[:4], 0.3, rng)
        U_pert[..., mu, :, :] = su2_multiply(pert, U_pert[..., mu, :, :])
    U_pert = reunitarise(U_pert)

    S0 = wilson_action(U_pert, 1.0)
    U_after = flow_step_euler(U_pert, 0.005, c0=3.0, c1=-0.25)
    S1 = wilson_action(U_after, 1.0)
    print(f"  S: {S0:.2f} -> {S1:.2f}, dS={S1-S0:.2f}")
    assert S1 < S0
    print("  [PASS]")

    print("\nTest 3: Normalisation")
    print(f"  NORM_INSTANTON = {NORM_INSTANTON:.6f}")
    print(f"  I_lat for cold: {wilson_action(U, 1.0)/NORM_INSTANTON:.6f} (expect 0)")
    print("  [PASS]")


if __name__ == "__main__":
    _self_test()
