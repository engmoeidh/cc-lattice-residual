"""
PFMT CC Programme — Instanton Extraction via Optimal-Stopping Flow

The Q-kick paradigm (nudging Q continuously toward a target) fails
because lattice topology is DISCRETE: on a periodic lattice with
finite spacing, Q jumps between integer sectors, and a small local
perturbation cannot inject fractional topology.

CORRECT STRATEGY: Optimal-Stopping Flow
    1. Generate thermal ensemble at physical beta (MC heatbath)
    2. Flow the configuration freely (no Q constraint)
    3. Track Q(t), S(t), I_lat(t) throughout the flow
    4. Identify the OPTIMAL FLOW TIME t* where:
       (a) Q is closest to an integer ±1
       (b) S has dropped significantly from the thermal value
       (c) I_lat = S/(2π²|Q|) is minimised
    5. The configuration at t* is the best lattice approximation
       to a single instanton available at this lattice size.

This replaces the failed kick-based constrained minimiser.

STATUS: [CHECK] — physically motivated extraction, not a
theorem-grade topology-preserving saddle-point solution.
The definitive version requires geometric or index-based topology.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional

from src.lattice import (
    su2_multiply, average_plaquette, wilson_action,
    reunitarise, SIGMA
)
from src.observables import (
    topological_charge, full_diagnostic, self_duality_violation
)
from src.cooling import (
    flow_step_rk3, flow_step_euler,
)
from src.config import NORM_INSTANTON


# ===================================================================
#  OPTIMAL-STOPPING FLOW
# ===================================================================

def optimal_stopping_flow(U: np.ndarray,
                           c0: float = 3.0, c1: float = -0.25,
                           dt: float = 0.005,
                           n_steps: int = 600,
                           measure_interval: int = 5,
                           integrator: str = "rk3",
                           verbose: bool = True
                           ) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Flow freely and identify optimal extraction point.

    The optimal point is where Q is closest to ±1 with minimum
    I_lat = S/(2π² |Q|), after the initial UV transient.

    Parameters
    ----------
    U : initial (thermal) configuration
    c0, c1 : flow kernel coefficients
    dt : flow time step
    n_steps : total flow steps
    measure_interval : measure every N steps
    integrator : "euler" or "rk3"
    verbose : print progress

    Returns
    -------
    U_best : configuration at optimal flow time
    history : full flow history
    best_info : dict with optimal-point diagnostics
    """
    step_func = flow_step_rk3 if integrator == "rk3" else flow_step_euler
    history = []
    U_cur = U.copy()
    dims = U.shape[:4]

    # Track best configuration
    U_best = U.copy()
    best_score = np.inf  # lower is better
    best_info = {}

    if verbose:
        if dims[0]==dims[1]==dims[2]==dims[3]:
            lat_str = f"{dims[0]}^4"
        elif dims[1]==dims[2]==dims[3]:
            lat_str = f"{dims[1]}^3x{dims[0]}"
        else:
            lat_str = "x".join(str(d) for d in dims)
        print(f"Optimal-stopping flow: {lat_str}, dt={dt}, {integrator}")
        print(f"  Kernel: c0={c0:.4f}, c1={c1:.4f}")
        print(f"  {'step':>5s}  {'t':>6s}  {'Q':>8s}  {'I_lat':>8s}  "
              f"{'I/|Q|':>8s}  {'<P>':>10s}  {'score':>8s}")

    t0 = time.time()

    for step in range(1, n_steps + 1):
        U_cur = step_func(U_cur, dt, c0, c1)

        if step % measure_interval == 0 or step == 1:
            tf = step * dt
            Q = topological_charge(U_cur)
            S = wilson_action(U_cur, beta=1.0)
            I_lat = S / NORM_INSTANTON
            plaq = average_plaquette(U_cur)
            Q_int = int(np.round(Q))
            Q_dev = abs(Q - Q_int)

            # I_lat per unit of |Q|
            I_per_Q = I_lat / max(abs(Q), 0.01)

            rec = {
                'step': step, 't_flow': tf,
                'Q': Q, 'Q_int': Q_int, 'Q_dev': Q_dev,
                'I_lat': I_lat, 'I_per_Q': I_per_Q,
                'plaq': plaq,
            }
            history.append(rec)

            # Scoring function for optimal stopping:
            # Want |Q| close to 1, Q_dev small, I_lat small
            # Only consider after initial transient (t > 0.05)
            # and when |Q| is in the right ballpark (0.5 < |Q| < 1.5)
            if tf > 0.05 and 0.5 < abs(Q) < 1.5 and abs(Q_int) == 1:
                # Score: penalise Q_dev and reward low I_per_Q
                score = Q_dev + 0.01 * I_per_Q
            else:
                score = np.inf

            rec['score'] = score

            if verbose:
                marker = " *" if score < best_score and score < np.inf else ""
                print(f"  {step:5d}  {tf:6.3f}  {Q:+8.4f}  "
                      f"{I_lat:8.3f}  {I_per_Q:8.3f}  "
                      f"{plaq:10.6f}  {score:8.4f}{marker}")

            if score < best_score:
                best_score = score
                U_best = U_cur.copy()
                best_info = {
                    't_flow': tf, 'Q': Q, 'Q_int': Q_int,
                    'Q_dev': Q_dev, 'I_lat': I_lat,
                    'I_per_Q': I_per_Q, 'plaq': plaq,
                    'score': score,
                }

    elapsed = time.time() - t0
    if verbose:
        print(f"\nFlow done: {len(history)} measurements, {elapsed:.1f}s")
        if best_info:
            print(f"  BEST: t={best_info['t_flow']:.3f}  "
                  f"Q={best_info['Q']:+.4f}  "
                  f"I_lat={best_info['I_lat']:.3f}  "
                  f"I/|Q|={best_info['I_per_Q']:.3f}")
        else:
            print(f"  No valid Q≈±1 window found.")

    return U_best, history, best_info


# ===================================================================
#  MULTI-CONFIG SCANNER
# ===================================================================

def multi_config_scan(dims=10, beta: float = 2.5,
                       n_configs: int = 10,
                       n_thermalise: int = 100,
                       n_decorrelate: int = 10,
                       flow_steps: int = 400,
                       flow_dt: float = 0.005,
                       flow_meas: int = 10,
                       seed: int = 42,
                       verbose: bool = True
                       ) -> Tuple[Optional[np.ndarray], Dict, List]:
    """Generate multiple thermal configs, flow each, find best instanton.

    Parameters
    ----------
    dims : int or tuple
        Lattice dimensions. int L -> (L,L,L,L). (Lt,Ls) -> (Lt,Ls,Ls,Ls).
    beta : gauge coupling for MC
    (other params unchanged)
    """
    from src.heatbath import heatbath_sweep
    from src.lattice import su2_random_uniform, _parse_dims

    d = _parse_dims(dims)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    if verbose:
        if d[0]==d[1]==d[2]==d[3]:
            lat_str = f"{d[0]}^4"
        elif d[1]==d[2]==d[3]:
            lat_str = f"{d[1]}^3x{d[0]}"
        else:
            lat_str = "x".join(str(x) for x in d)
        print(f"Multi-config scan: {lat_str}, beta={beta}, "
              f"{n_configs} configs")

    # Thermalise
    U = su2_random_uniform((*d, 4), rng)
    for sw in range(1, n_thermalise + 1):
        U = heatbath_sweep(U, beta, rng)
        if verbose and sw % 50 == 0:
            plaq = average_plaquette(U)
            print(f"  thermalise sw={sw}: <P>={plaq:.6f} "
                  f"[{time.time()-t0:.0f}s]")

    if verbose:
        print(f"  Thermalised: <P>={average_plaquette(U):.6f}")

    # Scan configs
    global_best = None
    global_best_info = {}
    global_best_score = np.inf
    all_results = []

    for cfg in range(1, n_configs + 1):
        # Decorrelate
        for _ in range(n_decorrelate):
            U = heatbath_sweep(U, beta, rng)

        if verbose:
            Q_raw = topological_charge(U)
            print(f"\n--- Config {cfg}/{n_configs}: "
                  f"Q_raw={Q_raw:+.3f} ---")

        # Flow and find optimal stopping point
        U_best_cfg, hist, info = optimal_stopping_flow(
            U, c0=3.0, c1=-0.25,
            dt=flow_dt, n_steps=flow_steps,
            measure_interval=flow_meas,
            integrator="rk3",
            verbose=verbose)

        result = {
            'config': cfg,
            'best_info': info,
            'n_measurements': len(hist),
        }
        all_results.append(result)

        if info and info.get('score', np.inf) < global_best_score:
            global_best = U_best_cfg.copy()
            global_best_info = info.copy()
            global_best_info['config'] = cfg
            global_best_score = info['score']
            if verbose:
                print(f"  >>> NEW GLOBAL BEST (config {cfg})")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n{'='*60}")
        print(f"SCAN COMPLETE: {elapsed:.0f}s total")
        if global_best_info:
            print(f"  Best config: #{global_best_info.get('config','?')}")
            print(f"  Q = {global_best_info['Q']:+.4f}  "
                  f"(int: {global_best_info['Q_int']})")
            print(f"  I_lat = {global_best_info['I_lat']:.4f}")
            print(f"  I/|Q| = {global_best_info['I_per_Q']:.4f}")
            a_C = 277.0 / (8 * np.pi**2 * global_best_info['I_per_Q'])
            print(f"  => a_C (if I/|Q| were I_lat) = {a_C:.2f}")
        else:
            print(f"  No valid instanton found.")
        print(f"{'='*60}")

    return global_best, global_best_info, all_results


# ===================================================================
#  SELF-TEST
# ===================================================================

def _self_test():
    """Test optimal-stopping flow on a small BPST seed."""
    from src.instanton import init_bpst_instanton

    # 1. Flow a BPST instanton on 6^4 and find optimal stopping
    print("Test 1: Optimal-stopping on BPST 6^4 (rho=2.5)")
    U = init_bpst_instanton(6, rho=2.5, n_sub=16)
    Q0 = topological_charge(U)
    print(f"  Initial Q = {Q0:.4f}")

    U_best, hist, info = optimal_stopping_flow(
        U, c0=3.0, c1=-0.25,
        dt=0.01, n_steps=50, measure_interval=5,
        verbose=True)

    print(f"\n  Best info: {info}")
    print("  [Ran without error]")

    # 2. Verify history is non-empty and has correct fields
    assert len(hist) > 0, "No measurements recorded"
    assert 'Q' in hist[0] and 'I_lat' in hist[0]
    print("  [PASS] History structure correct.")


if __name__ == "__main__":
    _self_test()
