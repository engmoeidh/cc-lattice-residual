#!/usr/bin/env python3
"""
lattice_diagnostics_rowsABC.py
===============================

Produces the per-row lattice diagnostics for Table 1b of CC_2.tex:
  1. Topological charge Q_cl (clover)
  2. Self-duality violation delta_+
  3. Average plaquette
  4. n_sub-stability of I_cl (16 -> 32)
  5. m^2-regulator dependence of delta_xi log zeta (Row A only,
     leading-order first-order response)
  6. Admissibility check

All values printed to stdout with full audit trail. Cached to
results/lattice_diagnostics_rowsABC.txt for archive inclusion.

Runtime estimate:
  - Diagnostics 1-4 per row: ~30-60 s on CPU
  - Diagnostic 5 (Row A only): ~2 min on GPU
  - Total: ~5 min wall time

Usage:
  python scripts/lattice_diagnostics_rowsABC.py 2>&1 | tee results/lattice_diagnostics_rowsABC.txt
"""
import os
import sys
import time
import json
import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(REPO_ROOT))

from src.caloron import init_caloron
from src.observables import (
    topological_charge,
    self_duality_violation,
    admissibility_check,
    full_diagnostic,
)
from src.lattice import average_plaquette

# Row definitions from CC_2.tex Section 3.2
ROWS = [
    {"label": "A", "L_spatial": 18, "L_temporal": 12, "rho": 3.0},
    {"label": "B", "L_spatial": 24, "L_temporal": 16, "rho": 3.5},
    {"label": "C", "L_spatial": 32, "L_temporal": 24, "rho": 4.5},
]

# Parameters for Diagnostic 5 (m^2 dependence)
M_SQ_STANDARD = 0.01
M_SQ_HALVED   = 0.005
XI_TEST       = 1.045        # One off-anchor node; not the full Simpson


def diag_1_through_4(row):
    """Cheap per-row diagnostics: Q, delta_+, plaquette, n_sub-stability,
    admissibility. All CPU, seconds per call."""
    L_s = row["L_spatial"]
    L_t = row["L_temporal"]
    rho = row["rho"]
    label = row["label"]

    print(f"\n{'='*70}")
    print(f" Row {label}: L_s={L_s}, L_t={L_t}, rho={rho}")
    print(f"{'='*70}")

    # Diagnostic 1-3: Build at standard n_sub=16
    print(f"\n  [1] Standard caloron build (n_sub=16) ...")
    t0 = time.time()
    U_16 = init_caloron(L_spatial=L_s, L_temporal=L_t, rho=rho, n_sub=16)
    print(f"      done ({time.time()-t0:.1f} s). shape: {U_16.shape}")

    # Topological charge (clover)
    Q_cl = topological_charge(U_16)
    print(f"      Q_cl (clover, summed over all sites)   = {Q_cl:+.6f}")

    # Self-duality violation
    delta_plus = self_duality_violation(U_16)
    print(f"      delta_+ (self-duality violation)        = {delta_plus:.3e}")

    # Average plaquette
    P_avg = average_plaquette(U_16)
    print(f"      <P> (average plaquette)                 = {P_avg:.6f}")

    # Admissibility
    adm = admissibility_check(U_16)
    print(f"      Admissibility (plaquette bound, e=0.01) = {adm}")

    # Full diagnostic dump for I_cl
    diag = full_diagnostic(U_16, label=label)
    I_cl_16 = diag.get('I_cl', None)
    if I_cl_16 is None:
        I_cl_16 = diag.get('I_W', None)
    print(f"      I_cl (from full_diagnostic)             = {I_cl_16:+.6f}")

    # Diagnostic 4: n_sub stability
    print(f"\n  [4] n_sub stability: rebuilding at n_sub=32 ...")
    t1 = time.time()
    U_32 = init_caloron(L_spatial=L_s, L_temporal=L_t, rho=rho, n_sub=32)
    print(f"      done ({time.time()-t1:.1f} s)")

    diag_32 = full_diagnostic(U_32, label=label + "_nsub32")
    I_cl_32 = diag_32.get('I_cl', None)
    if I_cl_32 is None:
        I_cl_32 = diag_32.get('I_W', None)
    rel_change_nsub = abs(I_cl_32 - I_cl_16) / abs(I_cl_16)
    print(f"      I_cl(n_sub=16) = {I_cl_16:+.6f}")
    print(f"      I_cl(n_sub=32) = {I_cl_32:+.6f}")
    print(f"      |Delta I_cl|/|I_cl|                    = {rel_change_nsub:.3e}")

    return {
        "label": label,
        "L_s": L_s, "L_t": L_t, "rho": rho,
        "Q_cl": float(Q_cl),
        "I_cl_16": float(I_cl_16),
        "I_cl_32": float(I_cl_32),
        "delta_plus": float(delta_plus),
        "plaquette_avg": float(P_avg),
        "nsub_stability": float(rel_change_nsub),
        "admissibility": adm,
        "full_diag": diag,
    }, U_16


def diag_5_m2_dependence(U_row_A):
    """Row A only: first-order m^2 dependence of delta_xi log zeta.

    Computes the leading-order response T(xi=1) = Tr'[Delta_1^{-1} K_0]
    at two values of m^2 and reports the relative change.

    This uses a stochastic trace estimator with N_S noise vectors.
    Runtime dominated by CG solves inside Hutchinson estimator.
    """
    print(f"\n{'='*70}")
    print(f" [5] m^2 dependence (Row A, first-order response)")
    print(f"{'='*70}")

    from src.spectral import build_vector_operator, build_adjoint_laplacian
    import scipy.sparse as sps
    import scipy.sparse.linalg as spla

    # Build temporal Laplacian K_0 (mass-free, temporal-only, 12V-embedded)
    L_s, L_t = U_row_A.shape[1], U_row_A.shape[0]
    V = L_t * L_s**3

    def embed_scalar_in_vector(K_scalar, V):
        K_coo = K_scalar.tocoo()
        x_s, a_s = divmod(K_coo.row, 3)
        y_s, b_s = divmod(K_coo.col, 3)
        rows_v = np.concatenate([12*x_s + 3*mu + a_s for mu in range(4)])
        cols_v = np.concatenate([12*y_s + 3*mu + b_s for mu in range(4)])
        data_v = np.concatenate([K_coo.data for _ in range(4)])
        return sps.coo_matrix((data_v, (rows_v, cols_v)),
                              shape=(12*V, 12*V)).tocsr()

    print(f"\n  Building K_0 (temporal adjoint Laplacian) ...")
    t0 = time.time()
    K_0_scalar = build_adjoint_laplacian(U_row_A, mass_sq=0.0, temporal_only=True)
    K_0 = embed_scalar_in_vector(K_0_scalar, V)
    print(f"  done ({time.time()-t0:.1f} s)")

    # Hutchinson trace estimator for Tr[Delta_1^{-1} K_0]
    N_S = 20       # fewer samples than production — this is a diagnostic
    CG_TOL = 1e-10

    def first_order_T(m_sq):
        """Compute T = Tr[Delta_1(m_sq)^{-1} K_0] stochastically."""
        print(f"\n    Building Delta_1 at m^2 = {m_sq} ...")
        t1 = time.time()
        D = build_vector_operator(U_row_A, mass_sq=m_sq)
        print(f"    done ({time.time()-t1:.1f} s). dim = {D.shape[0]}")

        # Stochastic trace: T ~ (1/N_s) sum_i <z_i | K_0 Delta^{-1} z_i>
        # Use Z2 noise for variance reduction
        rng = np.random.default_rng(42)  # PAIRED seed
        N = D.shape[0]
        T_samples = np.zeros(N_S)

        print(f"    Computing T = Tr[Delta_1^{{-1}} K_0] with N_S={N_S} ...")
        t2 = time.time()
        for i in range(N_S):
            z = rng.choice([-1.0, 1.0], size=N)
            # solve Delta_1 y = K_0 z
            rhs = K_0 @ z
            y, info = spla.cg(D, rhs, rtol=CG_TOL, maxiter=2000)
            if info != 0:
                print(f"      WARNING: CG did not converge (info={info}) on sample {i}")
            T_samples[i] = np.dot(z, y)
            if (i + 1) % 5 == 0:
                print(f"      sample {i+1}/{N_S}: partial mean = {T_samples[:i+1].mean():.4f}")

        T_mean = T_samples.mean()
        T_err = T_samples.std(ddof=1) / np.sqrt(N_S)
        print(f"    T({m_sq:.4f}) = {T_mean:+.4f} +/- {T_err:.4f} "
              f"({time.time()-t2:.1f} s)")
        return T_mean, T_err

    # Standard m^2
    T_std, T_std_err = first_order_T(M_SQ_STANDARD)

    # Halved m^2
    T_halved, T_halved_err = first_order_T(M_SQ_HALVED)

    # Report relative change
    rel_change = abs(T_halved - T_std) / abs(T_std) if abs(T_std) > 0 else float('inf')
    # First-order response: dlz/eps ~ -T + ... so relative change of T
    # approximates relative change of dlz at fixed eps.
    print(f"\n  m^2 = {M_SQ_STANDARD}: T = {T_std:+.4f}")
    print(f"  m^2 = {M_SQ_HALVED}: T = {T_halved:+.4f}")
    print(f"  |Delta T|/|T| (first-order response)     = {rel_change:.3e}")

    return {
        "T_m2_standard": float(T_std),
        "T_m2_standard_err": float(T_std_err),
        "T_m2_halved": float(T_halved),
        "T_m2_halved_err": float(T_halved_err),
        "m2_dependence_first_order": float(rel_change),
    }


def main():
    t_start = time.time()

    print("=" * 70)
    print(" LATTICE DIAGNOSTICS FOR TABLE 1b (CC_2.tex)")
    print("=" * 70)
    print(f" Rows: A (18^3 x 12, rho=3.0)")
    print(f"       B (24^3 x 16, rho=3.5)")
    print(f"       C (32^3 x 24, rho=4.5)")
    print(f"\n Diagnostics:")
    print(f"   1. Q_cl (topological charge, clover)")
    print(f"   2. delta_+ (self-duality violation)")
    print(f"   3. <P> (average plaquette)")
    print(f"   4. n_sub stability (16 -> 32)")
    print(f"   5. m^2 regulator dependence (Row A only)")

    # Per-row diagnostics 1-4
    all_results = []
    U_rowA_for_diag5 = None
    for row in ROWS:
        result, U = diag_1_through_4(row)
        all_results.append(result)
        if row["label"] == "A":
            U_rowA_for_diag5 = U

    # Diagnostic 5 on Row A only
    diag5_result = diag_5_m2_dependence(U_rowA_for_diag5)

    # Final table print
    print(f"\n{'='*70}")
    print(" TABLE 1b SUMMARY (values for CC_2.tex)")
    print(f"{'='*70}")
    header = f"  {'Row':<4} {'I_cl':>9} {'Q_cl':>9} {'delta_+':>11} {'<P>':>9} {'n_sub stab':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        print(f"  {r['label']:<4} "
              f"{r['I_cl_16']:>+9.4f} "
              f"{r['Q_cl']:>+9.4f} "
              f"{r['delta_plus']:>11.2e} "
              f"{r['plaquette_avg']:>9.4f} "
              f"{r['nsub_stability']:>12.2e}")

    print(f"\n  m^2 dependence (Row A, first-order response):")
    print(f"    |Delta T|/|T| at xi=1.0, halving m^2 = {M_SQ_STANDARD} -> {M_SQ_HALVED}")
    print(f"    = {diag5_result['m2_dependence_first_order']:.3e}")

    print(f"\n{'='*70}")
    print(f" Total wall time: {time.time()-t_start:.1f} s")
    print(f"{'='*70}")

    # Also dump machine-readable JSON for later parsing
    output_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "lattice_diagnostics_rowsABC.json")
    # Strip non-serializable entries
    clean_results = []
    for r in all_results:
        r_clean = {k: v for k, v in r.items()
                   if k not in ("full_diag", "admissibility")}
        # Extract scalar admissibility info
        if "admissibility" in r and isinstance(r["admissibility"], dict):
            for k, v in r["admissibility"].items():
                if isinstance(v, (int, float, bool, str)):
                    r_clean[f"adm_{k}"] = v
        clean_results.append(r_clean)
    clean_results.append({"diag_5_m2_Row_A": diag5_result})

    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\n JSON cached to: {json_path}")


if __name__ == "__main__":
    main()
