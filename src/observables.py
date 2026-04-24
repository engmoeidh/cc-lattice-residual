"""
PFMT CC Programme — Lattice Observables (v4: all normalisations corrected)

NORMALISATION LOCK (canonical):
    I_W   = S_W(beta=1) / (2 pi^2)       Wilson instanton factor
    I_cl  = S_cl / (4 pi^2)              Clover instanton factor
    Q     = (1/4 pi^2) sum_x [...]       Topological charge

    For a self-dual unit instanton in the continuum:  Q = I_W = I_cl = 1.

TOPOLOGICAL CHARGE — derivation:
    Q = (1/32 pi^2) int d^4x  eps_{munurhosigma} Tr(F_{munu} F_{rhosigma})

    where F = F^a T^a,  T^a = sigma^a/2,  Tr(T^a T^b) = (1/2) delta^{ab}.

    The (1/32pi^2) comes from Q = (1/16pi^2) Tr(F *F) with *F = (1/2) eps F.

    Free sum:  eps Tr(FF) = 8 * [Tr(F01 F23) - Tr(F02 F13) + Tr(F03 F12)]

    Therefore:  Q = (8/32pi^2) * [...] = (1/4pi^2) * [...]

    Verification: for BPST,  int F^a F^a = 32 pi^2.
    Q = (1/32pi^2) * eps Tr(FF) = (1/32pi^2) * (1/2) * 2 * 32pi^2 = 1.  ✓

    NOTE: previous code versions had 1/(2pi^2) which was WRONG by factor 2.
    The error was using (1/16pi^2) eps Tr(FF) instead of (1/32pi^2) eps Tr(FF).

CLOVER ACTION:
    s_cl(x) = (1/2) sum_{mu<nu} Tr(F^2)
    S_cl = sum_x s_cl(x)
    I_cl = S_cl / (4 pi^2)

    Wilson expansion: S_W(1) ~ S_cl/2 in the smooth-field limit.
    So I_W = S_W/(2pi^2) ~ S_cl/(4pi^2) = I_cl.  Bogomolny consistent.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from src.lattice import (
    su2_multiply, su2_dagger, su2_trace, su2_identity,
    shift, plaquette, average_plaquette, wilson_action
)
from src.config import NORM_INSTANTON


# ---------------------------------------------------------------------------
#  Clover field strength
# ---------------------------------------------------------------------------

def clover(U: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """Four-leaf clover in the mu-nu plane at every site.

    C_{mu,nu}(x) = Leaf_1 + Leaf_2 + Leaf_3 + Leaf_4

    Each leaf is a 1x1 plaquette loop starting and ending at x,
    circulating counterclockwise.
    """
    assert mu != nu

    # Leaf 1: +mu, +nu, -mu, -nu
    L1 = su2_multiply(
        U[..., mu, :, :],
        shift(U[..., nu, :, :], mu, +1))
    L1 = su2_multiply(L1, su2_dagger(shift(U[..., mu, :, :], nu, +1)))
    L1 = su2_multiply(L1, su2_dagger(U[..., nu, :, :]))

    # Leaf 2: +nu, -mu, -nu, +mu
    L2 = su2_multiply(
        U[..., nu, :, :],
        su2_dagger(shift(shift(U[..., mu, :, :], nu, +1), mu, -1)))
    L2 = su2_multiply(L2, su2_dagger(shift(U[..., nu, :, :], mu, -1)))
    L2 = su2_multiply(L2, shift(U[..., mu, :, :], mu, -1))

    # Leaf 3: -mu, -nu, +mu, +nu
    L3 = su2_multiply(
        su2_dagger(shift(U[..., mu, :, :], mu, -1)),
        su2_dagger(shift(shift(U[..., nu, :, :], mu, -1), nu, -1)))
    L3 = su2_multiply(L3, shift(shift(U[..., mu, :, :], mu, -1), nu, -1))
    L3 = su2_multiply(L3, shift(U[..., nu, :, :], nu, -1))

    # Leaf 4: -nu, +mu, +nu, -mu
    L4 = su2_multiply(
        su2_dagger(shift(U[..., nu, :, :], nu, -1)),
        shift(U[..., mu, :, :], nu, -1))
    L4 = su2_multiply(L4, shift(shift(U[..., nu, :, :], mu, +1), nu, -1))
    L4 = su2_multiply(L4, su2_dagger(U[..., mu, :, :]))

    return L1 + L2 + L3 + L4


def field_strength(U: np.ndarray, mu: int, nu: int) -> np.ndarray:
    """Clover field strength F_{mu,nu}(x) = (C - C^dag)/(8i).

    Hermitian, traceless, O(a^2) accurate.
    """
    C = clover(U, mu, nu)
    return -1j * (C - su2_dagger(C)) / 8.0


# ---------------------------------------------------------------------------
#  Topological charge  (normalisation: 1/(4 pi^2), CORRECTED v4)
# ---------------------------------------------------------------------------

def topological_charge_density(U: np.ndarray) -> np.ndarray:
    """q(x) = (1/(4 pi^2)) [Tr(F01 F23) - Tr(F02 F13) + Tr(F03 F12)]

    where Tr is the 2x2 matrix trace and F = F^a sigma^a/2.

    Derivation:
        Q = (1/32 pi^2) eps_{munurhosigma} Tr(F_{munu} F_{rhosigma})  [free sum]
        Free sum = 8 * [three dual pair terms]
        Q = (8 / 32 pi^2) * [...] = (1 / 4 pi^2) * [...]

    NOTE: the v2 code had 1/(2 pi^2) which was WRONG by factor 2.
    The error was: using (1/16pi^2) instead of (1/32pi^2) in the
    starting formula.  The (1/16pi^2) form uses Tr(F *F) which has
    an extra (1/2) from *F = (1/2) eps F.
    """
    F01 = field_strength(U, 0, 1)
    F02 = field_strength(U, 0, 2)
    F03 = field_strength(U, 0, 3)
    F12 = field_strength(U, 1, 2)
    F13 = field_strength(U, 1, 3)
    F23 = field_strength(U, 2, 3)

    t1 = np.real(su2_trace(su2_multiply(F01, F23)))
    t2 = np.real(su2_trace(su2_multiply(F02, F13)))
    t3 = np.real(su2_trace(su2_multiply(F03, F12)))

    # CORRECTED: 1/(4 pi^2), not 1/(2 pi^2)
    return (1.0 / (4.0 * np.pi**2)) * (t1 - t2 + t3)


def topological_charge(U: np.ndarray) -> float:
    """Q = sum_x q(x).  Integer for smooth fields."""
    return float(np.sum(topological_charge_density(U)))


# ---------------------------------------------------------------------------
#  Action density
# ---------------------------------------------------------------------------

def action_density_clover(U: np.ndarray) -> np.ndarray:
    """s(x) = (1/2) sum_{mu<nu} Tr(F^2)."""
    s = np.zeros(U.shape[:4], dtype=np.float64)
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F = field_strength(U, mu, nu)
            s += 0.5 * np.real(su2_trace(su2_multiply(F, F)))
    return s


# ---------------------------------------------------------------------------
#  Self-duality
# ---------------------------------------------------------------------------

def self_duality_violation(U: np.ndarray) -> float:
    """delta_+ = sum |F - *F|^2 / (4 sum |F|^2).

    Correctly normalised:
        0   = perfectly self-dual  (F = *F)
        1   = perfectly anti-self-dual (F = -*F, so F-*F = 2F, |2F|^2 = 4|F|^2)
        0.5 = no definite duality (random)

    Dual map (eps_{0123}=+1):
        *F_{01} = +F_{23},  *F_{02} = -F_{13},  *F_{03} = +F_{12}
    """
    F = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F[(mu, nu)] = field_strength(U, mu, nu)

    dual_map = {
        (0, 1): ((2, 3), +1.0),
        (0, 2): ((1, 3), -1.0),
        (0, 3): ((1, 2), +1.0),
        (2, 3): ((0, 1), +1.0),
        (1, 3): ((0, 2), -1.0),
        (1, 2): ((0, 3), +1.0),
    }

    num = 0.0  # sum |F - *F|^2
    den = 0.0  # sum |F|^2
    for (mu, nu), Fmn in F.items():
        (rho, sig), sign = dual_map[(mu, nu)]
        Fdual = sign * F[(rho, sig)]
        diff = Fmn - Fdual
        num += np.sum(np.real(su2_trace(su2_multiply(diff, diff))))
        den += np.sum(np.real(su2_trace(su2_multiply(Fmn, Fmn))))

    if abs(den) < 1e-30:
        return 0.5
    # Factor of 4: for ASD, |F - *F|^2 = |2F|^2 = 4|F|^2
    return float(num / (4.0 * den))


def self_duality_ratio(U: np.ndarray) -> float:
    """R_SD = 1 - delta_+.

    R_SD = 1 for self-dual, 0 for anti-self-dual, 0.5 for random.
    Kept for backward compatibility; prefer self_duality_violation.
    """
    return 1.0 - self_duality_violation(U)


# ---------------------------------------------------------------------------
#  Admissibility check for exact topology
# ---------------------------------------------------------------------------

def admissibility_check(U: np.ndarray) -> Dict:
    """Check Lüscher admissibility for exact topological sector.

    For SU(2), if ALL plaquettes satisfy:
        ||I - P_{mu,nu}(x)||_F < 1/sqrt(5) ≈ 0.447

    then an exact geometric topological charge Q_geo ∈ Z exists and
    is stable under continuous deformations within the admissible set.
    (Lüscher, Comm. Math. Phys. 85, 1982)

    IMPORTANT: Admissibility proves Q_geo ∈ Z exists.  It does NOT
    by itself prove Q_geo = round(Q_clover).  That requires explicit
    computation of Q_geo (geometric construction) or Q_index (overlap).

    The practical inference Q_geo = round(Q_clover) is heuristically
    strong when the seed is continuously connected to a known continuum
    solution, but it is not the content of the admissibility theorem.

    Returns
    -------
    result : dict with keys:
        'max_deviation' : max over all plaquettes of ||I - P||_F
        'avg_deviation' : average ||I - P||_F
        'admissible' : bool (True if max < threshold)
        'threshold' : the admissibility threshold used
        'Q_exact' : int or None (round(Q_clover) if admissible)
        'Q_clover' : float (the clover Q value)
        'Q_confidence' : str ('EXACT', 'LIKELY', or 'UNRELIABLE')
    """
    nc = 2
    # Compute ||I - P||_F for every plaquette
    max_dev = 0.0
    sum_dev = 0.0
    n_plaq = 0

    for mu in range(4):
        for nu in range(mu + 1, 4):
            P = plaquette(U, mu, nu)  # (..., 2, 2)
            I = su2_identity(P.shape[:-2])
            diff = I - P
            frob = np.sqrt(np.sum(np.abs(diff)**2, axis=(-2, -1)))
            max_dev = max(max_dev, float(np.max(frob)))
            sum_dev += float(np.sum(frob))
            n_plaq += frob.size

    avg_dev = sum_dev / n_plaq

    # Lüscher threshold for SU(2): delta < 1/sqrt(5)
    # Use a tighter practical threshold for confidence
    threshold_exact = 0.447      # 1/sqrt(5), Lüscher bound
    threshold_likely = 1.0       # practical "probably OK"

    Q_clover = topological_charge(U)
    Q_rounded = int(np.round(Q_clover))

    if max_dev < threshold_exact:
        admissible = True
        Q_exact = Q_rounded
        confidence = "EXACT"
    elif max_dev < threshold_likely:
        admissible = False
        Q_exact = Q_rounded
        confidence = "LIKELY"
    else:
        admissible = False
        Q_exact = None
        confidence = "UNRELIABLE"

    return {
        'max_deviation': max_dev,
        'avg_deviation': avg_dev,
        'admissible': admissible,
        'threshold': threshold_exact,
        'Q_exact': Q_exact,
        'Q_clover': Q_clover,
        'Q_rounded': Q_rounded,
        'Q_confidence': confidence,
    }


# ---------------------------------------------------------------------------
#  Instanton profile and size
# ---------------------------------------------------------------------------

def instanton_profile(U: np.ndarray,
                       x0: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Radial action-density profile s(r) around x0.
    Handles anisotropic lattices via per-axis minimum-image distance.
    """
    dims = U.shape[:4]  # (N0, N1, N2, N3)
    s = action_density_clover(U)
    if x0 is None:
        idx = np.unravel_index(np.argmax(s), s.shape)
        x0 = np.array(idx, dtype=np.float64)

    grids = [np.arange(d, dtype=np.float64) for d in dims]
    grid = np.stack(np.meshgrid(*grids, indexing='ij'), axis=-1)
    dx = grid - x0
    for mu in range(4):
        dx[..., mu] = dx[..., mu] - dims[mu] * np.round(dx[..., mu] / dims[mu])
    r = np.sqrt(np.sum(dx**2, axis=-1))

    r_max = np.sqrt(sum((d/2.0)**2 for d in dims))
    n_bins = int(np.ceil(r_max)) + 1
    r_flat = r.ravel()
    s_flat = s.ravel()
    s_bins = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    bin_idx = np.clip(np.round(r_flat).astype(int), 0, n_bins - 1)
    np.add.at(s_bins, bin_idx, s_flat)
    np.add.at(counts, bin_idx, 1)
    mask = counts > 0
    s_bins[mask] /= counts[mask]
    r_bins = np.arange(n_bins, dtype=np.float64)
    return r_bins[mask], s_bins[mask]


def fit_instanton_size(r_bins: np.ndarray, s_bins: np.ndarray,
                        rho_range: Tuple[float, float] = (1.0, 10.0),
                        n_grid: int = 200) -> Dict:
    """Fit s(r) to A rho^4 / (r^2 + rho^2)^4."""
    rho_grid = np.linspace(rho_range[0], rho_range[1], n_grid)
    best_chi2, best_rho, best_A = np.inf, rho_grid[0], 1.0
    w = np.maximum(s_bins, 1e-30)
    for rho in rho_grid:
        t = rho**4 / (r_bins**2 + rho**2)**4
        num = np.sum(w * s_bins * t)
        den = np.sum(w * t**2)
        if den < 1e-30:
            continue
        A = num / den
        chi2 = np.sum(w * (s_bins - A * t)**2)
        if chi2 < best_chi2:
            best_chi2, best_rho, best_A = chi2, rho, A
    return {'rho': best_rho, 'chi2': best_chi2, 'amplitude': best_A}


# ---------------------------------------------------------------------------
#  Full diagnostic  (CORRECTED normalisation)
# ---------------------------------------------------------------------------

def full_diagnostic(U: np.ndarray, label: str = "") -> Dict:
    """All observables with error decomposition.

    Reports four key quantities:
        I_W      = S_W(β=1) / (2π²)          Wilson action normalised
        I_clover = Σ s_clover / (2π²)         Clover action normalised
        Q_clover                               Topological charge (clover)
        δ₊                                     Self-duality violation

    For a perfect self-dual unit instanton:  I_W = I_clover = Q = 1, δ₊ = 0.

    Error decomposition:
        Δ_SD  = |I_clover - Q_clover|   self-duality + clover discretisation
        Δ_act = |I_W - I_clover|         Wilson vs clover action discrepancy
    """
    from src.lattice import unitarity_check, lattice_dims, lattice_volume

    dims = lattice_dims(U)
    V = lattice_volume(U)
    plaq = average_plaquette(U)
    Q = topological_charge(U)
    Q_int = int(np.round(Q))
    Q_dev = abs(Q - Q_int)

    S_W = wilson_action(U, beta=1.0)
    I_W = S_W / NORM_INSTANTON

    # Clover action: I_cl = S_cl / (4 pi^2)
    # This normalization ensures Bogomolny: Q = I_W = I_cl for self-dual.
    # 
    # Derivation: S_W = (1/8) Σ_{free} Tr(F²) and S_cl = (1/4) Σ_{free} Tr(F²)
    # so S_W = S_cl/2.  With I_W = S_W/(2π²) and I_cl = S_cl/(4π²):
    # I_W = S_cl/(4π²) = I_cl.  ✓
    s_cl = action_density_clover(U)
    S_clover = float(np.sum(s_cl))
    I_clover = S_clover / (4.0 * np.pi**2)   # NOT 2π²

    delta_SD = self_duality_violation(U)
    unit = unitarity_check(U)

    # Error decomposition
    Delta_SD = abs(I_clover - Q)      # self-duality error
    Delta_act = abs(I_W - I_clover)   # action discretisation error

    diag = {
        'dims': dims, 'volume': V, 'avg_plaq': plaq,
        'Q': Q, 'Q_int': Q_int, 'Q_deviation': Q_dev,
        'S_W_beta1': S_W,
        'I_W': I_W,                    # S_W(1) / (2π²)
        'I_lat': I_W,                  # alias for backward compat
        'I_clover': I_clover,          # Σ s_cl / (2π²)
        'delta_SD': delta_SD,
        'Delta_SD': Delta_SD,          # |I_clover - Q|
        'Delta_act': Delta_act,        # |I_W - I_clover|
        'unitarity': unit,
    }

    if abs(Q_int) >= 1 and Q_dev < 0.3:
        r_bins, s_bins = instanton_profile(U)
        r_max = min(d / 2.0 for d in dims)
        fit = fit_instanton_size(r_bins, s_bins, rho_range=(1.0, r_max))
        diag['rho_fit'] = fit['rho']

    # Format dims for display
    if dims[0] == dims[1] == dims[2] == dims[3]:
        dims_str = f"{dims[0]}^4"
    elif dims[1] == dims[2] == dims[3]:
        dims_str = f"{dims[1]}^3 x {dims[0]}"
    else:
        dims_str = f"{dims[0]}x{dims[1]}x{dims[2]}x{dims[3]}"

    hdr = f"=== Diagnostic{' [' + label + ']' if label else ''} ==="
    print(hdr)
    print(f"  lattice        = {dims_str}  (V={V})")
    print(f"  <plaq>         = {plaq:.8f}")
    print(f"  Q_clover       = {Q:.6f}  (int: {Q_int})")
    print(f"  I_W            = {I_W:.6f}   [Wilson, target: 1.0]")
    print(f"  I_clover       = {I_clover:.6f}   [clover, target: 1.0]")
    print(f"  delta_+        = {delta_SD:.6f}   [0=SD, 1=ASD]")
    print(f"  --- error decomposition ---")
    print(f"  |I_cl - Q|     = {Delta_SD:.6f}   [SD + clover discr.]")
    print(f"  |I_W - I_cl|   = {Delta_act:.6f}   [Wilson vs clover]")
    print(f"  unitarity      = {unit:.2e}")
    if 'rho_fit' in diag:
        print(f"  rho (fit)      = {diag['rho_fit']:.2f}")
    print(hdr.replace("=", "-"))
    return diag


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------

def _self_test():
    from src.lattice import init_cold

    print("Test 1: Cold start")
    U = init_cold(4)
    Q = topological_charge(U)
    print(f"  Q = {Q:.10f}  (expect 0)")
    assert abs(Q) < 1e-12
    s = action_density_clover(U)
    assert np.max(np.abs(s)) < 1e-12
    dSD = self_duality_violation(U)
    print(f"  delta_SD = {dSD:.4f}  (expect 0.5 for trivial)")
    print("  [PASS]")

    print("\nTest 2: Normalisation check")
    print(f"  NORM_INSTANTON = 2 pi^2 = {NORM_INSTANTON:.6f}")
    print(f"  For a unit instanton:  S_W(1) -> 2 pi^2 = {2*np.pi**2:.4f}")
    print(f"  I_lat = S_W(1)/(2 pi^2) -> 1.0")
    print("  [INFO]")


if __name__ == "__main__":
    _self_test()
