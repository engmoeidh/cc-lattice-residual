#!/usr/bin/env python3
r"""
mode_diagnostic.py
==================
Classify the lowest ~25 eigenmodes of Delta_1 on both
12^3x8 (rho=3) and 18^3x12 (rho=4.5).

For each eigenmode v_i, compute:
  1. lambda_i (eigenvalue)
  2. IPR_i = sum |v(x)|^4 / (sum |v(x)|^2)^2  (inverse participation ratio)
  3. Core fraction C_i(R) for R = rho, 2*rho
  4. Overlap with 5 physical LT tangents: O^phys_{iI}
  5. Gauge-divergence norm: |D.v| / |v|
  6. Overlap with pure-gauge orientation directions

Decision rule:
  - Physical collective mode: core-localized, small div, large phys overlap
  - Gauge artifact: core-localized, large gauge overlap
  - Lattice artifact: extended, oscillatory, weak core fraction
  - Physical non-collective: core-localized, small phys AND gauge overlap
"""
import sys, time
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy import sparse
sys.path.insert(0, '.')

from src.caloron import init_caloron
from src.observables import full_diagnostic
from src.spectral import build_adjoint_laplacian, build_vector_operator
from jacobian_exact import covariant_divergence, gauge_clean_tangent
from jacobian_left_triv import raw_tangent_translation_LT, raw_tangent_scale_LT


def compute_site_norm_sq(v, V):
    """Compute |v(x)|^2 for each site x. v is 12V-dimensional."""
    v_reshaped = v.reshape(V, 12)
    return np.sum(v_reshaped**2, axis=1)


def ipr(v, V):
    """Inverse participation ratio."""
    norms_sq = compute_site_norm_sq(v, V)
    return np.sum(norms_sq**2) / np.sum(norms_sq)**2


def core_fraction(v, V, dims, x0, R):
    """Fraction of |v|^2 within radius R of x0."""
    norms_sq = compute_site_norm_sq(v, V)
    Lt, Lx, Ly, Lz = dims
    total = np.sum(norms_sq)
    if total < 1e-30:
        return 0.0
    core = 0.0
    idx = 0
    for t in range(Lt):
        for x in range(Lx):
            for y in range(Ly):
                for z in range(Lz):
                    # Periodic distance
                    dt = min(abs(t - x0[0]), Lt - abs(t - x0[0]))
                    dx = min(abs(x - x0[1]), Lx - abs(x - x0[1]))
                    dy = min(abs(y - x0[2]), Ly - abs(y - x0[2]))
                    dz = min(abs(z - x0[3]), Lz - abs(z - x0[3]))
                    r = np.sqrt(dt**2 + dx**2 + dy**2 + dz**2)
                    if r < R:
                        core += norms_sq[idx]
                    idx += 1
    return core / total


def run_diagnostic(L_s, L_t, rho, x0, n_eig=25, mass_sq=0.01):
    """Run the full mode classification on one lattice."""

    print(f"\n{'='*70}")
    print(f"MODE DIAGNOSTIC: {L_s}^3 x {L_t}, rho={rho}")
    print(f"{'='*70}")

    U = init_caloron(L_s, L_t, rho=rho, x0=x0, self_dual=True, n_sub=16)
    full_diagnostic(U, f"{L_s}^3x{L_t}")
    dims = U.shape[:4]
    V = int(np.prod(dims))

    # Build Delta_1
    print(f"  Building Delta_1...")
    Delta1 = build_vector_operator(U, mass_sq=mass_sq)

    # Find lowest eigenpairs
    print(f"  Finding {n_eig} lowest eigenpairs of Delta_1...")
    t0 = time.time()
    vals, vecs = eigsh(Delta1, k=n_eig, which='SM')
    order = np.argsort(np.abs(vals))
    vals = vals[order]
    vecs = vecs[:, order]
    print(f"  Done in {time.time()-t0:.0f}s")

    # Build 5 physical tangents
    print(f"  Building 5 physical LT tangents...")
    raw = []
    for mu in range(4):
        Y = raw_tangent_translation_LT(U, L_s, L_t, rho, x0, mu, 0.1,
                                        self_dual=True, n_sub=16)
        raw.append(Y)
    Y = raw_tangent_scale_LT(U, L_s, L_t, rho, x0, 0.05,
                              self_dual=True, n_sub=16)
    raw.append(Y)

    cleaned = []
    for I in range(5):
        Z, _, info = gauge_clean_tangent(U, raw[I], mass_sq=1e-6,
                                          cg_tol=1e-10, cg_maxiter=5000)
        cleaned.append(Z)

    phys_vecs = [t.reshape(12*V) for t in cleaned]
    phys_ortho = []
    for v in phys_vecs:
        w = v.copy()
        for u in phys_ortho:
            w -= np.dot(u, w) * u
        n = np.linalg.norm(w)
        if n > 1e-10:
            phys_ortho.append(w / n)

    # Build 3 pure-gauge orientation directions
    # Y^{pg,a}_mu(x) = Ad(U_mu(x)) e_a - e_a
    # In adjoint rep for SU(2): Ad(U) e_a = R_{ab} e_b
    # where R is the SO(3) rotation matrix from U
    print(f"  Building 3 orientation directions...")
    gauge_dirs = []
    for a in range(3):
        Y_pg = np.zeros(12*V)
        for x_flat in range(V):
            # For each link direction mu
            for mu in range(4):
                # Get U_mu(x) as 2x2 matrix
                idx = np.unravel_index(x_flat, dims)
                U_link = U[idx[0], idx[1], idx[2], idx[3], mu]

                # Adjoint action: Ad(U) t_a = U t_a U^dag
                # t_a = sigma_a / (2i) in our convention
                sigma = [np.array([[0,1],[1,0]], dtype=complex),
                         np.array([[0,-1j],[1j,0]], dtype=complex),
                         np.array([[1,0],[0,-1]], dtype=complex)]

                t_a = sigma[a] / (2j)
                Ad_ta = U_link @ t_a @ U_link.conj().T
                diff = Ad_ta - t_a  # This is the pure-gauge perturbation

                # Extract adjoint components: diff = sum_b c_b t_b
                for b in range(3):
                    # Tr(t_b^dag . diff) / Tr(t_b^dag . t_b)
                    cb = np.trace(sigma[b].conj().T / (-2j) @ diff)
                    cb = cb.real  # Should be real for SU(2)
                    Y_pg[12*x_flat + 3*mu + b] = cb

        norm = np.linalg.norm(Y_pg)
        if norm > 1e-10:
            Y_pg /= norm
        gauge_dirs.append(Y_pg)

    # Orthonormalize gauge dirs
    gauge_ortho = []
    for v in gauge_dirs:
        w = v.copy()
        for u in gauge_ortho:
            w -= np.dot(u, w) * u
        n = np.linalg.norm(w)
        if n > 1e-10:
            gauge_ortho.append(w / n)

    # Classify each mode
    print(f"\n  {'i':>3} {'lambda':>10} {'IPR':>10} {'C(rho)':>8} {'C(2rho)':>8} "
          f"{'O_phys':>8} {'O_gauge':>8} {'class':>15}")
    print(f"  {'-'*75}")

    results = []
    for i in range(n_eig):
        v = vecs[:, i]
        lam = vals[i]

        # IPR
        ipr_val = ipr(v, V) * V  # Normalize so IPR*V ~ 1 for localized

        # Core fractions
        cf_rho = core_fraction(v, V, dims, x0, rho)
        cf_2rho = core_fraction(v, V, dims, x0, 2*rho)

        # Overlap with physical tangents
        o_phys = sum(np.dot(u, v)**2 for u in phys_ortho)

        # Overlap with gauge orientations
        o_gauge = sum(np.dot(u, v)**2 for u in gauge_ortho)

        # Classification
        if o_phys > 0.3:
            cls = "PHYS-COLL"
        elif o_gauge > 0.1:
            cls = "GAUGE-ORIENT"
        elif cf_2rho > 0.5 and abs(lam) < 0.02:
            cls = "CORE-NEARZERO"
        elif cf_2rho < 0.1:
            cls = "EXTENDED"
        elif abs(lam) < 0.06:
            cls = "NEAR-ZERO"
        else:
            cls = "BULK"

        print(f"  {i:>3} {lam:>10.6f} {ipr_val:>10.2f} {cf_rho:>8.3f} {cf_2rho:>8.3f} "
              f"{o_phys:>8.4f} {o_gauge:>8.4f} {cls:>15}")

        results.append({
            'i': i, 'lambda': lam, 'ipr': ipr_val,
            'cf_rho': cf_rho, 'cf_2rho': cf_2rho,
            'o_phys': o_phys, 'o_gauge': o_gauge,
            'class': cls
        })

    # Summary
    n_phys = sum(1 for r in results if r['class'] == 'PHYS-COLL')
    n_gauge = sum(1 for r in results if r['class'] == 'GAUGE-ORIENT')
    n_core_nz = sum(1 for r in results if r['class'] == 'CORE-NEARZERO')
    n_ext = sum(1 for r in results if r['class'] == 'EXTENDED')
    n_nz = sum(1 for r in results if r['class'] == 'NEAR-ZERO')
    n_bulk = sum(1 for r in results if r['class'] == 'BULK')

    print(f"\n  CLASSIFICATION SUMMARY:")
    print(f"    PHYS-COLL (physical collective):  {n_phys}")
    print(f"    GAUGE-ORIENT (gauge orientation): {n_gauge}")
    print(f"    CORE-NEARZERO (core-localized):   {n_core_nz}")
    print(f"    NEAR-ZERO (other near-zero):      {n_nz}")
    print(f"    EXTENDED (delocalized):            {n_ext}")
    print(f"    BULK (above gap):                  {n_bulk}")

    # Near-zero block analysis
    nz_modes = [r for r in results if abs(r['lambda']) < 0.06]
    if nz_modes:
        print(f"\n  NEAR-ZERO BLOCK ({len(nz_modes)} modes below |lambda|<0.06):")
        for r in nz_modes:
            print(f"    mode {r['i']}: lam={r['lambda']:.6f}, "
                  f"C(2rho)={r['cf_2rho']:.3f}, "
                  f"O_phys={r['o_phys']:.4f}, O_gauge={r['o_gauge']:.4f}, "
                  f"class={r['class']}")

    return results


if __name__ == "__main__":
    print("Running mode diagnostic on both lattices...\n")

    # Row 1: 12^3x8, rho=3 (baseline)
    r12 = run_diagnostic(12, 8, 3.0, [3.5, 5.5, 5.5, 5.5], n_eig=25)

    # Row 2: 18^3x12, rho=4.5 (refinement)
    r18 = run_diagnostic(18, 12, 4.5, [5.5, 8.5, 8.5, 8.5], n_eig=25)

    # Cross-comparison
    print(f"\n\n{'='*70}")
    print("CROSS-COMPARISON: NEAR-ZERO SECTOR")
    print(f"{'='*70}")

    for label, results in [("12^3x8 rho=3", r12), ("18^3x12 rho=4.5", r18)]:
        nz = [r for r in results if abs(r['lambda']) < 0.06]
        core_nz = [r for r in nz if r['cf_2rho'] > 0.3]
        ext_nz = [r for r in nz if r['cf_2rho'] < 0.1]

        print(f"\n  {label}:")
        print(f"    Total near-zero modes: {len(nz)}")
        print(f"    Core-localized (C(2rho)>0.3): {len(core_nz)}")
        print(f"    Extended (C(2rho)<0.1): {len(ext_nz)}")
        if core_nz:
            lams = [r['lambda'] for r in core_nz]
            o_phys = [r['o_phys'] for r in core_nz]
            o_gauge = [r['o_gauge'] for r in core_nz]
            print(f"    Core-localized eigenvalues: {[f'{l:.6f}' for l in lams]}")
            print(f"    Mean phys overlap: {np.mean(o_phys):.4f}")
            print(f"    Mean gauge overlap: {np.mean(o_gauge):.4f}")

    print(f"\n\n{'='*70}")
    print("DECISION")
    print(f"{'='*70}")
    print("""
  If the compressed 8-mode block on 18^3x12 is:
    (A) core-localized + large gauge/phys overlap
        => under-subtraction of continuum zero-mode sector
        => upgrade prime from 5 to 8 modes
    (B) core-localized + small gauge overlap + small phys overlap
        => genuinely physical non-collective modes
        => branch response really grows under refinement
    (C) extended + weak core fraction
        => lattice artifacts, exclude from prime
""")
