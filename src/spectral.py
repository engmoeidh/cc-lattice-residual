"""
PFMT CC Programme — Spectral Evaluation of Branch Response

Computes the one-loop determinant variation δ_ξ log ζ numerically
on the lattice caloron background.

The key traces are:
    Ghost:   Tr[Δ₀⁻¹ D₀²]      where Δ₀ = -D²  (adjoint scalar)
    Vector:  Tr'[Δ₁⁻¹ D₀²]     where Δ₁ = -D²δ + 2F  (adjoint vector)

These are evaluated via stochastic trace estimation:
    Tr[A⁻¹ B] ≈ (1/N) Σ η† (A⁻¹ B η)

where each sample requires one sparse CG solve.

The "temporal fraction" f₀ = Tr[Δ⁻¹ D₀²] / Tr[Δ⁻¹ D²] measures
the anisotropy of the inverse operator. For O(4)-symmetric backgrounds
f₀ = 1/4. The deviation (f₀ - 1/4) drives the branch response.
"""

import numpy as np
import time
from typing import Dict, Tuple
from scipy import sparse
from scipy.sparse.linalg import cg


# ===================================================================
#  ADJOINT REPRESENTATION
# ===================================================================

def adjoint_matrix(U: np.ndarray) -> np.ndarray:
    """3x3 adjoint representation of SU(2) element U.

    adj(U)^{ab} = (1/2) Tr(σ_a U σ_b U†)

    Parameters
    ----------
    U : (..., 2, 2) SU(2) matrices

    Returns
    -------
    R : (..., 3, 3) real orthogonal matrices
    """
    from src.lattice import SIGMA

    batch = U.shape[:-2]
    Ud = np.conj(np.swapaxes(U, -2, -1))
    R = np.zeros((*batch, 3, 3), dtype=np.float64)

    for a in range(3):
        for b in range(3):
            # Tr(σ_a U σ_b U†) = Tr(σ_a (U σ_b U†))
            # UσU† first
            M = np.einsum('...ij,...jk,...lk->...il', U, SIGMA[b], np.conj(U))
            # Then Tr(σ_a M)
            R[..., a, b] = 0.5 * np.real(
                np.einsum('ij,...ji->...', SIGMA[a], M))

    return R


# ===================================================================
#  LATTICE COVARIANT LAPLACIAN (ADJOINT)
# ===================================================================

def build_adjoint_laplacian(U: np.ndarray,
                              mass_sq: float = 0.0,
                              temporal_only: bool = False
                              ) -> sparse.csr_matrix:
    """Build the adjoint covariant Laplacian as a sparse matrix.

    Δ = -D² - m² = -Σ_μ [R_μ(x) shift_+ + R_μ†(x-μ) shift_- - 2] - m²

    In matrix form: Δ_{xa, yb} = ...

    Parameters
    ----------
    U : (N0, N1, N2, N3, 4, 2, 2) link field
    mass_sq : mass regulator (positive for stability)
    temporal_only : if True, only include μ=0 (for D₀²)

    Returns
    -------
    M : sparse CSR matrix, shape (3V, 3V)
    """
    dims = U.shape[:4]
    V = int(np.prod(dims))
    N = 3 * V  # adjoint dim × volume

    # Site indexing: site (n0,n1,n2,n3) → flat index
    # Flat index = n0*N1*N2*N3 + n1*N2*N3 + n2*N3 + n3
    N0, N1, N2, N3 = dims
    strides = np.array([N1*N2*N3, N2*N3, N3, 1])

    # Build COO lists
    rows = []
    cols = []
    vals = []

    mu_range = [0] if temporal_only else range(4)

    # Precompute all adjoint matrices
    R = {}
    for mu in mu_range:
        R[mu] = adjoint_matrix(U[..., mu, :, :])  # (N0,N1,N2,N3, 3,3)

    # For each site and direction, add hopping terms
    for mu in mu_range:
        R_mu = R[mu].reshape(V, 3, 3)

        # Forward shift: x → x+μ
        # Build index map for shift in direction mu
        shift_fwd = np.arange(V).reshape(dims)
        shift_fwd = np.roll(shift_fwd, -1, axis=mu).ravel()

        # Backward shift: x → x-μ
        shift_bwd = np.arange(V).reshape(dims)
        shift_bwd = np.roll(shift_bwd, +1, axis=mu).ravel()

        # Adjoint matrix at x-μ for backward hop
        R_mu_bwd = R[mu].reshape(V, 3, 3)
        R_mu_bwd = R_mu_bwd[shift_bwd]  # R_μ(x-μ)

        for x in range(V):
            for a in range(3):
                row = 3 * x + a

                # Forward hopping: -R_μ(x)^{ab} at site x+μ
                y_fwd = shift_fwd[x]
                for b in range(3):
                    col = 3 * y_fwd + b
                    val = -R_mu[x, a, b]
                    if abs(val) > 1e-15:
                        rows.append(row)
                        cols.append(col)
                        vals.append(val)

                # Backward hopping: -R_μ†(x-μ)^{ab} = -R_μ(x-μ)^{ba} at site x-μ
                y_bwd = shift_bwd[x]
                for b in range(3):
                    col = 3 * y_bwd + b
                    val = -R_mu_bwd[x, b, a]  # transpose for R†
                    if abs(val) > 1e-15:
                        rows.append(row)
                        cols.append(col)
                        vals.append(val)

                # Diagonal: +2 (for each direction)
                rows.append(row)
                cols.append(row)
                vals.append(2.0)

    # Add mass term
    if mass_sq != 0.0:
        for i in range(N):
            rows.append(i)
            cols.append(i)
            vals.append(mass_sq)

    M = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N))
    return M.tocsr()


# ===================================================================
#  STOCHASTIC TRACE ESTIMATION
# ===================================================================

def stochastic_trace(A_solve, B_apply, dim: int,
                      n_samples: int = 30,
                      rng: np.random.Generator = None,
                      verbose: bool = False) -> Tuple[float, float]:
    """Estimate Tr[A⁻¹ B] via stochastic trace estimation.

    Tr[A⁻¹ B] ≈ (1/N) Σ η† (A⁻¹ B η)

    Parameters
    ----------
    A_solve : callable(rhs) -> solution of A x = rhs
    B_apply : callable(v) -> B v
    dim : dimension of the space
    n_samples : number of random vectors
    rng : random generator

    Returns
    -------
    (mean, stderr) : estimate and standard error
    """
    if rng is None:
        rng = np.random.default_rng(42)

    samples = []
    for k in range(n_samples):
        # Random Z2 noise vector: entries ±1
        eta = rng.choice([-1.0, 1.0], size=dim)

        # Apply B
        Beta = B_apply(eta)

        # Solve A z = B eta
        z = A_solve(Beta)

        # Sample = η† z
        s = np.dot(eta, z)
        samples.append(s)

        if verbose and (k + 1) % 10 == 0:
            running_mean = np.mean(samples)
            running_std = np.std(samples) / np.sqrt(len(samples))
            print(f"    sample {k+1}/{n_samples}: "
                  f"running mean = {running_mean:.4f} ± {running_std:.4f}")

    samples = np.array(samples)
    return float(np.mean(samples)), float(np.std(samples) / np.sqrt(n_samples))


# ===================================================================
#  BRANCH RESPONSE COMPUTATION
# ===================================================================

def compute_ghost_response(U: np.ndarray,
                            mass_sq: float = 0.01,
                            n_samples: int = 30,
                            cg_tol: float = 1e-8,
                            verbose: bool = True) -> Dict:
    """Compute the ghost contribution to the branch response.

    Ghost response = 2ε × Tr[Δ₀⁻¹ D₀²]

    where Δ₀ = -D² + m² (with mass regulator) and D₀² is the
    temporal part of the covariant Laplacian.

    Also computes the "temporal fraction":
        f₀ = Tr[Δ₀⁻¹ D₀²] / Tr[Δ₀⁻¹ D²]

    For O(4)-symmetric backgrounds: f₀ = 1/4.
    For the caloron (O(3) × Z): f₀ ≠ 1/4 in general.

    Parameters
    ----------
    U : link field (should be admissible caloron)
    mass_sq : mass regulator (positive)
    n_samples : stochastic samples
    cg_tol : CG tolerance
    verbose : print progress
    """
    dims = U.shape[:4]
    V = int(np.prod(dims))
    N = 3 * V

    if verbose:
        print(f"Ghost response: {dims[1]}^3x{dims[0]}, "
              f"m²={mass_sq}, {n_samples} samples")
        print(f"  Building Δ₀ = -D² + m² ({N}x{N} sparse)...", end="", flush=True)

    t0 = time.time()

    # Build full Laplacian Δ₀ = -D² + m²
    Delta = build_adjoint_laplacian(U, mass_sq=mass_sq, temporal_only=False)

    # Build temporal Laplacian D₀² (just μ=0 part, no mass)
    D0sq = build_adjoint_laplacian(U, mass_sq=0.0, temporal_only=True)
    # D0sq currently has +2 on diagonal for temporal direction only
    # The actual -D₀² has this sign; we want D₀² = -(the temporal Laplacian)
    # Our build gives -D₀² + 0 (diagonal part absorbed). Actually:
    # build_adjoint_laplacian with temporal_only=True gives:
    #   M = -hop_fwd - hop_bwd + 2  (for mu=0 only)
    # This IS -D₀² (the temporal covariant Laplacian, positive semidefinite).

    if verbose:
        print(f" done ({time.time()-t0:.1f}s)")
        print(f"  Δ₀ nnz = {Delta.nnz}, D₀² nnz = {D0sq.nnz}")

    rng = np.random.default_rng(42)

    # CG solver for Δ₀
    n_cg_iters = [0]  # mutable counter

    def solve_Delta(rhs):
        n_cg_iters[0] += 1
        try:
            sol, info = cg(Delta, rhs, atol=cg_tol, maxiter=5000)
        except TypeError:
            sol, info = cg(Delta, rhs, tol=cg_tol, maxiter=5000)
        if info != 0 and verbose:
            print(f"    CG warning: info={info}")
        return sol

    # 1. Tr[Δ₀⁻¹ D₀²]
    if verbose:
        print(f"  Computing Tr[Δ₀⁻¹ D₀²]...")

    def apply_D0sq(v):
        return D0sq.dot(v)

    tr_temporal, err_temporal = stochastic_trace(
        solve_Delta, apply_D0sq, N,
        n_samples=n_samples, rng=rng, verbose=verbose)

    # 2. Tr[Δ₀⁻¹ Δ₀] = Tr[I] = N (as a check / normalisation)
    #    Actually Tr[Δ₀⁻¹ D²] where D² = Δ₀ - m² = full Laplacian
    #    So Tr[Δ₀⁻¹ (Δ₀ - m²)] = N - m² Tr[Δ₀⁻¹]
    #    For m² << eigenvalues: ≈ N

    if verbose:
        print(f"  Computing Tr[Δ₀⁻¹ D²] (full Laplacian)...")

    # Full spatial Laplacian = Δ₀ - m² I
    Dfull = build_adjoint_laplacian(U, mass_sq=0.0, temporal_only=False)

    def apply_Dfull(v):
        return Dfull.dot(v)

    tr_full, err_full = stochastic_trace(
        solve_Delta, apply_Dfull, N,
        n_samples=n_samples, rng=rng, verbose=verbose)

    # Temporal fraction
    f0 = tr_temporal / tr_full if abs(tr_full) > 1e-10 else 0.25

    if verbose:
        print(f"\n  Results:")
        print(f"    Tr[Δ₀⁻¹ D₀²]   = {tr_temporal:.4f} ± {err_temporal:.4f}")
        print(f"    Tr[Δ₀⁻¹ D²]    = {tr_full:.4f} ± {err_full:.4f}")
        print(f"    f₀ = temporal fraction = {f0:.6f}")
        print(f"    Isotropic value: f₀ = 0.25")
        print(f"    Anisotropy: f₀ - 0.25 = {f0 - 0.25:+.6f}")
        print(f"    CG solves: {n_cg_iters[0]}")
        print(f"    Total time: {time.time()-t0:.1f}s")

    return {
        'tr_temporal': tr_temporal,
        'tr_temporal_err': err_temporal,
        'tr_full': tr_full,
        'tr_full_err': err_full,
        'f0': f0,
        'f0_minus_quarter': f0 - 0.25,
        'N': N,
        'mass_sq': mass_sq,
        'n_cg_iters': n_cg_iters[0],
    }


# ===================================================================
#  FIELD STRENGTH COMPONENTS (adjoint)
# ===================================================================

def field_strength_adjoint(U: np.ndarray) -> np.ndarray:
    """Extract adjoint field strength F_μν^a(x) at every site.

    F_μν = (C - C†)/(8i)  (clover, Hermitian 2x2)
    F_μν^a = Tr(σ^a F_μν)  (adjoint components, real)

    Returns
    -------
    F : (N0,N1,N2,N3, 6, 3) — 6 planes (01,02,03,12,13,23), 3 colors
    """
    from src.observables import field_strength
    from src.lattice import SIGMA

    dims = U.shape[:4]
    planes = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    F = np.zeros((*dims, 6, 3), dtype=np.float64)

    for p, (mu, nu) in enumerate(planes):
        Fmn = field_strength(U, mu, nu)  # (..., 2, 2) Hermitian
        for a in range(3):
            # F^a = Tr(σ^a F)
            F[..., p, a] = np.real(
                np.einsum('ij,...ji->...', SIGMA[a], Fmn))

    return F


# ===================================================================
#  VECTOR FLUCTUATION OPERATOR
# ===================================================================

def build_vector_operator(U: np.ndarray,
                           mass_sq: float = 0.0) -> sparse.csr_matrix:
    """Build the vector fluctuation operator Δ₁ as a sparse matrix.

    Δ₁ = -D² δ_{μν} + 2 ad(F_{μν})  + m²

    Acts on adjoint-valued 1-forms: dimension = 4 × 3 × V.
    Index ordering: flat = 12*x + 3*μ + a

    The -D² part is block-diagonal in μ (4 copies of the ghost Laplacian).
    The 2F part couples different μ indices at the same site.
    """
    dims = U.shape[:4]
    V = int(np.prod(dims))
    N = 12 * V  # 4 Lorentz × 3 adjoint × volume

    # --- Part 1: -D² block-diagonal in μ ---
    # Build one copy of the adjoint Laplacian
    ghost_lap = build_adjoint_laplacian(U, mass_sq=mass_sq, temporal_only=False)

    # Replicate 4 times with offset
    rows_D = []
    cols_D = []
    vals_D = []

    ghost_coo = ghost_lap.tocoo()
    for mu in range(4):
        offset = 3 * V * mu  # shift for this Lorentz block
        # Map ghost indices (3*x + a) to vector indices (12*x + 3*mu + a)
        # ghost index g = 3*x + a → x = g // 3, a = g % 3
        # vector index v = 12*x + 3*mu + a = 4*(3*x + a) + ... no.
        # vector index: for site x, direction mu, color a:
        #   v = 12*x + 3*mu + a
        # ghost index: g = 3*x + a
        # Mapping: g → v = 4*g + mu ... wait.
        #   g = 3*x + a → x = g//3, a = g%3
        #   v = 12*(g//3) + 3*mu + (g%3) = 12*x + 3*mu + a

        for k in range(ghost_coo.nnz):
            g_row = ghost_coo.row[k]  # 3*x_r + a_r
            g_col = ghost_coo.col[k]  # 3*x_c + a_c
            x_r, a_r = divmod(g_row, 3)
            x_c, a_c = divmod(g_col, 3)
            v_row = 12 * x_r + 3 * mu + a_r
            v_col = 12 * x_c + 3 * mu + a_c
            rows_D.append(v_row)
            cols_D.append(v_col)
            vals_D.append(ghost_coo.data[k])

    # --- Part 2: 2F coupling ---
    # At each site x: M_{μa, νb} = -2 ε^{abc} F_{μν}^c(x)
    # (antisymmetric in a,b AND in μ,ν → symmetric in the pair)

    F_adj = field_strength_adjoint(U)  # (dims..., 6, 3)
    F_flat = F_adj.reshape(V, 6, 3)

    # Plane index mapping
    plane_idx = {}
    planes = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for p, (mu, nu) in enumerate(planes):
        plane_idx[(mu, nu)] = (p, +1)
        plane_idx[(nu, mu)] = (p, -1)  # F_{νμ} = -F_{μν}

    # ε tensor for SU(2)
    eps = np.zeros((3, 3, 3), dtype=np.float64)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = +1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

    rows_F = []
    cols_F = []
    vals_F = []

    for x in range(V):
        for mu in range(4):
            for nu in range(4):
                if mu == nu:
                    continue
                p, sign = plane_idx[(mu, nu)]
                Fc = sign * F_flat[x, p, :]  # F_{μν}^c at site x

                for a in range(3):
                    for b in range(3):
                        # M_{μa, νb} = -2 ε^{abc} F_{μν}^c
                        val = 0.0
                        for c in range(3):
                            val += -2.0 * eps[a, b, c] * Fc[c]
                        if abs(val) > 1e-15:
                            v_row = 12 * x + 3 * mu + a
                            v_col = 12 * x + 3 * nu + b
                            rows_F.append(v_row)
                            cols_F.append(v_col)
                            vals_F.append(val)

    # Combine
    all_rows = rows_D + rows_F
    all_cols = cols_D + cols_F
    all_vals = vals_D + vals_F

    M = sparse.coo_matrix((all_vals, (all_rows, all_cols)), shape=(N, N))
    return M.tocsr()


# ===================================================================
#  FULL BRANCH RESPONSE (ghost + vector, correlated)
# ===================================================================

def compute_branch_response(U_inst: np.ndarray, U_free: np.ndarray,
                              mass_sq: float = 0.01,
                              n_samples: int = 50,
                              cg_tol: float = 1e-10,
                              verbose: bool = True) -> Dict:
    """Compute the full branch response δ_ξ log ζ.

    Uses correlated stochastic subtraction (same noise for inst and free).

    Computes:
        Ghost:  dTr_gh = Tr[(Δ₀_inst)⁻¹ D₀²] - Tr[(Δ₀_free)⁻¹ D₀²]
        Vector: dTr_vec = Tr[(Δ₁_inst)⁻¹ D₀²_vec] - Tr[(Δ₁_free)⁻¹ D₀²_vec]

    where D₀²_vec is the temporal Laplacian acting on the vector space
    (12V-dimensional, block-diagonal in μ).
    """
    dims = U_inst.shape[:4]
    V = int(np.prod(dims))
    N_gh = 3 * V
    N_vec = 12 * V

    if verbose:
        print(f"Branch response: {dims}, m²={mass_sq}")

    t0 = time.time()

    # --- Build ghost operators ---
    if verbose:
        print(f"  Building ghost operators ({N_gh}x{N_gh})...", end="", flush=True)
    Dgh_inst = build_adjoint_laplacian(U_inst, mass_sq=mass_sq)
    Dgh_free = build_adjoint_laplacian(U_free, mass_sq=mass_sq)
    D0gh_inst = build_adjoint_laplacian(U_inst, mass_sq=0.0, temporal_only=True)
    D0gh_free = build_adjoint_laplacian(U_free, mass_sq=0.0, temporal_only=True)
    if verbose:
        print(f" done ({time.time()-t0:.1f}s)")

    # --- Build vector operators ---
    if verbose:
        print(f"  Building vector operators ({N_vec}x{N_vec})...", end="", flush=True)
    t1 = time.time()
    Dvec_inst = build_vector_operator(U_inst, mass_sq=mass_sq)
    Dvec_free = build_vector_operator(U_free, mass_sq=mass_sq)

    # Temporal Laplacian for vector space: block-diagonal in μ
    # Must use the same (x, μ, a) index ordering as the vector operator:
    #   v = 12*x + 3*μ + a
    # Cannot use block_diag (that would give (μ, x, a) ordering)
    D0gh_inst_3 = build_adjoint_laplacian(U_inst, mass_sq=0.0, temporal_only=True)
    D0gh_free_3 = build_adjoint_laplacian(U_free, mass_sq=0.0, temporal_only=True)

    # Remap ghost indices to vector indices for each μ
    def replicate_to_vector(M_3V):
        """Replicate a 3V×3V ghost matrix to 12V×12V vector space."""
        M_coo = M_3V.tocoo()
        rows, cols, vals = [], [], []
        for mu in range(4):
            for k in range(M_coo.nnz):
                g_r = M_coo.row[k]  # 3*x_r + a_r
                g_c = M_coo.col[k]  # 3*x_c + a_c
                x_r, a_r = divmod(g_r, 3)
                x_c, a_c = divmod(g_c, 3)
                rows.append(12*x_r + 3*mu + a_r)
                cols.append(12*x_c + 3*mu + a_c)
                vals.append(M_coo.data[k])
        return sparse.coo_matrix((vals, (rows, cols)),
                                  shape=(N_vec, N_vec)).tocsr()

    D0vec_inst = replicate_to_vector(D0gh_inst_3)
    D0vec_free = replicate_to_vector(D0gh_free_3)

    if verbose:
        print(f" done ({time.time()-t1:.1f}s)")
        print(f"  Vector nnz: inst={Dvec_inst.nnz}, free={Dvec_free.nnz}")

    rng = np.random.default_rng(42)

    def cg_solve(A, rhs):
        try:
            sol, info = cg(A, rhs, atol=cg_tol, maxiter=5000)
        except TypeError:
            sol, info = cg(A, rhs, tol=cg_tol, maxiter=5000)
        return sol

    # --- Correlated ghost traces ---
    if verbose:
        print(f"  Ghost traces ({n_samples} samples)...", flush=True)

    gh_samples = []
    for k in range(n_samples):
        eta = rng.choice([-1.0, 1.0], size=N_gh)
        v_inst = D0gh_inst.dot(eta)
        v_free = D0gh_free.dot(eta)
        z_inst = cg_solve(Dgh_inst, v_inst)
        z_free = cg_solve(Dgh_free, v_free)
        gh_samples.append(np.dot(eta, z_inst) - np.dot(eta, z_free))
        if verbose and (k+1) % 20 == 0:
            m = np.mean(gh_samples)
            e = np.std(gh_samples) / np.sqrt(len(gh_samples))
            print(f"    gh {k+1}/{n_samples}: dTr = {m:.4f} +/- {e:.4f}")

    dTr_gh = np.mean(gh_samples)
    err_gh = np.std(gh_samples) / np.sqrt(n_samples)

    # --- Correlated vector traces ---
    if verbose:
        print(f"  Vector traces ({n_samples} samples)...", flush=True)

    vec_samples = []
    for k in range(n_samples):
        eta = rng.choice([-1.0, 1.0], size=N_vec)
        v_inst = D0vec_inst.dot(eta)
        v_free = D0vec_free.dot(eta)
        z_inst = cg_solve(Dvec_inst, v_inst)
        z_free = cg_solve(Dvec_free, v_free)
        vec_samples.append(np.dot(eta, z_inst) - np.dot(eta, z_free))
        if verbose and (k+1) % 10 == 0:
            m = np.mean(vec_samples)
            e = np.std(vec_samples) / np.sqrt(len(vec_samples))
            print(f"    vec {k+1}/{n_samples}: dTr = {m:.4f} +/- {e:.4f}")

    dTr_vec = np.mean(vec_samples)
    err_vec = np.std(vec_samples) / np.sqrt(n_samples)

    # --- Assemble ---
    # δ log ζ = -(1/2) dTr_vec + dTr_gh  (per unit of δΔ operator)
    # For the temporal response (D₀² insertion):
    #   δ_ξ log ζ = 2ε × [-(1/2) dTr_vec + dTr_gh + zero-mode]
    delta_logzeta_temporal = -0.5 * dTr_vec + dTr_gh
    err_combined = np.sqrt((0.5 * err_vec)**2 + err_gh**2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS (correlated free-subtracted)")
        print(f"{'='*60}")
        print(f"  Ghost:   dTr[Δ₀⁻¹ D₀²]  = {dTr_gh:+.4f} ± {err_gh:.4f}")
        print(f"  Vector:  dTr[Δ₁⁻¹ D₀²]  = {dTr_vec:+.4f} ± {err_vec:.4f}")
        print(f"")
        print(f"  δ log ζ (temporal, no zero-mode) = -(1/2)×vec + ghost")
        print(f"         = {delta_logzeta_temporal:+.4f} ± {err_combined:.4f}")
        print(f"")
        print(f"  For ε = ξ-1 = 0.09 (PFMT working band):")
        print(f"    δ_ξ log ζ ≈ 2ε × ({delta_logzeta_temporal:.4f})")
        print(f"              = {2*0.09*delta_logzeta_temporal:+.4f} ± {2*0.09*err_combined:.4f}")
        print(f"  Total time: {time.time()-t0:.1f}s")

    return {
        'dTr_ghost': dTr_gh,
        'err_ghost': err_gh,
        'dTr_vector': dTr_vec,
        'err_vector': err_vec,
        'delta_logzeta': delta_logzeta_temporal,
        'err_combined': err_combined,
    }


# ===================================================================
#  SELF-TEST
# ===================================================================

def _self_test():
    from src.lattice import init_cold
    from src.caloron import init_caloron

    print("="*72)
    print("FULL BRANCH RESPONSE: Ghost + Vector on caloron 8^3x8")
    print("="*72)

    Ls, Lt, rho = 8, 8, 3.0
    U_free = init_cold((Lt, Ls, Ls, Ls))

    print("\nBuilding caloron...")
    U_inst = init_caloron(Ls, Lt, rho=rho, n_sub=16)

    print()
    result = compute_branch_response(
        U_inst, U_free,
        mass_sq=0.01,
        n_samples=50,
        verbose=True)


if __name__ == "__main__":
    _self_test()
