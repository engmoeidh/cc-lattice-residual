"""
PFMT CC Programme — Configuration (v2)

NORMALISATION CONVENTION (LOCKED — do not change)
==================================================

Wilson action for SU(2) in the fundamental representation:

    S_W(beta) = beta * sum_{x, mu<nu} (1 - (1/2) Re Tr P_{mu,nu}(x))

Generators:  T^a = sigma^a / 2,   Tr(T^a T^b) = (1/2) delta^{ab}

Continuum limit of a unit self-dual SU(2) instanton:

    S_cont = (1/2g^2) int d^4x F^a_{mu,nu} F^a_{mu,nu}
           = 8 pi^2 / g^2
           = 2 pi^2 beta       [since beta = 4/g^2 for SU(2)]

Therefore:
    S_W(beta=1) -> 2 pi^2     for a smooth unit instanton

The LATTICE INSTANTON FACTOR is defined as:

    I_lat = S_W(beta=1) / (2 pi^2)

so that  I_lat -> 1  for a smooth continuum-like unit instanton.

The full instanton action in the CC formula is:

    S_inst = beta * S_W(beta=1) = 2 pi^2 * beta * I_lat

With  beta = 4 a_C  (bare lattice curvature stiffness):

    S_inst = 8 pi^2 * a_C * I_lat

Target:  S_inst ~ 277  =>  a_C * I_lat ~ 3.5

THREE-ACTION SEPARATION
==================================================

S_target :  the physical action whose saddle we seek.
            Eventually: the induced chiral curvature-squared action.
            For now (proxy): Wilson plaquette action.

S_flow   :  the kernel used for gradient flow smoothing.
            Chosen for UV properties, topology preservation.
            Over-improved Symanzik:  c0 = 3, c1 = -1/4.
            NOT the same as S_target in general.

S_diag   :  the action monitored for diagnostics.
            Can be Wilson, Symanzik, or clover-based.
            Used only for human readability, never in the solver.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# Locked normalisation constant
NORM_INSTANTON = 2.0 * np.pi**2   # S_W(beta=1) for unit instanton -> 2 pi^2


@dataclass
class LatticeConfig:
    """Lattice geometry."""
    L: int = 16
    nd: int = 4
    nc: int = 2

    def __post_init__(self):
        assert self.nd == 4 and self.nc == 2


@dataclass
class TargetActionConfig:
    """S_target: the action whose saddle we seek.

    For the proxy SU(2) problem, this is the Wilson action
    at coupling beta_target.  Eventually replaced by the
    physical chiral/coframe action.
    """
    beta: float = 2.5
    # Wilson action: c0=1, c1=0
    c0: float = 1.0
    c1: float = 0.0

    def __post_init__(self):
        assert abs(self.c0 + 8 * self.c1 - 1.0) < 1e-12, \
            f"Symanzik normalisation violated: c0+8c1={self.c0+8*self.c1}"


@dataclass
class FlowActionConfig:
    """S_flow: the kernel for gradient flow.

    Over-improved Symanzik for topology stabilisation:
        c0 = 1 + 8|c1| = 3.0,  c1 = -0.25

    This creates a barrier at small instanton size rho ~ a,
    preventing instantons from shrinking through the lattice.

    Reference: de Forcrand, Garcia Perez, Stamatescu (1997)
    """
    c0: float = 3.0
    c1: float = -0.25

    dt: float = 0.005
    n_steps_max: int = 600
    integrator: str = "rk3"
    measure_interval: int = 10

    def __post_init__(self):
        assert abs(self.c0 + 8 * self.c1 - 1.0) < 1e-12


@dataclass
class DiagnosticConfig:
    """S_diag: what we monitor (never enters solver)."""
    use_clover_action: bool = True
    use_clover_Q: bool = True
    # Planned additions:
    use_geometric_Q: bool = False   # Luscher geometric charge
    use_overlap_index: bool = False  # fermionic index


@dataclass
class SolverConfig:
    """Top-level configuration."""
    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    target: TargetActionConfig = field(default_factory=TargetActionConfig)
    flow: FlowActionConfig = field(default_factory=FlowActionConfig)
    diag: DiagnosticConfig = field(default_factory=DiagnosticConfig)

    output_dir: str = "results"
    log_dir: str = "logs"
    seed: Optional[int] = 42

    @staticmethod
    def I_lat(S_W_beta1: float) -> float:
        """Compute I_lat from Wilson action at beta=1.

        I_lat = S_W(beta=1) / (2 pi^2)

        I_lat -> 1 for a smooth unit instanton.
        """
        return S_W_beta1 / NORM_INSTANTON

    @staticmethod
    def a_C_from_I_lat(I_lat: float, S_target: float = 277.0) -> float:
        """Infer a_C from I_lat and the target exponent.

        S_inst = 8 pi^2 a_C I_lat = S_target
        => a_C = S_target / (8 pi^2 I_lat)
        """
        return S_target / (8.0 * np.pi**2 * I_lat)
