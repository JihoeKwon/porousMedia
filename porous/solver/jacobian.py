"""
Jacobian matrix assembly for Newton-Raphson solver.

Computes the Jacobian matrix using numerical differentiation
of the residual vector with respect to primary variables.
"""

import numpy as np
from scipy import sparse
from typing import List, Optional, TYPE_CHECKING

from ..physics.flow import assemble_residual

if TYPE_CHECKING:
    from ..core.mesh import Mesh
    from ..core.state import State, StateManager
    from ..core.properties import RockProperties


class JacobianAssembler:
    """
    Assembles the Jacobian matrix for the Newton-Raphson solver.

    Uses numerical differentiation to compute partial derivatives
    of the residual with respect to primary variables.

    Attributes:
        mesh: Mesh object
        state_manager: StateManager object
        rock_properties: List of RockProperties
        num_eq: Number of equations per cell
        include_energy: Whether to include energy equation
    """

    def __init__(self,
                 mesh: 'Mesh',
                 state_manager: 'StateManager',
                 rock_properties: List['RockProperties'],
                 include_energy: bool = False):
        """
        Initialize Jacobian assembler.

        Args:
            mesh: Mesh object
            state_manager: StateManager object
            rock_properties: List of RockProperties for each cell
            include_energy: Whether to include energy equation
        """
        self.mesh = mesh
        self.state_manager = state_manager
        self.rock_properties = rock_properties
        self.include_energy = include_energy
        self.num_eq = 3 if include_energy else 2

        # Perturbation sizes for numerical differentiation
        self.dp = 100.0  # Pressure perturbation [Pa]
        self.dT = 0.01   # Temperature perturbation [K]
        self.dS = 1e-4   # Saturation perturbation [-] (was 1e-6, too small)

    def compute_jacobian(self, dt: float,
                        source_terms: Optional[np.ndarray] = None
                        ) -> sparse.csr_matrix:
        """
        Compute the Jacobian matrix.

        Uses numerical differentiation:
        J[i,j] = (R(x + dx_j) - R(x)) / dx_j

        Args:
            dt: Time step [s]
            source_terms: Source terms array

        Returns:
            Jacobian matrix in CSR sparse format
        """
        num_cells = self.mesh.num_cells
        num_pv = 2  # Number of primary variables per cell
        n_total = num_cells * self.num_eq
        m_total = num_cells * num_pv

        # Build sparsity pattern
        # Each cell's equations depend on:
        # - Its own primary variables
        # - Neighbor cells' primary variables (through flux terms)
        rows = []
        cols = []
        data = []

        # Get base residual
        residual_base = assemble_residual(
            self.mesh,
            self.state_manager.states,
            self.state_manager.states_old,
            self.rock_properties,
            dt,
            source_terms,
            self.include_energy
        )

        # Perturb each primary variable and compute derivative
        for cell_idx in range(num_cells):
            state = self.state_manager.states[cell_idx]

            # Get perturbation sizes based on phase state
            from ..core.state import PhaseState
            if state.phase_state == PhaseState.TWO_PHASE:
                perturbations = [self.dp, self.dS]
            else:
                perturbations = [self.dp, self.dT]

            for pv_idx in range(num_pv):
                # Save original primary variables
                p1_orig, p2_orig = state.get_primary_variables()

                # Perturb
                delta = perturbations[pv_idx]
                if pv_idx == 0:
                    state.set_primary_variables(p1_orig + delta, p2_orig)
                else:
                    state.set_primary_variables(p1_orig, p2_orig + delta)

                # Compute perturbed residual
                residual_pert = assemble_residual(
                    self.mesh,
                    self.state_manager.states,
                    self.state_manager.states_old,
                    self.rock_properties,
                    dt,
                    source_terms,
                    self.include_energy
                )

                # Restore original state
                state.set_primary_variables(p1_orig, p2_orig)

                # Compute derivatives
                # This cell's equations
                for eq_idx in range(self.num_eq):
                    row = cell_idx * self.num_eq + eq_idx
                    col = cell_idx * num_pv + pv_idx
                    deriv = (residual_pert[row] - residual_base[row]) / delta
                    if abs(deriv) > 1e-30:
                        rows.append(row)
                        cols.append(col)
                        data.append(deriv)

                # Neighbor cells' equations (due to flux coupling)
                for neighbor_idx in self.mesh.get_neighbor_cells(cell_idx):
                    for eq_idx in range(self.num_eq):
                        row = neighbor_idx * self.num_eq + eq_idx
                        col = cell_idx * num_pv + pv_idx
                        deriv = (residual_pert[row] - residual_base[row]) / delta
                        if abs(deriv) > 1e-30:
                            rows.append(row)
                            cols.append(col)
                            data.append(deriv)

        # Create sparse matrix
        jacobian = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_total, m_total)
        )

        return jacobian

    def compute_jacobian_dense(self, dt: float,
                               source_terms: Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """
        Compute dense Jacobian matrix (for small problems).

        Args:
            dt: Time step [s]
            source_terms: Source terms array

        Returns:
            Dense Jacobian matrix
        """
        num_cells = self.mesh.num_cells
        num_pv = 2
        n_total = num_cells * self.num_eq
        m_total = num_cells * num_pv

        jacobian = np.zeros((n_total, m_total))

        # Get base residual
        residual_base = assemble_residual(
            self.mesh,
            self.state_manager.states,
            self.state_manager.states_old,
            self.rock_properties,
            dt,
            source_terms,
            self.include_energy
        )

        # Perturb each primary variable
        for cell_idx in range(num_cells):
            state = self.state_manager.states[cell_idx]

            from ..core.state import PhaseState
            if state.phase_state == PhaseState.TWO_PHASE:
                perturbations = [self.dp, self.dS]
            else:
                perturbations = [self.dp, self.dT]

            for pv_idx in range(num_pv):
                p1_orig, p2_orig = state.get_primary_variables()

                delta = perturbations[pv_idx]
                if pv_idx == 0:
                    state.set_primary_variables(p1_orig + delta, p2_orig)
                else:
                    state.set_primary_variables(p1_orig, p2_orig + delta)

                residual_pert = assemble_residual(
                    self.mesh,
                    self.state_manager.states,
                    self.state_manager.states_old,
                    self.rock_properties,
                    dt,
                    source_terms,
                    self.include_energy
                )

                state.set_primary_variables(p1_orig, p2_orig)

                col = cell_idx * num_pv + pv_idx
                jacobian[:, col] = (residual_pert - residual_base) / delta

        return jacobian


def compute_sparsity_pattern(mesh: 'Mesh',
                             num_eq: int = 2,
                             num_pv: int = 2
                             ) -> sparse.csr_matrix:
    """
    Compute the sparsity pattern of the Jacobian matrix.

    This can be used to preallocate the Jacobian structure
    for more efficient assembly.

    Args:
        mesh: Mesh object
        num_eq: Number of equations per cell
        num_pv: Number of primary variables per cell

    Returns:
        Sparse matrix with 1s at nonzero positions
    """
    num_cells = mesh.num_cells
    n_total = num_cells * num_eq
    m_total = num_cells * num_pv

    rows = []
    cols = []

    for cell_idx in range(num_cells):
        # Diagonal block (cell's own variables)
        for eq_idx in range(num_eq):
            for pv_idx in range(num_pv):
                rows.append(cell_idx * num_eq + eq_idx)
                cols.append(cell_idx * num_pv + pv_idx)

        # Off-diagonal blocks (neighbor cells)
        for neighbor_idx in mesh.get_neighbor_cells(cell_idx):
            for eq_idx in range(num_eq):
                for pv_idx in range(num_pv):
                    rows.append(cell_idx * num_eq + eq_idx)
                    cols.append(neighbor_idx * num_pv + pv_idx)

    data = np.ones(len(rows))
    pattern = sparse.csr_matrix((data, (rows, cols)), shape=(n_total, m_total))

    return pattern
