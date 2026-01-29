"""
Linear system solvers for Newton-Raphson iteration.

Provides interfaces to sparse matrix solvers from scipy.sparse.linalg.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from typing import Tuple, Optional
from enum import Enum


class SolverType(Enum):
    """Available linear solver types."""
    DIRECT = "direct"           # LU decomposition (scipy.sparse.linalg.spsolve)
    GMRES = "gmres"             # Generalized Minimal Residual
    BICGSTAB = "bicgstab"       # Bi-Conjugate Gradient Stabilized
    CG = "cg"                   # Conjugate Gradient (for SPD matrices)


class LinearSolver:
    """
    Linear system solver wrapper.

    Solves Ax = b for sparse matrix A using various methods.

    Attributes:
        solver_type: Type of solver to use
        tolerance: Convergence tolerance for iterative solvers
        max_iterations: Maximum iterations for iterative solvers
        preconditioner: Preconditioner type for iterative solvers
    """

    def __init__(self,
                 solver_type: SolverType = SolverType.DIRECT,
                 tolerance: float = 1e-10,
                 max_iterations: int = 1000,
                 preconditioner: str = 'ilu'):
        """
        Initialize linear solver.

        Args:
            solver_type: Type of solver
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            preconditioner: Preconditioner type ('ilu', 'jacobi', 'none')
        """
        self.solver_type = solver_type
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.preconditioner = preconditioner

        # Cached preconditioner
        self._M = None
        self._last_matrix_id = None

    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Solve linear system Ax = b.

        Args:
            A: Sparse coefficient matrix
            b: Right-hand side vector

        Returns:
            Tuple of (solution vector, success flag)
        """
        if self.solver_type == SolverType.DIRECT:
            return self._solve_direct(A, b)
        elif self.solver_type == SolverType.GMRES:
            return self._solve_gmres(A, b)
        elif self.solver_type == SolverType.BICGSTAB:
            return self._solve_bicgstab(A, b)
        elif self.solver_type == SolverType.CG:
            return self._solve_cg(A, b)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")

    def _solve_direct(self, A: sparse.spmatrix, b: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Solve using direct LU decomposition.

        Args:
            A: Coefficient matrix
            b: Right-hand side

        Returns:
            Tuple of (solution, success)
        """
        try:
            # Convert to CSC format for efficient factorization
            A_csc = sparse.csc_matrix(A)

            # Use SuperLU via spsolve
            x = splinalg.spsolve(A_csc, b)

            # Check for NaN or Inf
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                return np.zeros_like(b), False

            return x, True

        except Exception:
            return np.zeros_like(b), False

    def _solve_gmres(self, A: sparse.spmatrix, b: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Solve using GMRES iterative method.

        Args:
            A: Coefficient matrix
            b: Right-hand side

        Returns:
            Tuple of (solution, success)
        """
        try:
            # Get preconditioner
            M = self._get_preconditioner(A)

            # Initial guess
            x0 = np.zeros_like(b)

            # Solve
            x, info = splinalg.gmres(
                A, b, x0=x0, M=M,
                rtol=self.tolerance,
                maxiter=self.max_iterations
            )

            if info == 0:
                return x, True
            else:
                return x, False

        except Exception:
            return np.zeros_like(b), False

    def _solve_bicgstab(self, A: sparse.spmatrix, b: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Solve using BiCGSTAB iterative method.

        Args:
            A: Coefficient matrix
            b: Right-hand side

        Returns:
            Tuple of (solution, success)
        """
        try:
            M = self._get_preconditioner(A)
            x0 = np.zeros_like(b)

            x, info = splinalg.bicgstab(
                A, b, x0=x0, M=M,
                rtol=self.tolerance,
                maxiter=self.max_iterations
            )

            if info == 0:
                return x, True
            else:
                return x, False

        except Exception:
            return np.zeros_like(b), False

    def _solve_cg(self, A: sparse.spmatrix, b: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Solve using Conjugate Gradient method.

        Note: Only works for symmetric positive definite matrices.

        Args:
            A: Coefficient matrix (must be SPD)
            b: Right-hand side

        Returns:
            Tuple of (solution, success)
        """
        try:
            M = self._get_preconditioner(A)
            x0 = np.zeros_like(b)

            x, info = splinalg.cg(
                A, b, x0=x0, M=M,
                rtol=self.tolerance,
                maxiter=self.max_iterations
            )

            if info == 0:
                return x, True
            else:
                return x, False

        except Exception:
            return np.zeros_like(b), False

    def _get_preconditioner(self, A: sparse.spmatrix) -> Optional[splinalg.LinearOperator]:
        """
        Get preconditioner for iterative solvers.

        Args:
            A: Coefficient matrix

        Returns:
            Preconditioner as LinearOperator, or None
        """
        if self.preconditioner == 'none':
            return None

        # Check if we can reuse cached preconditioner
        matrix_id = id(A)
        if self._M is not None and self._last_matrix_id == matrix_id:
            return self._M

        try:
            if self.preconditioner == 'ilu':
                # Incomplete LU factorization
                A_csc = sparse.csc_matrix(A)
                ilu = splinalg.spilu(A_csc)
                self._M = splinalg.LinearOperator(A.shape, ilu.solve)

            elif self.preconditioner == 'jacobi':
                # Jacobi (diagonal) preconditioner
                diag = A.diagonal()
                diag = np.where(np.abs(diag) > 1e-15, diag, 1.0)
                self._M = splinalg.LinearOperator(
                    A.shape,
                    matvec=lambda x: x / diag
                )
            else:
                self._M = None

            self._last_matrix_id = matrix_id
            return self._M

        except Exception:
            self._M = None
            return None


def solve_linear_system(A: sparse.spmatrix, b: np.ndarray,
                       method: str = 'direct') -> Tuple[np.ndarray, bool]:
    """
    Convenience function to solve linear system.

    Args:
        A: Coefficient matrix
        b: Right-hand side
        method: Solver method ('direct', 'gmres', 'bicgstab', 'cg')

    Returns:
        Tuple of (solution, success)
    """
    solver_map = {
        'direct': SolverType.DIRECT,
        'gmres': SolverType.GMRES,
        'bicgstab': SolverType.BICGSTAB,
        'cg': SolverType.CG,
    }

    solver_type = solver_map.get(method.lower(), SolverType.DIRECT)
    solver = LinearSolver(solver_type=solver_type)

    return solver.solve(A, b)


def compute_condition_number(A: sparse.spmatrix,
                            estimate: bool = True) -> float:
    """
    Compute or estimate the condition number of a matrix.

    Args:
        A: Sparse matrix
        estimate: If True, use estimation (faster for large matrices)

    Returns:
        Condition number (or estimate)
    """
    if estimate:
        # Use norm estimation
        try:
            norm_A = splinalg.norm(A)
            A_csc = sparse.csc_matrix(A)
            lu = splinalg.splu(A_csc)

            # Estimate ||A^{-1}|| by solving A*x = e for random e
            n = A.shape[0]
            e = np.random.randn(n)
            e /= np.linalg.norm(e)
            x = lu.solve(e)
            norm_Ainv = np.linalg.norm(x)

            return norm_A * norm_Ainv

        except Exception:
            return np.inf
    else:
        # Convert to dense and compute exactly (only for small matrices)
        if A.shape[0] > 1000:
            raise ValueError("Matrix too large for exact condition number")

        A_dense = A.toarray()
        return np.linalg.cond(A_dense)
