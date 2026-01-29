"""
Newton-Raphson solver for nonlinear system of equations.

Implements the fully implicit time integration with
Newton-Raphson iteration for the coupled mass and energy
conservation equations.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
from enum import IntEnum

from ..utils.constants import (
    NEWTON_TOLERANCE, MAX_NEWTON_ITERATIONS,
    MAX_TIME_STEP_INCREASE, TIME_STEP_SAFETY_FACTOR
)
from ..physics.flow import assemble_residual
from .jacobian import JacobianAssembler
from .linear import LinearSolver

if TYPE_CHECKING:
    from ..core.mesh import Mesh
    from ..core.state import StateManager
    from ..core.properties import RockProperties


class ConvergenceStatus(IntEnum):
    """Status of Newton iteration."""
    CONVERGED = 0
    ITERATING = 1
    DIVERGED = 2
    MAX_ITERATIONS = 3


@dataclass
class NewtonResult:
    """
    Result of Newton-Raphson iteration.

    Attributes:
        converged: Whether solution converged
        iterations: Number of iterations taken
        residual_norm: Final residual norm
        max_residual: Maximum residual component
        status: Convergence status
    """
    converged: bool
    iterations: int
    residual_norm: float
    max_residual: float
    status: ConvergenceStatus


@dataclass
class TimeStepResult:
    """
    Result of a time step.

    Attributes:
        success: Whether time step succeeded
        dt_used: Time step actually used [s]
        dt_next: Suggested next time step [s]
        newton_result: Result of Newton iteration
        time: Current simulation time [s]
    """
    success: bool
    dt_used: float
    dt_next: float
    newton_result: NewtonResult
    time: float


class NewtonSolver:
    """
    Newton-Raphson solver for fully implicit time stepping.

    Solves the nonlinear system:
        R(x^{n+1}) = 0

    where R is the residual vector and x is the vector of
    primary variables at the new time level.

    Attributes:
        mesh: Mesh object
        state_manager: StateManager object
        rock_properties: List of RockProperties
        include_energy: Whether to include energy equation
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
    """

    def __init__(self,
                 mesh: 'Mesh',
                 state_manager: 'StateManager',
                 rock_properties: List['RockProperties'],
                 include_energy: bool = False,
                 tolerance: float = NEWTON_TOLERANCE,
                 max_iterations: int = MAX_NEWTON_ITERATIONS):
        """
        Initialize Newton solver.

        Args:
            mesh: Mesh object
            state_manager: StateManager object
            rock_properties: List of RockProperties
            include_energy: Whether to include energy equation
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
        """
        self.mesh = mesh
        self.state_manager = state_manager
        self.rock_properties = rock_properties
        self.include_energy = include_energy
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Number of equations per cell
        self.num_eq = 3 if include_energy else 2

        # Jacobian assembler
        self.jacobian_assembler = JacobianAssembler(
            mesh, state_manager, rock_properties, include_energy
        )

        # Linear solver
        self.linear_solver = LinearSolver()

        # Damping parameters
        self.max_dp = 1e6  # Maximum pressure change per iteration [Pa]
        self.max_dT = 10.0  # Maximum temperature change per iteration [K]
        self.max_dS = 0.2   # Maximum saturation change per iteration [-]

    def solve(self, dt: float,
             source_terms: Optional[np.ndarray] = None) -> NewtonResult:
        """
        Perform Newton-Raphson iteration for a time step.

        Args:
            dt: Time step [s]
            source_terms: Source terms array

        Returns:
            NewtonResult object
        """
        # Save old states for potential rollback
        self.state_manager.save_old_states()

        for iteration in range(self.max_iterations):
            # Assemble residual
            residual = assemble_residual(
                self.mesh,
                self.state_manager.states,
                self.state_manager.states_old,
                self.rock_properties,
                dt,
                source_terms,
                self.include_energy
            )

            # Check convergence
            residual_norm = np.linalg.norm(residual)
            max_residual = np.max(np.abs(residual))

            if residual_norm < self.tolerance:
                return NewtonResult(
                    converged=True,
                    iterations=iteration + 1,
                    residual_norm=residual_norm,
                    max_residual=max_residual,
                    status=ConvergenceStatus.CONVERGED
                )

            # Assemble Jacobian
            jacobian = self.jacobian_assembler.compute_jacobian(dt, source_terms)

            # Solve linear system: J * dx = -R
            dx, success = self.linear_solver.solve(jacobian, -residual)

            if not success:
                return NewtonResult(
                    converged=False,
                    iterations=iteration + 1,
                    residual_norm=residual_norm,
                    max_residual=max_residual,
                    status=ConvergenceStatus.DIVERGED
                )

            # Apply damping/limiting to update
            dx = self._limit_update(dx)

            # Update primary variables
            self._apply_update(dx)

            # Check for phase transitions
            self.state_manager.check_all_phase_transitions()

        # Max iterations reached
        residual = assemble_residual(
            self.mesh,
            self.state_manager.states,
            self.state_manager.states_old,
            self.rock_properties,
            dt,
            source_terms,
            self.include_energy
        )

        return NewtonResult(
            converged=False,
            iterations=self.max_iterations,
            residual_norm=np.linalg.norm(residual),
            max_residual=np.max(np.abs(residual)),
            status=ConvergenceStatus.MAX_ITERATIONS
        )

    def _limit_update(self, dx: np.ndarray) -> np.ndarray:
        """
        Apply damping/limiting to the Newton update.

        Args:
            dx: Raw update vector

        Returns:
            Limited update vector
        """
        from ..core.state import PhaseState

        num_cells = self.mesh.num_cells
        num_pv = 2

        for i in range(num_cells):
            state = self.state_manager.states[i]

            # Pressure limiting
            dp = dx[i * num_pv]
            if abs(dp) > self.max_dp:
                dp = np.sign(dp) * self.max_dp
                dx[i * num_pv] = dp

            # Second variable limiting
            d2 = dx[i * num_pv + 1]
            if state.phase_state == PhaseState.TWO_PHASE:
                # Saturation limiting
                if abs(d2) > self.max_dS:
                    d2 = np.sign(d2) * self.max_dS
            else:
                # Temperature limiting
                if abs(d2) > self.max_dT:
                    d2 = np.sign(d2) * self.max_dT
            dx[i * num_pv + 1] = d2

        return dx

    def _apply_update(self, dx: np.ndarray):
        """
        Apply the Newton update to primary variables.

        Args:
            dx: Update vector
        """
        from ..core.state import PhaseState
        from ..utils.constants import SATURATION_EPSILON

        num_cells = self.mesh.num_cells
        num_pv = 2

        for i in range(num_cells):
            state = self.state_manager.states[i]
            p1, p2 = state.get_primary_variables()

            p1_new = p1 + dx[i * num_pv]
            p2_new = p2 + dx[i * num_pv + 1]

            # Enforce bounds
            p1_new = max(p1_new, 1e3)  # Minimum pressure

            if state.phase_state == PhaseState.TWO_PHASE:
                # Bound saturation
                p2_new = max(SATURATION_EPSILON, min(1.0 - SATURATION_EPSILON, p2_new))
            else:
                # Bound temperature
                p2_new = max(273.15, min(647.0, p2_new))  # 0°C to near critical

            state.set_primary_variables(p1_new, p2_new)


class TimeStepController:
    """
    Adaptive time step controller for transient simulations.

    Adjusts time step based on Newton iteration performance.

    Attributes:
        dt_initial: Initial time step [s]
        dt_min: Minimum time step [s]
        dt_max: Maximum time step [s]
        max_cuts: Maximum number of time step cuts before failure
    """

    def __init__(self,
                 dt_initial: float = 1.0,
                 dt_min: float = 1e-6,
                 dt_max: float = 1e8,
                 max_cuts: int = 10):
        """
        Initialize time step controller.

        Args:
            dt_initial: Initial time step [s]
            dt_min: Minimum time step [s]
            dt_max: Maximum time step [s]
            max_cuts: Maximum time step cuts
        """
        self.dt = dt_initial
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.max_cuts = max_cuts
        self.current_time = 0.0

    def step(self, solver: NewtonSolver,
            source_terms: Optional[np.ndarray] = None) -> TimeStepResult:
        """
        Perform a single time step with adaptive control.

        Args:
            solver: NewtonSolver object
            source_terms: Source terms array

        Returns:
            TimeStepResult object
        """
        dt = self.dt
        num_cuts = 0

        while num_cuts <= self.max_cuts:
            # Attempt Newton solve
            result = solver.solve(dt, source_terms)

            if result.converged:
                # Success - update time and suggest next dt
                self.current_time += dt
                dt_next = self._compute_next_dt(dt, result.iterations)
                self.dt = dt_next

                return TimeStepResult(
                    success=True,
                    dt_used=dt,
                    dt_next=dt_next,
                    newton_result=result,
                    time=self.current_time
                )
            else:
                # Failed - cut time step and retry
                solver.state_manager.restore_from_old()
                dt *= 0.5
                num_cuts += 1

                if dt < self.dt_min:
                    break

        # Failed after max cuts
        return TimeStepResult(
            success=False,
            dt_used=0.0,
            dt_next=self.dt_min,
            newton_result=result,
            time=self.current_time
        )

    def _compute_next_dt(self, dt: float, iterations: int) -> float:
        """
        Compute next time step based on iteration count.

        Args:
            dt: Current time step [s]
            iterations: Number of iterations taken

        Returns:
            Suggested next time step [s]
        """
        if iterations <= 3:
            # Very fast convergence - increase dt
            factor = MAX_TIME_STEP_INCREASE
        elif iterations <= 5:
            # Good convergence - modest increase
            factor = 1.5
        elif iterations <= 8:
            # Adequate convergence - keep same
            factor = 1.0
        else:
            # Slow convergence - decrease
            factor = 0.8

        dt_next = dt * factor * TIME_STEP_SAFETY_FACTOR
        dt_next = max(self.dt_min, min(self.dt_max, dt_next))

        return dt_next

    def set_time(self, time: float):
        """Set current simulation time."""
        self.current_time = time

    def set_dt(self, dt: float):
        """Set current time step."""
        self.dt = max(self.dt_min, min(self.dt_max, dt))


class Simulator:
    """
    Main simulation driver.

    Integrates the Newton solver with time stepping
    and output control.
    """

    def __init__(self,
                 mesh: 'Mesh',
                 state_manager: 'StateManager',
                 rock_properties: List['RockProperties'],
                 include_energy: bool = False,
                 log_file: Optional[str] = None):
        """
        Initialize simulator.

        Args:
            mesh: Mesh object
            state_manager: StateManager object
            rock_properties: List of RockProperties
            include_energy: Whether to include energy equation
            log_file: Path to log file (optional)
        """
        self.mesh = mesh
        self.state_manager = state_manager
        self.rock_properties = rock_properties
        self.include_energy = include_energy
        self.log_file = log_file

        # Create solver
        self.newton_solver = NewtonSolver(
            mesh, state_manager, rock_properties, include_energy
        )

        # Create time step controller
        self.time_controller = TimeStepController()

        # Simulation parameters
        self.end_time = 0.0
        self.source_terms = None

        # Output storage
        self.times = []
        self.pressures = []
        self.saturations = []
        self.temperatures = []

        # Log storage
        self.log_data = []

    def set_simulation_time(self, end_time: float, dt_initial: float = 1.0):
        """
        Set simulation end time and initial time step.

        Args:
            end_time: End time [s]
            dt_initial: Initial time step [s]
        """
        self.end_time = end_time
        self.time_controller.set_dt(dt_initial)

    def set_source_terms(self, source_terms: np.ndarray):
        """
        Set source/sink terms.

        Args:
            source_terms: Array of shape (num_cells, num_eq)
        """
        self.source_terms = source_terms

    def run(self, verbose: bool = True) -> bool:
        """
        Run the simulation.

        Args:
            verbose: Whether to print progress

        Returns:
            True if simulation completed successfully
        """
        # Store initial state
        self._store_state()
        self.log_data = []

        # Open log file if specified
        log_fh = None
        if self.log_file:
            log_fh = open(self.log_file, 'w')
            header = f"{'Step':>6},{'Time_s':>14},{'Time_days':>12},{'dt_s':>14},{'Newton':>7},{'Res_norm':>14},{'Sw_min':>10},{'Sw_max':>10},{'P_min_bar':>12},{'P_max_bar':>12}\n"
            log_fh.write(header)

        step_count = 0
        while self.time_controller.current_time < self.end_time:
            # Adjust last time step to hit end time exactly
            remaining = self.end_time - self.time_controller.current_time
            if self.time_controller.dt > remaining:
                self.time_controller.set_dt(remaining)

            # Take time step
            result = self.time_controller.step(
                self.newton_solver,
                self.source_terms
            )

            if not result.success:
                if verbose:
                    print(f"Time step failed at t = {self.time_controller.current_time:.3e}")
                if log_fh:
                    log_fh.close()
                return False

            step_count += 1

            # Store state
            self._store_state()

            # Get current state info
            sats = self.state_manager.get_liquid_saturations()
            pres = self.state_manager.get_pressures()

            # Log entry
            log_entry = {
                'step': step_count,
                'time': result.time,
                'time_days': result.time / 86400.0,
                'dt': result.dt_used,
                'newton_iters': result.newton_result.iterations,
                'residual_norm': result.newton_result.residual_norm,
                'sw_min': sats.min(),
                'sw_max': sats.max(),
                'p_min': pres.min() / 1e5,
                'p_max': pres.max() / 1e5
            }
            self.log_data.append(log_entry)

            # Write to log file
            if log_fh:
                log_fh.write(f"{step_count:6d},{result.time:14.4f},{result.time/86400:12.6f},{result.dt_used:14.4f},{result.newton_result.iterations:7d},{result.newton_result.residual_norm:14.6e},{sats.min():10.6f},{sats.max():10.6f},{pres.min()/1e5:12.4f},{pres.max()/1e5:12.4f}\n")
                log_fh.flush()

            if verbose and step_count % 5 == 0:
                print(f"Step {step_count}: t = {result.time:.3e}, "
                      f"dt = {result.dt_used:.3e}, "
                      f"Newton iters = {result.newton_result.iterations}")

        if log_fh:
            log_fh.close()

        if verbose:
            print(f"Simulation completed: {step_count} time steps")
            if self.log_file:
                print(f"Log saved to: {self.log_file}")

        return True

    def _store_state(self):
        """Store current state for output."""
        self.times.append(self.time_controller.current_time)
        self.pressures.append(self.state_manager.get_pressures().copy())
        self.saturations.append(self.state_manager.get_liquid_saturations().copy())
        self.temperatures.append(self.state_manager.get_temperatures().copy())

    def get_results(self) -> dict:
        """
        Get simulation results.

        Returns:
            Dictionary with times, pressures, saturations, temperatures
        """
        return {
            'times': np.array(self.times),
            'pressures': np.array(self.pressures),
            'saturations': np.array(self.saturations),
            'temperatures': np.array(self.temperatures)
        }
