"""
State variable management for porous media simulation.

Manages primary and secondary variables, phase state tracking,
and primary variable switching for phase transitions.
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING
from copy import deepcopy

from ..utils.constants import (
    PHASE_LIQUID, PHASE_GAS, SATURATION_EPSILON, EPSILON,
    CELSIUS_OFFSET, STANDARD_PRESSURE
)
from ..physics.eos import (
    WaterEOS, SteamEOS, AirEOS,
    saturation_pressure, saturation_temperature
)
from ..physics.relative_perm import RelativePermeability, CoreyRelPerm
from ..physics.capillary import CapillaryPressure, VanGenuchtenCapillary

if TYPE_CHECKING:
    from .properties import RockProperties


class PhaseState(IntEnum):
    """
    Enumeration of possible phase states.

    SINGLE_LIQUID: Only liquid phase present
    SINGLE_GAS: Only gas phase present
    TWO_PHASE: Both liquid and gas phases present
    """
    SINGLE_LIQUID = 0
    SINGLE_GAS = 1
    TWO_PHASE = 2


class FluidSystem(IntEnum):
    """
    Enumeration of fluid system types.

    WATER_STEAM: Water with phase change (thermal)
    WATER_AIR: Immiscible water-air (isothermal)
    """
    WATER_STEAM = 0
    WATER_AIR = 1


@dataclass
class State:
    """
    Complete thermodynamic state for a cell.

    Manages primary variables, secondary (derived) variables,
    and handles phase state transitions.

    Primary Variables (depending on phase state and fluid system):
    - WATER_STEAM system:
      - SINGLE_LIQUID: (pressure, temperature)
      - TWO_PHASE: (pressure, saturation) - temperature from saturation
    - WATER_AIR system (isothermal):
      - Always: (pressure, saturation) - temperature is fixed

    Attributes:
        pressure: Reference pressure (gas phase pressure) [Pa]
        temperature: Temperature [K]
        liquid_saturation: Liquid phase saturation [-]
        phase_state: Current phase state
        fluid_system: Type of fluid system

        # Derived properties (computed from EOS)
        liquid_density: Liquid phase density [kg/m³]
        gas_density: Gas phase density [kg/m³]
        liquid_viscosity: Liquid dynamic viscosity [Pa·s]
        gas_viscosity: Gas dynamic viscosity [Pa·s]
        liquid_enthalpy: Liquid specific enthalpy [J/kg]
        gas_enthalpy: Gas specific enthalpy [J/kg]
        liquid_internal_energy: Liquid specific internal energy [J/kg]
        gas_internal_energy: Gas specific internal energy [J/kg]
        liquid_rel_perm: Liquid relative permeability [-]
        gas_rel_perm: Gas relative permeability [-]
        capillary_pressure: Capillary pressure Pg - Pl [Pa]
    """
    # Primary variables
    pressure: float = STANDARD_PRESSURE
    temperature: float = 293.15  # 20°C
    liquid_saturation: float = 1.0
    phase_state: PhaseState = PhaseState.SINGLE_LIQUID
    fluid_system: FluidSystem = field(default=FluidSystem.WATER_AIR)

    # Secondary variables (derived)
    liquid_density: float = field(default=1000.0, repr=False)
    gas_density: float = field(default=1.0, repr=False)
    liquid_viscosity: float = field(default=1e-3, repr=False)
    gas_viscosity: float = field(default=1.8e-5, repr=False)
    liquid_enthalpy: float = field(default=0.0, repr=False)
    gas_enthalpy: float = field(default=0.0, repr=False)
    liquid_internal_energy: float = field(default=0.0, repr=False)
    gas_internal_energy: float = field(default=0.0, repr=False)
    liquid_rel_perm: float = field(default=1.0, repr=False)
    gas_rel_perm: float = field(default=0.0, repr=False)
    capillary_pressure: float = field(default=0.0, repr=False)

    # Model references (set during initialization)
    _rel_perm_model: Optional[RelativePermeability] = field(default=None, repr=False)
    _cap_pressure_model: Optional[CapillaryPressure] = field(default=None, repr=False)

    @property
    def gas_saturation(self) -> float:
        """Gas phase saturation."""
        return 1.0 - self.liquid_saturation

    @property
    def gas_pressure(self) -> float:
        """Gas phase pressure."""
        return self.pressure

    @property
    def liquid_pressure(self) -> float:
        """Liquid phase pressure (accounting for capillary pressure)."""
        return self.pressure - self.capillary_pressure

    def set_models(self, rel_perm: RelativePermeability,
                   cap_pressure: CapillaryPressure):
        """
        Set the constitutive models for this state.

        Args:
            rel_perm: Relative permeability model
            cap_pressure: Capillary pressure model
        """
        self._rel_perm_model = rel_perm
        self._cap_pressure_model = cap_pressure

    def update_secondary_variables(self):
        """
        Update all secondary (derived) variables from primary variables.

        This should be called after any change to primary variables.
        """
        # Compute relative permeabilities
        if self._rel_perm_model is not None:
            self.liquid_rel_perm = self._rel_perm_model.liquid(self.liquid_saturation)
            self.gas_rel_perm = self._rel_perm_model.gas(self.liquid_saturation)
        else:
            # Default: linear rel perm
            self.liquid_rel_perm = self.liquid_saturation
            self.gas_rel_perm = self.gas_saturation

        # Compute capillary pressure
        if self._cap_pressure_model is not None:
            self.capillary_pressure = self._cap_pressure_model.pressure(self.liquid_saturation)
        else:
            self.capillary_pressure = 0.0

        # Liquid properties from EOS
        # Always compute liquid properties for proper Jacobian derivatives
        P_l = self.liquid_pressure
        water_state = WaterEOS.get_state(P_l, self.temperature)
        self.liquid_density = water_state.density
        self.liquid_viscosity = water_state.viscosity
        self.liquid_enthalpy = water_state.enthalpy
        self.liquid_internal_energy = water_state.internal_energy

        # Gas properties from EOS
        # Always compute gas properties for proper Jacobian derivatives
        # even when gas saturation is zero
        if self.fluid_system == FluidSystem.WATER_AIR:
            # Use air EOS for water-air system
            air_state = AirEOS.get_state(self.pressure, self.temperature)
            self.gas_density = air_state.density
            self.gas_viscosity = air_state.viscosity
            self.gas_enthalpy = air_state.enthalpy
            self.gas_internal_energy = air_state.internal_energy
        else:
            # Use steam EOS for water-steam system
            steam_state = SteamEOS.get_state(self.pressure, self.temperature)
            self.gas_density = steam_state.density
            self.gas_viscosity = steam_state.viscosity
            self.gas_enthalpy = steam_state.enthalpy
            self.gas_internal_energy = steam_state.internal_energy

    def set_primary_variables(self, pressure: float, second_var: float,
                             phase_state: Optional[PhaseState] = None):
        """
        Set primary variables and update secondary variables.

        Args:
            pressure: Reference pressure [Pa]
            second_var: Second primary variable
                - WATER_AIR system: always liquid saturation [-]
                - WATER_STEAM system:
                  - SINGLE_LIQUID: temperature [K]
                  - SINGLE_GAS: temperature [K]
                  - TWO_PHASE: liquid saturation [-]
            phase_state: Phase state (if None, current state is kept)
        """
        self.pressure = pressure

        if phase_state is not None:
            self.phase_state = phase_state

        if self.fluid_system == FluidSystem.WATER_AIR:
            # Isothermal water-air: second variable is always saturation
            self.liquid_saturation = max(SATURATION_EPSILON,
                                        min(1.0 - SATURATION_EPSILON, second_var))
            # Temperature stays fixed (isothermal)
            # Determine phase state from saturation
            if self.liquid_saturation >= 1.0 - SATURATION_EPSILON:
                self.phase_state = PhaseState.SINGLE_LIQUID
            elif self.liquid_saturation <= SATURATION_EPSILON:
                self.phase_state = PhaseState.SINGLE_GAS
            else:
                self.phase_state = PhaseState.TWO_PHASE
        else:
            # Water-steam system
            if self.phase_state == PhaseState.TWO_PHASE:
                self.liquid_saturation = second_var
                # Temperature from saturation
                self.temperature = saturation_temperature(pressure)
            else:
                self.temperature = second_var
                if self.phase_state == PhaseState.SINGLE_LIQUID:
                    self.liquid_saturation = 1.0
                else:  # SINGLE_GAS
                    self.liquid_saturation = 0.0

        self.update_secondary_variables()

    def get_primary_variables(self) -> Tuple[float, float]:
        """
        Get current primary variables.

        Returns:
            Tuple of (pressure, second_variable)
        """
        if self.fluid_system == FluidSystem.WATER_AIR:
            # Water-air: always (P, Sl)
            return self.pressure, self.liquid_saturation
        else:
            # Water-steam: depends on phase state
            if self.phase_state == PhaseState.TWO_PHASE:
                return self.pressure, self.liquid_saturation
            else:
                return self.pressure, self.temperature

    def check_phase_transition(self) -> bool:
        """
        Check if phase state should change and perform transition if needed.

        Returns:
            True if phase state changed
        """
        if self.fluid_system == FluidSystem.WATER_AIR:
            # For water-air, phase state determined by saturation bounds
            old_state = self.phase_state
            if self.liquid_saturation >= 1.0 - SATURATION_EPSILON:
                self.phase_state = PhaseState.SINGLE_LIQUID
            elif self.liquid_saturation <= SATURATION_EPSILON:
                self.phase_state = PhaseState.SINGLE_GAS
            else:
                self.phase_state = PhaseState.TWO_PHASE
            return self.phase_state != old_state

        # Water-steam system: check saturation pressure
        old_state = self.phase_state
        P_sat = saturation_pressure(self.temperature)

        if self.phase_state == PhaseState.SINGLE_LIQUID:
            # Check if pressure dropped below saturation
            if self.pressure < P_sat:
                # Transition to two-phase
                self.phase_state = PhaseState.TWO_PHASE
                self.liquid_saturation = 1.0 - SATURATION_EPSILON

        elif self.phase_state == PhaseState.SINGLE_GAS:
            # Check if pressure exceeds saturation
            if self.pressure > P_sat:
                # Transition to two-phase
                self.phase_state = PhaseState.TWO_PHASE
                self.liquid_saturation = SATURATION_EPSILON

        elif self.phase_state == PhaseState.TWO_PHASE:
            # Check saturation bounds
            if self.liquid_saturation >= 1.0 - SATURATION_EPSILON:
                # Transition to single liquid
                self.phase_state = PhaseState.SINGLE_LIQUID
                self.liquid_saturation = 1.0
            elif self.liquid_saturation <= SATURATION_EPSILON:
                # Transition to single gas
                self.phase_state = PhaseState.SINGLE_GAS
                self.liquid_saturation = 0.0

        if self.phase_state != old_state:
            self.update_secondary_variables()
            return True

        return False

    def copy(self) -> 'State':
        """Create a deep copy of this state."""
        new_state = deepcopy(self)
        return new_state


class StateManager:
    """
    Manages states for all cells in the simulation.

    Handles initialization, updates, and primary variable
    management for the entire mesh.
    """

    def __init__(self, num_cells: int,
                 default_rel_perm: Optional[RelativePermeability] = None,
                 default_cap_pressure: Optional[CapillaryPressure] = None,
                 fluid_system: FluidSystem = FluidSystem.WATER_AIR):
        """
        Initialize state manager.

        Args:
            num_cells: Number of cells in mesh
            default_rel_perm: Default relative permeability model
            default_cap_pressure: Default capillary pressure model
            fluid_system: Type of fluid system (WATER_AIR or WATER_STEAM)
        """
        self.num_cells = num_cells
        self.fluid_system = fluid_system

        # Default models
        if default_rel_perm is None:
            default_rel_perm = CoreyRelPerm()
        if default_cap_pressure is None:
            default_cap_pressure = VanGenuchtenCapillary()

        self.default_rel_perm = default_rel_perm
        self.default_cap_pressure = default_cap_pressure

        # Initialize states
        self.states = [State(fluid_system=fluid_system) for _ in range(num_cells)]
        for state in self.states:
            state.set_models(default_rel_perm, default_cap_pressure)

        # Old states (for time stepping)
        self.states_old = [state.copy() for state in self.states]

    def initialize_uniform(self, pressure: float, temperature: float,
                          liquid_saturation: float = 1.0):
        """
        Initialize all cells with uniform conditions.

        Args:
            pressure: Initial pressure [Pa]
            temperature: Initial temperature [K]
            liquid_saturation: Initial liquid saturation [-]
        """
        for state in self.states:
            if liquid_saturation >= 1.0 - SATURATION_EPSILON:
                phase_state = PhaseState.SINGLE_LIQUID
            elif liquid_saturation <= SATURATION_EPSILON:
                phase_state = PhaseState.SINGLE_GAS
            else:
                phase_state = PhaseState.TWO_PHASE

            state.pressure = pressure
            state.temperature = temperature
            state.liquid_saturation = liquid_saturation
            state.phase_state = phase_state
            state.update_secondary_variables()

        # Copy to old states
        self.states_old = [state.copy() for state in self.states]

    def initialize_from_arrays(self, pressure: np.ndarray,
                               temperature: np.ndarray,
                               liquid_saturation: np.ndarray):
        """
        Initialize cells from arrays.

        Args:
            pressure: Array of pressures [Pa]
            temperature: Array of temperatures [K]
            liquid_saturation: Array of liquid saturations [-]
        """
        for i in range(self.num_cells):
            sl = liquid_saturation[i]
            if sl >= 1.0 - SATURATION_EPSILON:
                phase_state = PhaseState.SINGLE_LIQUID
            elif sl <= SATURATION_EPSILON:
                phase_state = PhaseState.SINGLE_GAS
            else:
                phase_state = PhaseState.TWO_PHASE

            self.states[i].pressure = pressure[i]
            self.states[i].temperature = temperature[i]
            self.states[i].liquid_saturation = sl
            self.states[i].phase_state = phase_state
            self.states[i].update_secondary_variables()

        self.states_old = [state.copy() for state in self.states]

    def save_old_states(self):
        """Save current states as old states (for next time step)."""
        self.states_old = [state.copy() for state in self.states]

    def restore_from_old(self):
        """Restore states from old states (for time step retry)."""
        self.states = [state.copy() for state in self.states_old]

    def get_primary_variable_array(self) -> np.ndarray:
        """
        Get array of all primary variables.

        Returns:
            Array of shape (num_cells * 2,) containing primary variables
        """
        pv = np.zeros(self.num_cells * 2)
        for i, state in enumerate(self.states):
            p1, p2 = state.get_primary_variables()
            pv[2*i] = p1
            pv[2*i + 1] = p2
        return pv

    def set_primary_variable_array(self, pv: np.ndarray):
        """
        Set primary variables from array.

        Args:
            pv: Array of primary variables
        """
        for i, state in enumerate(self.states):
            pressure = pv[2*i]
            second_var = pv[2*i + 1]
            state.set_primary_variables(pressure, second_var)

    def update_primary_variables(self, delta: np.ndarray):
        """
        Update primary variables by adding delta.

        Args:
            delta: Array of primary variable changes
        """
        for i, state in enumerate(self.states):
            p1, p2 = state.get_primary_variables()
            p1_new = p1 + delta[2*i]
            p2_new = p2 + delta[2*i + 1]
            state.set_primary_variables(p1_new, p2_new)

    def check_all_phase_transitions(self) -> int:
        """
        Check and perform phase transitions for all cells.

        Returns:
            Number of cells that changed phase state
        """
        count = 0
        for state in self.states:
            if state.check_phase_transition():
                count += 1
        return count

    def get_pressures(self) -> np.ndarray:
        """Get array of all cell pressures."""
        return np.array([state.pressure for state in self.states])

    def get_temperatures(self) -> np.ndarray:
        """Get array of all cell temperatures."""
        return np.array([state.temperature for state in self.states])

    def get_liquid_saturations(self) -> np.ndarray:
        """Get array of all cell liquid saturations."""
        return np.array([state.liquid_saturation for state in self.states])

    def get_gas_saturations(self) -> np.ndarray:
        """Get array of all cell gas saturations."""
        return np.array([state.gas_saturation for state in self.states])

    def set_cell_models(self, cell_index: int,
                       rel_perm: RelativePermeability,
                       cap_pressure: CapillaryPressure):
        """
        Set constitutive models for a specific cell.

        Args:
            cell_index: Cell index
            rel_perm: Relative permeability model
            cap_pressure: Capillary pressure model
        """
        self.states[cell_index].set_models(rel_perm, cap_pressure)
        self.states[cell_index].update_secondary_variables()
