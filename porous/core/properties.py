"""
Rock and fluid property classes for porous media simulation.

Manages material properties including porosity, permeability,
thermal properties, and constitutive model parameters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from ..utils.constants import (
    DEFAULT_POROSITY, DEFAULT_PERMEABILITY,
    DEFAULT_ROCK_DENSITY, DEFAULT_ROCK_SPECIFIC_HEAT,
    DEFAULT_ROCK_THERMAL_CONDUCTIVITY, DARCY_TO_M2
)


@dataclass
class RockProperties:
    """
    Properties of the rock/porous matrix.

    Attributes:
        name: Identifier for the rock type
        porosity: Porosity [-]
        permeability: Absolute permeability tensor [m²]
        density: Rock grain density [kg/m³]
        specific_heat: Rock specific heat capacity [J/(kg·K)]
        thermal_conductivity: Rock thermal conductivity [W/(m·K)]
        compressibility: Rock compressibility [1/Pa]
        rel_perm_model: Name of relative permeability model
        rel_perm_params: Parameters for relative permeability model
        cap_pressure_model: Name of capillary pressure model
        cap_pressure_params: Parameters for capillary pressure model
    """
    name: str = "default"
    porosity: float = DEFAULT_POROSITY
    permeability: np.ndarray = field(
        default_factory=lambda: np.array([DEFAULT_PERMEABILITY] * 3)
    )
    density: float = DEFAULT_ROCK_DENSITY
    specific_heat: float = DEFAULT_ROCK_SPECIFIC_HEAT
    thermal_conductivity: float = DEFAULT_ROCK_THERMAL_CONDUCTIVITY
    compressibility: float = 0.0

    # Relative permeability model
    rel_perm_model: str = "corey"
    rel_perm_params: Dict[str, float] = field(default_factory=lambda: {
        "slr": 0.1,    # Residual liquid saturation
        "sgr": 0.05,   # Residual gas saturation
        "nl": 2.0,     # Liquid Corey exponent
        "ng": 2.0,     # Gas Corey exponent
    })

    # Capillary pressure model
    cap_pressure_model: str = "van_genuchten"
    cap_pressure_params: Dict[str, float] = field(default_factory=lambda: {
        "slr": 0.1,      # Residual liquid saturation
        "alpha": 1e-4,   # van Genuchten alpha [1/Pa]
        "m": 0.45,       # van Genuchten m
        "pmax": 1e7,     # Maximum capillary pressure [Pa]
    })

    def __post_init__(self):
        """Ensure permeability is a numpy array."""
        if np.isscalar(self.permeability):
            self.permeability = np.array([self.permeability] * 3)
        else:
            self.permeability = np.asarray(self.permeability, dtype=np.float64)
            if len(self.permeability) == 1:
                self.permeability = np.array([self.permeability[0]] * 3)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RockProperties':
        """
        Create RockProperties from dictionary.

        Args:
            data: Dictionary with property values

        Returns:
            RockProperties instance
        """
        # Handle permeability in different units
        if 'permeability_darcy' in data:
            perm = data.pop('permeability_darcy')
            if np.isscalar(perm):
                data['permeability'] = perm * DARCY_TO_M2
            else:
                data['permeability'] = np.array(perm) * DARCY_TO_M2
        elif 'permeability_md' in data:  # millidarcy
            perm = data.pop('permeability_md')
            if np.isscalar(perm):
                data['permeability'] = perm * 1e-3 * DARCY_TO_M2
            else:
                data['permeability'] = np.array(perm) * 1e-3 * DARCY_TO_M2

        return cls(**data)

    def get_isotropic_permeability(self) -> float:
        """Get scalar permeability (for isotropic case)."""
        return self.permeability[0]

    def get_directional_permeability(self, direction: int) -> float:
        """
        Get permeability in a specific direction.

        Args:
            direction: 0=x, 1=y, 2=z

        Returns:
            Permeability [m²]
        """
        return self.permeability[direction]


@dataclass
class FluidProperties:
    """
    Reference properties for a fluid phase.

    Actual properties are computed by the equation of state (EOS)
    module based on pressure and temperature. This class stores
    reference values and optional parameters.

    Attributes:
        name: Fluid identifier
        molecular_weight: Molecular weight [kg/mol]
        reference_density: Density at reference conditions [kg/m³]
        reference_viscosity: Viscosity at reference conditions [Pa·s]
        compressibility: Fluid compressibility [1/Pa]
        thermal_expansion: Thermal expansion coefficient [1/K]
    """
    name: str = "water"
    molecular_weight: float = 18.015e-3  # kg/mol
    reference_density: float = 1000.0  # kg/m³
    reference_viscosity: float = 1.0e-3  # Pa·s
    compressibility: float = 4.5e-10  # 1/Pa
    thermal_expansion: float = 2.1e-4  # 1/K


class MaterialDatabase:
    """
    Database of rock and fluid materials.

    Provides a centralized way to manage multiple rock types
    and fluid properties for a simulation.
    """

    def __init__(self):
        """Initialize empty material database."""
        self.rock_types: Dict[int, RockProperties] = {}
        self.fluids: Dict[str, FluidProperties] = {}

        # Add default materials
        self._add_defaults()

    def _add_defaults(self):
        """Add default rock and fluid materials."""
        self.add_rock(0, RockProperties(name="default"))
        self.add_fluid(FluidProperties(name="water"))
        self.add_fluid(FluidProperties(
            name="air",
            molecular_weight=28.97e-3,
            reference_density=1.225,
            reference_viscosity=1.8e-5,
            compressibility=1.0e-5,
            thermal_expansion=3.4e-3
        ))

    def add_rock(self, rock_type_id: int, properties: RockProperties):
        """
        Add a rock type to the database.

        Args:
            rock_type_id: Integer identifier for this rock type
            properties: RockProperties instance
        """
        self.rock_types[rock_type_id] = properties

    def get_rock(self, rock_type_id: int) -> RockProperties:
        """
        Get rock properties by type ID.

        Args:
            rock_type_id: Rock type identifier

        Returns:
            RockProperties for the specified type
        """
        if rock_type_id not in self.rock_types:
            raise KeyError(f"Rock type {rock_type_id} not found")
        return self.rock_types[rock_type_id]

    def add_fluid(self, properties: FluidProperties):
        """
        Add a fluid to the database.

        Args:
            properties: FluidProperties instance
        """
        self.fluids[properties.name] = properties

    def get_fluid(self, name: str) -> FluidProperties:
        """
        Get fluid properties by name.

        Args:
            name: Fluid name

        Returns:
            FluidProperties for the specified fluid
        """
        if name not in self.fluids:
            raise KeyError(f"Fluid '{name}' not found")
        return self.fluids[name]

    def get_all_rock_ids(self) -> list:
        """Get list of all rock type IDs."""
        return list(self.rock_types.keys())

    def get_all_fluid_names(self) -> list:
        """Get list of all fluid names."""
        return list(self.fluids.keys())


def create_rock_with_corey_model(
    name: str,
    porosity: float,
    permeability: float,
    slr: float = 0.1,
    sgr: float = 0.05,
    nl: float = 2.0,
    ng: float = 2.0
) -> RockProperties:
    """
    Create rock properties with Corey relative permeability model.

    Args:
        name: Rock name
        porosity: Porosity [-]
        permeability: Isotropic permeability [m²]
        slr: Residual liquid saturation
        sgr: Residual gas saturation
        nl: Liquid Corey exponent
        ng: Gas Corey exponent

    Returns:
        RockProperties instance
    """
    return RockProperties(
        name=name,
        porosity=porosity,
        permeability=np.array([permeability, permeability, permeability]),
        rel_perm_model="corey",
        rel_perm_params={"slr": slr, "sgr": sgr, "nl": nl, "ng": ng}
    )


def create_rock_with_van_genuchten(
    name: str,
    porosity: float,
    permeability: float,
    slr: float = 0.1,
    sls: float = 1.0,
    alpha: float = 1e-4,
    m: float = 0.45,
    pmax: float = 1e7
) -> RockProperties:
    """
    Create rock properties with van Genuchten models.

    Args:
        name: Rock name
        porosity: Porosity [-]
        permeability: Isotropic permeability [m²]
        slr: Residual liquid saturation
        sls: Maximum liquid saturation
        alpha: van Genuchten alpha [1/Pa]
        m: van Genuchten m parameter
        pmax: Maximum capillary pressure [Pa]

    Returns:
        RockProperties instance
    """
    return RockProperties(
        name=name,
        porosity=porosity,
        permeability=np.array([permeability, permeability, permeability]),
        rel_perm_model="van_genuchten",
        rel_perm_params={"slr": slr, "sls": sls, "m": m},
        cap_pressure_model="van_genuchten",
        cap_pressure_params={"slr": slr, "alpha": alpha, "m": m, "pmax": pmax}
    )
