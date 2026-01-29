"""
Input file parser for POROUS simulator.

Parses JSON/YAML-style input files defining mesh, properties,
initial conditions, and simulation parameters.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ..core.mesh import Mesh, create_mesh_1d, create_mesh_2d, create_mesh_3d
from ..core.properties import RockProperties, MaterialDatabase
from ..core.state import StateManager
from ..physics.relative_perm import create_relative_permeability
from ..physics.capillary import create_capillary_pressure
from ..utils.constants import DARCY_TO_M2, BAR_TO_PA, CELSIUS_OFFSET


class InputParser:
    """
    Parses input files for POROUS simulation.

    Input file format (JSON):
    {
        "mesh": {
            "type": "1d" | "2d" | "3d",
            "dimensions": {...},
            ...
        },
        "rock": {
            "default": {...},
            "regions": [...]
        },
        "initial_conditions": {...},
        "boundary_conditions": [...],
        "simulation": {...},
        "output": {...}
    }
    """

    def __init__(self):
        """Initialize parser."""
        self.data: Dict[str, Any] = {}
        self.mesh: Optional[Mesh] = None
        self.materials: Optional[MaterialDatabase] = None
        self.rock_properties: List[RockProperties] = []
        self.state_manager: Optional[StateManager] = None

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """
        Parse input file.

        Args:
            filepath: Path to input file

        Returns:
            Parsed data dictionary
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")

        with open(path, 'r') as f:
            if path.suffix == '.json':
                self.data = json.load(f)
            else:
                # Default to JSON
                self.data = json.load(f)

        return self.data

    def parse_string(self, content: str) -> Dict[str, Any]:
        """
        Parse input from string.

        Args:
            content: JSON string

        Returns:
            Parsed data dictionary
        """
        self.data = json.loads(content)
        return self.data

    def create_mesh(self) -> Mesh:
        """
        Create mesh from parsed data.

        Returns:
            Mesh object
        """
        if 'mesh' not in self.data:
            raise ValueError("No mesh definition in input")

        mesh_data = self.data['mesh']
        mesh_type = mesh_data.get('type', '1d').lower()

        if mesh_type == '1d':
            self.mesh = self._create_mesh_1d(mesh_data)
        elif mesh_type == '2d':
            self.mesh = self._create_mesh_2d(mesh_data)
        elif mesh_type == '3d':
            self.mesh = self._create_mesh_3d(mesh_data)
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}")

        return self.mesh

    def _create_mesh_1d(self, data: Dict) -> Mesh:
        """Create 1D mesh from data."""
        length = data.get('length', 100.0)
        num_cells = data.get('num_cells', 10)
        area = data.get('area', 1.0)
        origin = data.get('origin', 0.0)

        return create_mesh_1d(length, num_cells, area, origin)

    def _create_mesh_2d(self, data: Dict) -> Mesh:
        """Create 2D mesh from data."""
        lx = data.get('lx', 100.0)
        ly = data.get('ly', 100.0)
        nx = data.get('nx', 10)
        ny = data.get('ny', 10)
        thickness = data.get('thickness', 1.0)
        origin = tuple(data.get('origin', [0.0, 0.0]))

        return create_mesh_2d(lx, ly, nx, ny, thickness, origin)

    def _create_mesh_3d(self, data: Dict) -> Mesh:
        """Create 3D mesh from data."""
        lx = data.get('lx', 100.0)
        ly = data.get('ly', 100.0)
        lz = data.get('lz', 100.0)
        nx = data.get('nx', 10)
        ny = data.get('ny', 10)
        nz = data.get('nz', 10)
        origin = tuple(data.get('origin', [0.0, 0.0, 0.0]))

        return create_mesh_3d(lx, ly, lz, nx, ny, nz, origin)

    def create_materials(self) -> Tuple[MaterialDatabase, List[RockProperties]]:
        """
        Create material database and rock properties list.

        Returns:
            Tuple of (MaterialDatabase, list of RockProperties per cell)
        """
        if self.mesh is None:
            raise ValueError("Mesh must be created before materials")

        self.materials = MaterialDatabase()
        rock_data = self.data.get('rock', {})

        # Default rock properties
        default_props = self._parse_rock_properties(
            rock_data.get('default', {})
        )
        self.materials.add_rock(0, default_props)

        # Initialize all cells with default
        self.rock_properties = [default_props] * self.mesh.num_cells

        # Parse rock regions
        regions = rock_data.get('regions', [])
        for i, region_data in enumerate(regions):
            props = self._parse_rock_properties(region_data)
            rock_id = i + 1
            self.materials.add_rock(rock_id, props)

            # Apply to specified cells
            cell_indices = region_data.get('cells', [])
            for idx in cell_indices:
                if 0 <= idx < self.mesh.num_cells:
                    self.rock_properties[idx] = props
                    self.mesh.cells[idx].rock_type = rock_id

        return self.materials, self.rock_properties

    def _parse_rock_properties(self, data: Dict) -> RockProperties:
        """Parse rock properties from dictionary."""
        props = {}

        # Name
        props['name'] = data.get('name', 'default')

        # Porosity
        props['porosity'] = data.get('porosity', 0.1)

        # Permeability (handle various units)
        if 'permeability' in data:
            perm = data['permeability']
            if isinstance(perm, dict):
                # With units
                value = perm.get('value', 1e-15)
                unit = perm.get('unit', 'm2')
                if unit in ['darcy', 'D']:
                    value *= DARCY_TO_M2
                elif unit in ['millidarcy', 'mD']:
                    value *= 1e-3 * DARCY_TO_M2
                props['permeability'] = np.array([value, value, value])
            else:
                props['permeability'] = np.array([perm, perm, perm])
        elif 'permeability_darcy' in data:
            perm = data['permeability_darcy'] * DARCY_TO_M2
            props['permeability'] = np.array([perm, perm, perm])

        # Density
        props['density'] = data.get('density', 2650.0)

        # Thermal properties
        props['specific_heat'] = data.get('specific_heat', 1000.0)
        props['thermal_conductivity'] = data.get('thermal_conductivity', 2.0)

        # Relative permeability model
        rel_perm_data = data.get('relative_permeability', {})
        props['rel_perm_model'] = rel_perm_data.get('model', 'corey')
        props['rel_perm_params'] = rel_perm_data.get('params', {
            'slr': 0.1, 'sgr': 0.05, 'nl': 2.0, 'ng': 2.0
        })

        # Capillary pressure model
        cap_data = data.get('capillary_pressure', {})
        props['cap_pressure_model'] = cap_data.get('model', 'van_genuchten')
        props['cap_pressure_params'] = cap_data.get('params', {
            'slr': 0.1, 'alpha': 1e-4, 'm': 0.45, 'pmax': 1e7
        })

        return RockProperties(**props)

    def create_state_manager(self) -> StateManager:
        """
        Create and initialize state manager.

        Returns:
            StateManager object
        """
        if self.mesh is None:
            raise ValueError("Mesh must be created first")
        if not self.rock_properties:
            raise ValueError("Materials must be created first")

        # Create default constitutive models
        default_rock = self.rock_properties[0]
        rel_perm = create_relative_permeability(
            default_rock.rel_perm_model,
            default_rock.rel_perm_params
        )
        cap_pressure = create_capillary_pressure(
            default_rock.cap_pressure_model,
            default_rock.cap_pressure_params
        )

        self.state_manager = StateManager(
            self.mesh.num_cells,
            default_rel_perm=rel_perm,
            default_cap_pressure=cap_pressure
        )

        # Set cell-specific models where different
        for i, rock in enumerate(self.rock_properties):
            if rock != default_rock:
                cell_rel_perm = create_relative_permeability(
                    rock.rel_perm_model,
                    rock.rel_perm_params
                )
                cell_cap_pressure = create_capillary_pressure(
                    rock.cap_pressure_model,
                    rock.cap_pressure_params
                )
                self.state_manager.set_cell_models(i, cell_rel_perm, cell_cap_pressure)

        # Apply initial conditions
        self._apply_initial_conditions()

        return self.state_manager

    def _apply_initial_conditions(self):
        """Apply initial conditions from input data."""
        ic_data = self.data.get('initial_conditions', {})

        # Default uniform conditions
        pressure = ic_data.get('pressure', 101325.0)
        temperature = ic_data.get('temperature', 293.15)
        saturation = ic_data.get('liquid_saturation', 1.0)

        # Handle units
        if isinstance(pressure, dict):
            value = pressure.get('value', 101325.0)
            unit = pressure.get('unit', 'Pa')
            if unit == 'bar':
                value *= BAR_TO_PA
            elif unit == 'MPa':
                value *= 1e6
            pressure = value

        if isinstance(temperature, dict):
            value = temperature.get('value', 20.0)
            unit = temperature.get('unit', 'C')
            if unit in ['C', 'celsius']:
                value += CELSIUS_OFFSET
            temperature = value

        self.state_manager.initialize_uniform(pressure, temperature, saturation)

        # Apply regional initial conditions
        regions = ic_data.get('regions', [])
        for region in regions:
            cells = region.get('cells', [])
            p = region.get('pressure', pressure)
            T = region.get('temperature', temperature)
            s = region.get('liquid_saturation', saturation)

            for idx in cells:
                if 0 <= idx < self.mesh.num_cells:
                    state = self.state_manager.states[idx]
                    state.set_primary_variables(p, T if s >= 0.999 else s)

    def get_simulation_params(self) -> Dict[str, Any]:
        """
        Get simulation parameters.

        Returns:
            Dictionary of simulation parameters
        """
        sim_data = self.data.get('simulation', {})

        params = {
            'end_time': sim_data.get('end_time', 86400.0),  # 1 day default
            'dt_initial': sim_data.get('dt_initial', 1.0),
            'dt_min': sim_data.get('dt_min', 1e-6),
            'dt_max': sim_data.get('dt_max', 1e8),
            'newton_tolerance': sim_data.get('newton_tolerance', 1e-6),
            'max_newton_iterations': sim_data.get('max_newton_iterations', 20),
            'include_energy': sim_data.get('include_energy', False),
        }

        return params

    def get_source_terms(self) -> Optional[np.ndarray]:
        """
        Get source/sink terms.

        Returns:
            Source terms array or None
        """
        source_data = self.data.get('sources', [])

        if not source_data:
            return None

        if self.mesh is None:
            raise ValueError("Mesh must be created first")

        num_eq = 3 if self.data.get('simulation', {}).get('include_energy', False) else 2
        sources = np.zeros((self.mesh.num_cells, num_eq))

        for source in source_data:
            cell = source.get('cell', 0)
            mass_rate = source.get('mass_rate', 0.0)  # kg/s
            enthalpy = source.get('enthalpy', 0.0)  # J/kg

            if 0 <= cell < self.mesh.num_cells:
                # Assume liquid injection/production
                sources[cell, 0] = mass_rate

                if num_eq > 2:
                    sources[cell, 2] = mass_rate * enthalpy

        return sources

    def get_output_params(self) -> Dict[str, Any]:
        """
        Get output parameters.

        Returns:
            Dictionary of output parameters
        """
        output_data = self.data.get('output', {})

        params = {
            'format': output_data.get('format', 'vtk'),
            'output_dir': output_data.get('directory', './output'),
            'output_times': output_data.get('times', []),
            'output_interval': output_data.get('interval', 0),
            'variables': output_data.get('variables', ['pressure', 'saturation', 'temperature']),
        }

        return params


def create_simple_input(
    mesh_type: str = '1d',
    length: float = 100.0,
    num_cells: int = 20,
    porosity: float = 0.1,
    permeability: float = 1e-14,
    initial_pressure: float = 1e5,
    initial_temperature: float = 293.15,
    end_time: float = 86400.0
) -> Dict[str, Any]:
    """
    Create a simple input dictionary for quick setup.

    Args:
        mesh_type: Mesh type ('1d', '2d', '3d')
        length: Domain length [m]
        num_cells: Number of cells
        porosity: Rock porosity [-]
        permeability: Rock permeability [m²]
        initial_pressure: Initial pressure [Pa]
        initial_temperature: Initial temperature [K]
        end_time: Simulation end time [s]

    Returns:
        Input dictionary
    """
    return {
        'mesh': {
            'type': mesh_type,
            'length': length,
            'num_cells': num_cells,
            'area': 1.0
        },
        'rock': {
            'default': {
                'name': 'default',
                'porosity': porosity,
                'permeability': permeability,
            }
        },
        'initial_conditions': {
            'pressure': initial_pressure,
            'temperature': initial_temperature,
            'liquid_saturation': 1.0
        },
        'simulation': {
            'end_time': end_time,
            'dt_initial': 1.0,
            'include_energy': False
        }
    }
