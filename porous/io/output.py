"""
Output writers for simulation results.

Supports VTK format for visualization and CSV/NPY for data analysis.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import struct

if TYPE_CHECKING:
    from ..core.mesh import Mesh
    from ..core.state import StateManager


class OutputWriter:
    """
    Base class for output writers.

    Handles common output operations like directory creation
    and time tracking.
    """

    def __init__(self, output_dir: str = './output'):
        """
        Initialize output writer.

        Args:
            output_dir: Output directory path
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_count = 0

    def write(self, mesh: 'Mesh', state_manager: 'StateManager',
             time: float, **kwargs):
        """
        Write output (to be implemented by subclasses).

        Args:
            mesh: Mesh object
            state_manager: StateManager object
            time: Current simulation time [s]
            **kwargs: Additional data to write
        """
        raise NotImplementedError


class VTKWriter(OutputWriter):
    """
    Writes simulation results in VTK format.

    Creates VTK files compatible with ParaView and other
    VTK-based visualization tools.
    """

    def __init__(self, output_dir: str = './output',
                 base_name: str = 'porous'):
        """
        Initialize VTK writer.

        Args:
            output_dir: Output directory path
            base_name: Base name for output files
        """
        super().__init__(output_dir)
        self.base_name = base_name

    def write(self, mesh: 'Mesh', state_manager: 'StateManager',
             time: float, **kwargs):
        """
        Write VTK file.

        Args:
            mesh: Mesh object
            state_manager: StateManager object
            time: Current simulation time [s]
            **kwargs: Additional scalar fields to write
        """
        filename = self.output_dir / f"{self.base_name}_{self.file_count:04d}.vtk"

        if mesh.dimension == 1:
            self._write_vtk_1d(filename, mesh, state_manager, time, **kwargs)
        elif mesh.dimension == 2:
            self._write_vtk_2d(filename, mesh, state_manager, time, **kwargs)
        else:
            self._write_vtk_3d(filename, mesh, state_manager, time, **kwargs)

        self.file_count += 1

    def _write_vtk_1d(self, filename: Path, mesh: 'Mesh',
                     state_manager: 'StateManager', time: float, **kwargs):
        """Write 1D data as VTK polydata."""
        centers = mesh.get_cell_centers()

        with open(filename, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"POROUS output time={time:.6e}\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            # Points (cell centers)
            n = mesh.num_cells
            f.write(f"POINTS {n} float\n")
            for i in range(n):
                f.write(f"{centers[i, 0]:.6e} {centers[i, 1]:.6e} {centers[i, 2]:.6e}\n")

            # Vertices
            f.write(f"VERTICES {n} {2*n}\n")
            for i in range(n):
                f.write(f"1 {i}\n")

            # Cell data
            f.write(f"POINT_DATA {n}\n")

            # Pressure
            f.write("SCALARS Pressure float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for state in state_manager.states:
                f.write(f"{state.pressure:.6e}\n")

            # Temperature
            f.write("SCALARS Temperature float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for state in state_manager.states:
                f.write(f"{state.temperature:.6e}\n")

            # Liquid saturation
            f.write("SCALARS LiquidSaturation float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for state in state_manager.states:
                f.write(f"{state.liquid_saturation:.6e}\n")

            # Gas saturation
            f.write("SCALARS GasSaturation float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for state in state_manager.states:
                f.write(f"{state.gas_saturation:.6e}\n")

            # Additional fields
            for name, values in kwargs.items():
                if len(values) == n:
                    f.write(f"SCALARS {name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for v in values:
                        f.write(f"{v:.6e}\n")

    def _write_vtk_2d(self, filename: Path, mesh: 'Mesh',
                     state_manager: 'StateManager', time: float, **kwargs):
        """Write 2D data as VTK structured grid."""
        nx, ny = mesh.nx, mesh.ny

        with open(filename, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"POROUS output time={time:.6e}\n")
            f.write("ASCII\n")
            f.write("DATASET RECTILINEAR_GRID\n")
            f.write(f"DIMENSIONS {nx+1} {ny+1} 1\n")

            # X coordinates
            centers = mesh.get_cell_centers()
            x_coords = np.zeros(nx + 1)
            if nx > 0:
                dx = centers[1, 0] - centers[0, 0] if nx > 1 else 1.0
                x_coords[0] = centers[0, 0] - dx/2
                for i in range(nx):
                    x_coords[i+1] = centers[i, 0] + dx/2

            f.write(f"X_COORDINATES {nx+1} float\n")
            for x in x_coords:
                f.write(f"{x:.6e} ")
            f.write("\n")

            # Y coordinates
            y_coords = np.zeros(ny + 1)
            if ny > 0:
                dy = centers[nx, 1] - centers[0, 1] if ny > 1 else 1.0
                y_coords[0] = centers[0, 1] - dy/2
                for j in range(ny):
                    y_coords[j+1] = centers[j*nx, 1] + dy/2

            f.write(f"Y_COORDINATES {ny+1} float\n")
            for y in y_coords:
                f.write(f"{y:.6e} ")
            f.write("\n")

            # Z coordinates (single layer)
            f.write("Z_COORDINATES 1 float\n")
            f.write("0.0\n")

            # Cell data
            n = mesh.num_cells
            f.write(f"CELL_DATA {n}\n")

            # Pressure
            f.write("SCALARS Pressure float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for state in state_manager.states:
                f.write(f"{state.pressure:.6e}\n")

            # Temperature
            f.write("SCALARS Temperature float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for state in state_manager.states:
                f.write(f"{state.temperature:.6e}\n")

            # Liquid saturation
            f.write("SCALARS LiquidSaturation float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for state in state_manager.states:
                f.write(f"{state.liquid_saturation:.6e}\n")

            # Additional fields
            for name, values in kwargs.items():
                if len(values) == n:
                    f.write(f"SCALARS {name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for v in values:
                        f.write(f"{v:.6e}\n")

    def _write_vtk_3d(self, filename: Path, mesh: 'Mesh',
                     state_manager: 'StateManager', time: float, **kwargs):
        """Write 3D data as VTK structured grid."""
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
        centers = mesh.get_cell_centers()

        with open(filename, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"POROUS output time={time:.6e}\n")
            f.write("ASCII\n")
            f.write("DATASET RECTILINEAR_GRID\n")
            f.write(f"DIMENSIONS {nx+1} {ny+1} {nz+1}\n")

            # Compute grid coordinates from cell centers
            dx = centers[1, 0] - centers[0, 0] if nx > 1 else 1.0
            dy = centers[nx, 1] - centers[0, 1] if ny > 1 else 1.0
            dz = centers[nx*ny, 2] - centers[0, 2] if nz > 1 else 1.0

            # X coordinates
            f.write(f"X_COORDINATES {nx+1} float\n")
            x0 = centers[0, 0] - dx/2
            for i in range(nx + 1):
                f.write(f"{x0 + i*dx:.6e} ")
            f.write("\n")

            # Y coordinates
            f.write(f"Y_COORDINATES {ny+1} float\n")
            y0 = centers[0, 1] - dy/2
            for j in range(ny + 1):
                f.write(f"{y0 + j*dy:.6e} ")
            f.write("\n")

            # Z coordinates
            f.write(f"Z_COORDINATES {nz+1} float\n")
            z0 = centers[0, 2] - dz/2
            for k in range(nz + 1):
                f.write(f"{z0 + k*dz:.6e} ")
            f.write("\n")

            # Cell data
            n = mesh.num_cells
            f.write(f"CELL_DATA {n}\n")

            # Pressure
            f.write("SCALARS Pressure float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for state in state_manager.states:
                f.write(f"{state.pressure:.6e}\n")

            # Temperature
            f.write("SCALARS Temperature float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for state in state_manager.states:
                f.write(f"{state.temperature:.6e}\n")

            # Saturations
            f.write("SCALARS LiquidSaturation float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for state in state_manager.states:
                f.write(f"{state.liquid_saturation:.6e}\n")

            # Additional fields
            for name, values in kwargs.items():
                if len(values) == n:
                    f.write(f"SCALARS {name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for v in values:
                        f.write(f"{v:.6e}\n")


class CSVWriter(OutputWriter):
    """
    Writes simulation results in CSV format.

    Creates CSV files with cell data for easy analysis
    in spreadsheet applications or pandas.
    """

    def __init__(self, output_dir: str = './output',
                 base_name: str = 'porous'):
        """
        Initialize CSV writer.

        Args:
            output_dir: Output directory path
            base_name: Base name for output files
        """
        super().__init__(output_dir)
        self.base_name = base_name

    def write(self, mesh: 'Mesh', state_manager: 'StateManager',
             time: float, **kwargs):
        """
        Write CSV file.

        Args:
            mesh: Mesh object
            state_manager: StateManager object
            time: Current simulation time [s]
            **kwargs: Additional fields to write
        """
        filename = self.output_dir / f"{self.base_name}_{self.file_count:04d}.csv"

        centers = mesh.get_cell_centers()

        with open(filename, 'w') as f:
            # Header
            headers = ['cell', 'x', 'y', 'z', 'pressure', 'temperature',
                      'liquid_saturation', 'gas_saturation']
            headers.extend(kwargs.keys())
            f.write(','.join(headers) + '\n')

            # Data
            for i, state in enumerate(state_manager.states):
                row = [
                    str(i),
                    f"{centers[i, 0]:.6e}",
                    f"{centers[i, 1]:.6e}",
                    f"{centers[i, 2]:.6e}",
                    f"{state.pressure:.6e}",
                    f"{state.temperature:.6e}",
                    f"{state.liquid_saturation:.6e}",
                    f"{state.gas_saturation:.6e}",
                ]
                for name, values in kwargs.items():
                    row.append(f"{values[i]:.6e}")
                f.write(','.join(row) + '\n')

        self.file_count += 1


class TimeHistoryWriter:
    """
    Writes time history data for selected cells.

    Useful for monitoring specific locations over time.
    """

    def __init__(self, output_dir: str = './output',
                 filename: str = 'history.csv',
                 cell_indices: List[int] = None):
        """
        Initialize time history writer.

        Args:
            output_dir: Output directory path
            filename: Output filename
            cell_indices: List of cell indices to track
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename
        self.cell_indices = cell_indices or [0]
        self.first_write = True

    def write(self, state_manager: 'StateManager', time: float):
        """
        Append time step data to history file.

        Args:
            state_manager: StateManager object
            time: Current simulation time [s]
        """
        mode = 'w' if self.first_write else 'a'

        with open(self.filepath, mode) as f:
            if self.first_write:
                # Header
                headers = ['time']
                for idx in self.cell_indices:
                    headers.extend([
                        f'P_{idx}',
                        f'T_{idx}',
                        f'Sl_{idx}'
                    ])
                f.write(','.join(headers) + '\n')
                self.first_write = False

            # Data row
            row = [f"{time:.6e}"]
            for idx in self.cell_indices:
                state = state_manager.states[idx]
                row.extend([
                    f"{state.pressure:.6e}",
                    f"{state.temperature:.6e}",
                    f"{state.liquid_saturation:.6e}"
                ])
            f.write(','.join(row) + '\n')


def save_results_numpy(filename: str, results: Dict[str, np.ndarray]):
    """
    Save simulation results to numpy file.

    Args:
        filename: Output filename (.npz)
        results: Dictionary of result arrays
    """
    np.savez(filename, **results)


def load_results_numpy(filename: str) -> Dict[str, np.ndarray]:
    """
    Load simulation results from numpy file.

    Args:
        filename: Input filename (.npz)

    Returns:
        Dictionary of result arrays
    """
    data = np.load(filename)
    return {key: data[key] for key in data.files}
