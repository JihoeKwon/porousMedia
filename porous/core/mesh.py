"""
Mesh generation and management for porous media simulation.

Supports 1D, 2D, and 3D structured grids with cell-centered discretization.
Uses integral finite difference method (IFDM) formulation compatible with TOUGH2.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class Cell:
    """
    Represents a single cell (element) in the mesh.

    Attributes:
        index: Global cell index
        volume: Cell volume [m³]
        center: Cell center coordinates (x, y, z) [m]
        rock_type: Index of rock type for this cell
        active: Whether cell is active in simulation
    """
    index: int
    volume: float
    center: np.ndarray
    rock_type: int = 0
    active: bool = True

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64)


@dataclass
class Connection:
    """
    Represents a connection (interface) between two cells.

    The connection stores geometric information needed for flux calculations
    using the integral finite difference method.

    Attributes:
        index: Global connection index
        cell1: Index of first cell
        cell2: Index of second cell
        area: Interface area [m²]
        distance1: Distance from cell1 center to interface [m]
        distance2: Distance from cell2 center to interface [m]
        direction: Direction vector from cell1 to cell2 (unit vector)
        beta: Cosine of angle between connection and vertical (for gravity)
    """
    index: int
    cell1: int
    cell2: int
    area: float
    distance1: float
    distance2: float
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    beta: float = 0.0  # cos(angle with gravity direction)

    def __post_init__(self):
        self.direction = np.asarray(self.direction, dtype=np.float64)

    @property
    def total_distance(self) -> float:
        """Total distance between cell centers."""
        return self.distance1 + self.distance2


class Mesh:
    """
    Manages the computational mesh for porous media simulation.

    The mesh consists of cells (control volumes) and connections
    (interfaces between cells). Uses a cell-centered finite volume
    formulation compatible with the integral finite difference method.

    Attributes:
        cells: List of Cell objects
        connections: List of Connection objects
        num_cells: Number of cells
        num_connections: Number of connections
        dimension: Mesh dimension (1, 2, or 3)
    """

    def __init__(self, dimension: int = 1):
        """
        Initialize empty mesh.

        Args:
            dimension: Spatial dimension (1, 2, or 3)
        """
        self.cells: List[Cell] = []
        self.connections: List[Connection] = []
        self.dimension = dimension
        self._cell_to_connections: Dict[int, List[int]] = {}

        # Grid structure info (for structured grids)
        self.nx: int = 0
        self.ny: int = 0
        self.nz: int = 0

    @property
    def num_cells(self) -> int:
        """Number of cells in mesh."""
        return len(self.cells)

    @property
    def num_connections(self) -> int:
        """Number of connections in mesh."""
        return len(self.connections)

    def add_cell(self, volume: float, center: np.ndarray,
                 rock_type: int = 0, active: bool = True) -> int:
        """
        Add a cell to the mesh.

        Args:
            volume: Cell volume [m³]
            center: Cell center coordinates [m]
            rock_type: Rock type index
            active: Whether cell is active

        Returns:
            Index of newly added cell
        """
        index = len(self.cells)
        cell = Cell(index=index, volume=volume, center=center,
                   rock_type=rock_type, active=active)
        self.cells.append(cell)
        self._cell_to_connections[index] = []
        return index

    def add_connection(self, cell1: int, cell2: int, area: float,
                      distance1: float, distance2: float,
                      direction: Optional[np.ndarray] = None,
                      beta: Optional[float] = None) -> int:
        """
        Add a connection between two cells.

        Args:
            cell1: Index of first cell
            cell2: Index of second cell
            area: Interface area [m²]
            distance1: Distance from cell1 to interface [m]
            distance2: Distance from cell2 to interface [m]
            direction: Unit vector from cell1 to cell2
            beta: Cosine of angle with gravity (computed if None)

        Returns:
            Index of newly added connection
        """
        index = len(self.connections)

        # Compute direction if not provided
        if direction is None:
            c1 = self.cells[cell1].center
            c2 = self.cells[cell2].center
            diff = c2 - c1
            dist = np.linalg.norm(diff)
            direction = diff / dist if dist > 0 else np.array([1.0, 0.0, 0.0])

        # Compute beta (gravity factor) if not provided
        # Assumes gravity acts in negative z-direction
        if beta is None:
            gravity_dir = np.array([0.0, 0.0, -1.0])
            beta = np.dot(direction, gravity_dir)

        connection = Connection(
            index=index, cell1=cell1, cell2=cell2,
            area=area, distance1=distance1, distance2=distance2,
            direction=direction, beta=beta
        )
        self.connections.append(connection)

        # Update cell-to-connection mapping
        self._cell_to_connections[cell1].append(index)
        self._cell_to_connections[cell2].append(index)

        return index

    def get_cell_connections(self, cell_index: int) -> List[int]:
        """
        Get indices of all connections involving a cell.

        Args:
            cell_index: Cell index

        Returns:
            List of connection indices
        """
        return self._cell_to_connections.get(cell_index, [])

    def get_neighbor_cells(self, cell_index: int) -> List[int]:
        """
        Get indices of all cells connected to a given cell.

        Args:
            cell_index: Cell index

        Returns:
            List of neighboring cell indices
        """
        neighbors = []
        for conn_idx in self.get_cell_connections(cell_index):
            conn = self.connections[conn_idx]
            if conn.cell1 == cell_index:
                neighbors.append(conn.cell2)
            else:
                neighbors.append(conn.cell1)
        return neighbors

    def get_cell_volumes(self) -> np.ndarray:
        """Get array of all cell volumes."""
        return np.array([cell.volume for cell in self.cells])

    def get_cell_centers(self) -> np.ndarray:
        """Get array of all cell center coordinates."""
        return np.array([cell.center for cell in self.cells])

    def cell_index_3d(self, i: int, j: int, k: int) -> int:
        """
        Convert 3D grid indices to cell index.

        Args:
            i: Index in x-direction (0 to nx-1)
            j: Index in y-direction (0 to ny-1)
            k: Index in z-direction (0 to nz-1)

        Returns:
            Global cell index
        """
        return i + j * self.nx + k * self.nx * self.ny

    def cell_indices_from_global(self, index: int) -> Tuple[int, int, int]:
        """
        Convert global cell index to 3D grid indices.

        Args:
            index: Global cell index

        Returns:
            Tuple (i, j, k) of grid indices
        """
        k = index // (self.nx * self.ny)
        remainder = index % (self.nx * self.ny)
        j = remainder // self.nx
        i = remainder % self.nx
        return i, j, k


def create_mesh_1d(length: float, num_cells: int,
                   area: float = 1.0,
                   origin: float = 0.0) -> Mesh:
    """
    Create a 1D mesh.

    Args:
        length: Total length of domain [m]
        num_cells: Number of cells
        area: Cross-sectional area [m²]
        origin: Starting coordinate [m]

    Returns:
        Mesh object
    """
    mesh = Mesh(dimension=1)
    mesh.nx = num_cells
    mesh.ny = 1
    mesh.nz = 1

    dx = length / num_cells

    # Create cells
    for i in range(num_cells):
        x = origin + (i + 0.5) * dx
        center = np.array([x, 0.0, 0.0])
        volume = area * dx
        mesh.add_cell(volume=volume, center=center)

    # Create connections
    for i in range(num_cells - 1):
        mesh.add_connection(
            cell1=i, cell2=i+1,
            area=area,
            distance1=dx/2, distance2=dx/2,
            direction=np.array([1.0, 0.0, 0.0]),
            beta=0.0  # Horizontal, no gravity component
        )

    return mesh


def create_mesh_2d(lx: float, ly: float,
                   nx: int, ny: int,
                   thickness: float = 1.0,
                   origin: Tuple[float, float] = (0.0, 0.0)) -> Mesh:
    """
    Create a 2D mesh in the x-y plane.

    Args:
        lx: Domain length in x-direction [m]
        ly: Domain length in y-direction [m]
        nx: Number of cells in x-direction
        ny: Number of cells in y-direction
        thickness: Thickness in z-direction [m]
        origin: Origin coordinates (x0, y0) [m]

    Returns:
        Mesh object
    """
    mesh = Mesh(dimension=2)
    mesh.nx = nx
    mesh.ny = ny
    mesh.nz = 1

    dx = lx / nx
    dy = ly / ny
    x0, y0 = origin

    # Create cells
    for j in range(ny):
        for i in range(nx):
            x = x0 + (i + 0.5) * dx
            y = y0 + (j + 0.5) * dy
            center = np.array([x, y, 0.0])
            volume = dx * dy * thickness
            mesh.add_cell(volume=volume, center=center)

    # Create connections in x-direction
    for j in range(ny):
        for i in range(nx - 1):
            cell1 = i + j * nx
            cell2 = cell1 + 1
            mesh.add_connection(
                cell1=cell1, cell2=cell2,
                area=dy * thickness,
                distance1=dx/2, distance2=dx/2,
                direction=np.array([1.0, 0.0, 0.0]),
                beta=0.0
            )

    # Create connections in y-direction
    for j in range(ny - 1):
        for i in range(nx):
            cell1 = i + j * nx
            cell2 = cell1 + nx
            mesh.add_connection(
                cell1=cell1, cell2=cell2,
                area=dx * thickness,
                distance1=dy/2, distance2=dy/2,
                direction=np.array([0.0, 1.0, 0.0]),
                beta=0.0
            )

    return mesh


def create_mesh_3d(lx: float, ly: float, lz: float,
                   nx: int, ny: int, nz: int,
                   origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Mesh:
    """
    Create a 3D mesh.

    Args:
        lx: Domain length in x-direction [m]
        ly: Domain length in y-direction [m]
        lz: Domain length in z-direction [m]
        nx: Number of cells in x-direction
        ny: Number of cells in y-direction
        nz: Number of cells in z-direction
        origin: Origin coordinates (x0, y0, z0) [m]

    Returns:
        Mesh object
    """
    mesh = Mesh(dimension=3)
    mesh.nx = nx
    mesh.ny = ny
    mesh.nz = nz

    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    x0, y0, z0 = origin

    # Create cells
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = x0 + (i + 0.5) * dx
                y = y0 + (j + 0.5) * dy
                z = z0 + (k + 0.5) * dz
                center = np.array([x, y, z])
                volume = dx * dy * dz
                mesh.add_cell(volume=volume, center=center)

    # Create connections in x-direction
    for k in range(nz):
        for j in range(ny):
            for i in range(nx - 1):
                cell1 = mesh.cell_index_3d(i, j, k)
                cell2 = mesh.cell_index_3d(i+1, j, k)
                mesh.add_connection(
                    cell1=cell1, cell2=cell2,
                    area=dy * dz,
                    distance1=dx/2, distance2=dx/2,
                    direction=np.array([1.0, 0.0, 0.0]),
                    beta=0.0
                )

    # Create connections in y-direction
    for k in range(nz):
        for j in range(ny - 1):
            for i in range(nx):
                cell1 = mesh.cell_index_3d(i, j, k)
                cell2 = mesh.cell_index_3d(i, j+1, k)
                mesh.add_connection(
                    cell1=cell1, cell2=cell2,
                    area=dx * dz,
                    distance1=dy/2, distance2=dy/2,
                    direction=np.array([0.0, 1.0, 0.0]),
                    beta=0.0
                )

    # Create connections in z-direction
    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                cell1 = mesh.cell_index_3d(i, j, k)
                cell2 = mesh.cell_index_3d(i, j, k+1)
                mesh.add_connection(
                    cell1=cell1, cell2=cell2,
                    area=dx * dy,
                    distance1=dz/2, distance2=dz/2,
                    direction=np.array([0.0, 0.0, 1.0]),
                    beta=-1.0  # Gravity acts in -z direction
                )

    return mesh


def create_radial_mesh_1d(radii: np.ndarray, height: float = 1.0,
                          angle: float = 2*np.pi) -> Mesh:
    """
    Create a 1D radial mesh for cylindrical geometry.

    Args:
        radii: Array of cell boundary radii [m] (n+1 values for n cells)
        height: Height of cylinder [m]
        angle: Angular extent [rad] (default: full cylinder)

    Returns:
        Mesh object
    """
    mesh = Mesh(dimension=1)
    num_cells = len(radii) - 1
    mesh.nx = num_cells
    mesh.ny = 1
    mesh.nz = 1

    # Create cells
    for i in range(num_cells):
        r_inner = radii[i]
        r_outer = radii[i+1]
        r_center = (r_inner + r_outer) / 2
        volume = angle * height * (r_outer**2 - r_inner**2) / 2
        center = np.array([r_center, 0.0, 0.0])
        mesh.add_cell(volume=volume, center=center)

    # Create connections
    for i in range(num_cells - 1):
        r_interface = radii[i+1]
        area = angle * height * r_interface

        # Distances to interface
        r1 = (radii[i] + radii[i+1]) / 2  # center of cell i
        r2 = (radii[i+1] + radii[i+2]) / 2  # center of cell i+1
        d1 = r_interface - r1
        d2 = r2 - r_interface

        mesh.add_connection(
            cell1=i, cell2=i+1,
            area=area,
            distance1=d1, distance2=d2,
            direction=np.array([1.0, 0.0, 0.0]),
            beta=0.0
        )

    return mesh
