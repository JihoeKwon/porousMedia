"""Core modules for mesh, properties, and state management."""

from .mesh import Cell, Connection, Mesh, create_mesh_1d, create_mesh_2d, create_mesh_3d
from .properties import RockProperties, FluidProperties
from .state import State, PhaseState
