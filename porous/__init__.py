"""
POROUS: Porous Media Multiphase Flow Simulator

A Python-based simulator for multiphase flow in porous media,
similar to TOUGH2. Supports water-steam-air systems with
1D/2D/3D structured grids.
"""

__version__ = "0.1.0"
__author__ = "POROUS Development Team"

from . import core
from . import physics
from . import solver
from . import io
from . import utils
