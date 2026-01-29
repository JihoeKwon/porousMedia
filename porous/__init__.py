"""
POROUS: Porous Media Multiphase Flow Simulator

A Python-based simulator for multiphase flow in porous media,
similar to TOUGH2. Supports water-steam-air systems with
1D/2D/3D structured grids.
"""

__version__ = "0.1.0"
__author__ = "권지회 (Jihoe Kwon), 한국지질자원연구원 AI융합연구실"
__copyright__ = "Copyright (c) 2025 권지회, 한국지질자원연구원. All rights reserved."

from . import core
from . import physics
from . import solver
from . import io
from . import utils
