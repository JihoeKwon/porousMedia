"""Physics modules for equations of state, flow, and constitutive relations."""

from .eos import WaterEOS, SteamEOS, AirEOS, saturation_pressure, saturation_temperature
from .relative_perm import RelativePermeability, CoreyRelPerm, VanGenuchtenRelPerm
from .capillary import CapillaryPressure, VanGenuchtenCapillary, BrooksCoreyCapillary
from .flow import compute_mass_flux, compute_energy_flux, assemble_residual
