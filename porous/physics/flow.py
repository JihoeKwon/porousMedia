"""
Multiphase flow equations for porous media.

Implements Darcy's law based flux calculations and residual assembly
for the mass and energy conservation equations.
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple
from dataclasses import dataclass

from ..utils.constants import GRAVITY, EPSILON, PHASE_LIQUID, PHASE_GAS

if TYPE_CHECKING:
    from ..core.mesh import Mesh, Connection
    from ..core.state import State
    from ..core.properties import RockProperties


@dataclass
class PhaseFlux:
    """
    Container for phase flux at an interface.

    Attributes:
        mass_flux: Mass flux [kg/s]
        enthalpy_flux: Enthalpy flux [J/s]
        upstream_cell: Index of upstream cell
    """
    mass_flux: float
    enthalpy_flux: float
    upstream_cell: int


def compute_phase_mobility(
    density: float,
    viscosity: float,
    rel_perm: float,
    permeability: float
) -> float:
    """
    Calculate phase mobility.

    mobility = k * kr / mu

    Args:
        density: Phase density [kg/m³]
        viscosity: Phase dynamic viscosity [Pa·s]
        rel_perm: Phase relative permeability [-]
        permeability: Absolute permeability [m²]

    Returns:
        Phase mobility [m²/(Pa·s)]
    """
    if viscosity < EPSILON:
        return 0.0
    return permeability * rel_perm / viscosity


def compute_phase_flux_darcy(
    mobility1: float,
    mobility2: float,
    density1: float,
    density2: float,
    pressure1: float,
    pressure2: float,
    area: float,
    distance1: float,
    distance2: float,
    beta: float = 0.0
) -> Tuple[float, int]:
    """
    Calculate phase mass flux using Darcy's law with upstream weighting.

    F = -A * lambda * (dP/dx - rho*g*cos(theta))

    Args:
        mobility1: Phase mobility in cell 1 [m²/(Pa·s)]
        mobility2: Phase mobility in cell 2 [m²/(Pa·s)]
        density1: Phase density in cell 1 [kg/m³]
        density2: Phase density in cell 2 [kg/m³]
        pressure1: Phase pressure in cell 1 [Pa]
        pressure2: Phase pressure in cell 2 [Pa]
        area: Interface area [m²]
        distance1: Distance from cell 1 center to interface [m]
        distance2: Distance from cell 2 center to interface [m]
        beta: cos(angle) with gravity direction (negative z)

    Returns:
        Tuple of (mass flux [kg/s], upstream cell index 0 or 1)
    """
    total_distance = distance1 + distance2
    if total_distance < EPSILON:
        return 0.0, 0

    # Harmonic average of mobility for interface
    if mobility1 < EPSILON and mobility2 < EPSILON:
        return 0.0, 0

    # Pressure gradient (positive from cell 1 to cell 2)
    dp_dx = (pressure2 - pressure1) / total_distance

    # Gravity term (average density)
    rho_avg = 0.5 * (density1 + density2)
    gravity_term = rho_avg * GRAVITY * beta

    # Total driving force
    driving_force = dp_dx - gravity_term

    # Upstream weighting for mobility
    if driving_force >= 0:
        # Flow from cell 1 to cell 2
        mobility = mobility1
        density = density1
        upstream = 0
    else:
        # Flow from cell 2 to cell 1
        mobility = mobility2
        density = density2
        upstream = 1

    # Transmissibility
    # Using harmonic average distance weighting
    trans = area * mobility / total_distance

    # Mass flux (positive from cell 1 to cell 2)
    mass_flux = -trans * (pressure2 - pressure1 - rho_avg * GRAVITY * beta * total_distance)

    return mass_flux * density / (density + EPSILON), upstream


def compute_mass_flux(
    state1: 'State',
    state2: 'State',
    rock1: 'RockProperties',
    rock2: 'RockProperties',
    connection: 'Connection',
    phase: int,
    direction: int = 0
) -> PhaseFlux:
    """
    Calculate mass and enthalpy flux for a phase across a connection.

    Args:
        state1: State of cell 1
        state2: State of cell 2
        rock1: Rock properties of cell 1
        rock2: Rock properties of cell 2
        connection: Connection object
        phase: Phase index (PHASE_LIQUID or PHASE_GAS)
        direction: Direction for permeability tensor (0=x, 1=y, 2=z)

    Returns:
        PhaseFlux object
    """
    # Get permeabilities
    k1 = rock1.get_directional_permeability(direction)
    k2 = rock2.get_directional_permeability(direction)

    # Harmonic average permeability
    d1 = connection.distance1
    d2 = connection.distance2
    denom = k1 * d2 + k2 * d1
    if denom > 0:
        k_harm = (d1 + d2) * k1 * k2 / denom
    else:
        k_harm = 0.0

    # Get phase properties from states
    if phase == PHASE_LIQUID:
        rho1 = state1.liquid_density
        rho2 = state2.liquid_density
        mu1 = state1.liquid_viscosity
        mu2 = state2.liquid_viscosity
        kr1 = state1.liquid_rel_perm
        kr2 = state2.liquid_rel_perm
        P1 = state1.liquid_pressure
        P2 = state2.liquid_pressure
        h1 = state1.liquid_enthalpy
        h2 = state2.liquid_enthalpy
    else:  # PHASE_GAS
        rho1 = state1.gas_density
        rho2 = state2.gas_density
        mu1 = state1.gas_viscosity
        mu2 = state2.gas_viscosity
        kr1 = state1.gas_rel_perm
        kr2 = state2.gas_rel_perm
        P1 = state1.gas_pressure
        P2 = state2.gas_pressure
        h1 = state1.gas_enthalpy
        h2 = state2.gas_enthalpy

    # Mobilities
    mob1 = compute_phase_mobility(rho1, mu1, kr1, k_harm)
    mob2 = compute_phase_mobility(rho2, mu2, kr2, k_harm)

    # Calculate flux
    total_dist = d1 + d2
    if total_dist <= 0 or (mob1 <= 0 and mob2 <= 0):
        return PhaseFlux(mass_flux=0.0, enthalpy_flux=0.0, upstream_cell=0)

    # Pressure gradient
    dp = P2 - P1

    # Gravity term
    rho_avg = 0.5 * (rho1 + rho2)
    grav_term = rho_avg * GRAVITY * connection.beta * total_dist

    # Potential difference (drives flow)
    potential_diff = dp - grav_term

    # Upstream weighting
    if potential_diff <= 0:
        # Flow from cell 1 to cell 2
        mobility = mob1
        upstream_rho = rho1
        upstream_h = h1
        upstream_cell = connection.cell1
    else:
        # Flow from cell 2 to cell 1
        mobility = mob2
        upstream_rho = rho2
        upstream_h = h2
        upstream_cell = connection.cell2

    # Transmissibility
    trans = connection.area * mobility / total_dist

    # Volume flux (positive = flow from cell1 to cell2)
    volume_flux = -trans * potential_diff

    # Mass flux = volume flux * upstream density
    mass_flux = volume_flux * upstream_rho

    # Enthalpy flux
    enthalpy_flux = mass_flux * upstream_h

    return PhaseFlux(
        mass_flux=mass_flux,
        enthalpy_flux=enthalpy_flux,
        upstream_cell=upstream_cell
    )


def compute_energy_flux(
    state1: 'State',
    state2: 'State',
    rock1: 'RockProperties',
    rock2: 'RockProperties',
    connection: 'Connection'
) -> float:
    """
    Calculate conductive heat flux across a connection.

    Q = -k * A * dT/dx

    Args:
        state1: State of cell 1
        state2: State of cell 2
        rock1: Rock properties of cell 1
        rock2: Rock properties of cell 2
        connection: Connection object

    Returns:
        Conductive heat flux [W]
    """
    # Harmonic average thermal conductivity
    k1 = rock1.thermal_conductivity
    k2 = rock2.thermal_conductivity
    d1 = connection.distance1
    d2 = connection.distance2

    denom = k1 * d2 + k2 * d1
    if denom > 0:
        k_harm = (d1 + d2) * k1 * k2 / denom
    else:
        return 0.0

    # Temperature gradient
    total_dist = d1 + d2
    if total_dist <= 0:
        return 0.0

    dT = state2.temperature - state1.temperature

    # Heat flux (positive = flow from cell1 to cell2)
    heat_flux = -k_harm * connection.area * dT / total_dist

    return heat_flux


def assemble_residual(
    mesh: 'Mesh',
    states: list,
    states_old: list,
    rock_properties: list,
    dt: float,
    source_terms: Optional[np.ndarray] = None,
    include_energy: bool = False
) -> np.ndarray:
    """
    Assemble the residual vector for all conservation equations.

    For isothermal water-air two-phase flow:
    - Equation 1: Water (liquid) mass balance
    - Equation 2: Air (gas) mass balance

    For water-steam thermal system:
    - Single-phase: use temperature constraint for second equation
    - Two-phase: use gas mass balance

    Args:
        mesh: Mesh object
        states: List of current State objects for each cell
        states_old: List of old (previous time step) State objects
        rock_properties: List of RockProperties for each cell
        dt: Time step [s]
        source_terms: Source/sink terms [kg/s] per cell per component
        include_energy: Whether to include energy equation

    Returns:
        Residual vector [num_cells * num_equations]
    """
    num_cells = mesh.num_cells
    num_eq = 3 if include_energy else 2  # Water mass, gas/air mass, (energy)
    residual = np.zeros(num_cells * num_eq)

    # Initialize source terms if not provided
    if source_terms is None:
        source_terms = np.zeros((num_cells, num_eq))

    # Import PhaseState and FluidSystem for checking conditions
    from ..core.state import PhaseState, FluidSystem

    # Determine fluid system from first cell (assuming all cells use same system)
    is_water_air = (len(states) > 0 and
                    states[0].fluid_system == FluidSystem.WATER_AIR)

    # Accumulation terms (for each cell)
    for i in range(num_cells):
        state = states[i]
        state_old = states_old[i]
        rock = rock_properties[i]
        vol = mesh.cells[i].volume
        phi = rock.porosity

        # Equation 1: Water (liquid) mass balance
        mass_w = phi * state.liquid_saturation * state.liquid_density
        mass_w_old = phi * state_old.liquid_saturation * state_old.liquid_density
        residual[i * num_eq + 0] = vol * (mass_w - mass_w_old) / dt

        # Equation 2: depends on fluid system
        if is_water_air:
            # Isothermal water-air: always use air (gas) mass balance
            mass_g = phi * state.gas_saturation * state.gas_density
            mass_g_old = phi * state_old.gas_saturation * state_old.gas_density
            residual[i * num_eq + 1] = vol * (mass_g - mass_g_old) / dt
        else:
            # Water-steam: depends on phase state
            if state.phase_state == PhaseState.SINGLE_LIQUID:
                # Temperature constraint for single-phase liquid
                residual[i * num_eq + 1] = (state.temperature - state_old.temperature) * 1e3
            elif state.phase_state == PhaseState.SINGLE_GAS:
                # Temperature constraint for single-phase gas
                residual[i * num_eq + 1] = (state.temperature - state_old.temperature) * 1e3
            else:
                # Two-phase: gas mass balance
                mass_g = phi * state.gas_saturation * state.gas_density
                mass_g_old = phi * state_old.gas_saturation * state_old.gas_density
                residual[i * num_eq + 1] = vol * (mass_g - mass_g_old) / dt

        if include_energy:
            # Total energy (fluid + rock)
            energy = (phi * (state.liquid_saturation * state.liquid_density *
                           state.liquid_internal_energy +
                           state.gas_saturation * state.gas_density *
                           state.gas_internal_energy) +
                     (1 - phi) * rock.density * rock.specific_heat * state.temperature)

            energy_old = (phi * (state_old.liquid_saturation * state_old.liquid_density *
                                state_old.liquid_internal_energy +
                                state_old.gas_saturation * state_old.gas_density *
                                state_old.gas_internal_energy) +
                         (1 - phi) * rock.density * rock.specific_heat * state_old.temperature)

            residual[i * num_eq + 2] = vol * (energy - energy_old) / dt

        # Add source terms (negative because convention is source = positive)
        for eq in range(num_eq):
            residual[i * num_eq + eq] -= source_terms[i, eq]

    # Flux terms (for each connection)
    for conn in mesh.connections:
        i1 = conn.cell1
        i2 = conn.cell2

        state1 = states[i1]
        state2 = states[i2]
        rock1 = rock_properties[i1]
        rock2 = rock_properties[i2]

        # Determine direction for permeability
        direction = 0  # x-direction default
        if abs(conn.direction[1]) > 0.5:
            direction = 1
        elif abs(conn.direction[2]) > 0.5:
            direction = 2

        # Liquid phase flux
        flux_l = compute_mass_flux(state1, state2, rock1, rock2,
                                   conn, PHASE_LIQUID, direction)

        # Gas phase flux
        flux_g = compute_mass_flux(state1, state2, rock1, rock2,
                                   conn, PHASE_GAS, direction)

        # Add flux contributions to residual
        # Positive flux = flow from cell1 to cell2
        # For cell1: flux leaves (positive contribution to residual)
        # For cell2: flux enters (negative contribution to residual)

        # Equation 1: Water mass (liquid phase) - always active
        residual[i1 * num_eq + 0] += flux_l.mass_flux
        residual[i2 * num_eq + 0] -= flux_l.mass_flux

        # Equation 2: Gas flux handling
        if is_water_air:
            # Water-air: always add gas flux to both cells
            residual[i1 * num_eq + 1] += flux_g.mass_flux
            residual[i2 * num_eq + 1] -= flux_g.mass_flux
        else:
            # Water-steam: only add gas flux for two-phase cells
            if state1.phase_state == PhaseState.TWO_PHASE:
                residual[i1 * num_eq + 1] += flux_g.mass_flux
            if state2.phase_state == PhaseState.TWO_PHASE:
                residual[i2 * num_eq + 1] -= flux_g.mass_flux

        if include_energy:
            # Advective enthalpy flux
            enthalpy_flux = flux_l.enthalpy_flux + flux_g.enthalpy_flux

            # Conductive heat flux
            heat_flux = compute_energy_flux(state1, state2, rock1, rock2, conn)

            total_energy_flux = enthalpy_flux + heat_flux

            residual[i1 * num_eq + 2] += total_energy_flux
            residual[i2 * num_eq + 2] -= total_energy_flux

    return residual


def compute_connection_flux(
    state1: 'State',
    state2: 'State',
    rock1: 'RockProperties',
    rock2: 'RockProperties',
    connection: 'Connection',
    direction: int = 0
) -> Tuple[float, float, float]:
    """
    Calculate total mass fluxes across a connection.

    Args:
        state1: State of cell 1
        state2: State of cell 2
        rock1: Rock properties of cell 1
        rock2: Rock properties of cell 2
        connection: Connection object
        direction: Direction for permeability

    Returns:
        Tuple of (liquid mass flux, gas mass flux, total mass flux) [kg/s]
    """
    flux_l = compute_mass_flux(state1, state2, rock1, rock2,
                               connection, PHASE_LIQUID, direction)
    flux_g = compute_mass_flux(state1, state2, rock1, rock2,
                               connection, PHASE_GAS, direction)

    return flux_l.mass_flux, flux_g.mass_flux, flux_l.mass_flux + flux_g.mass_flux
