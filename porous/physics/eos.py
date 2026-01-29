"""
Equations of State (EOS) for water, steam, and air.

Provides thermodynamic property calculations including density,
enthalpy, viscosity, and phase equilibrium. Uses simplified
correlations approximating IAPWS-IF97.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from ..utils.constants import (
    R_GAS, WATER_MOLECULAR_WEIGHT, AIR_MOLECULAR_WEIGHT,
    WATER_CRITICAL_TEMPERATURE, WATER_CRITICAL_PRESSURE,
    CELSIUS_OFFSET, EPSILON
)


@dataclass
class FluidState:
    """
    Container for fluid thermodynamic state.

    Attributes:
        density: Density [kg/m³]
        enthalpy: Specific enthalpy [J/kg]
        viscosity: Dynamic viscosity [Pa·s]
        internal_energy: Specific internal energy [J/kg]
    """
    density: float
    enthalpy: float
    viscosity: float
    internal_energy: float = 0.0


class WaterEOS:
    """
    Equation of state for liquid water.

    Uses polynomial correlations fitted to IAPWS-IF97 data
    for the compressed liquid region.
    """

    @staticmethod
    def density(pressure: float, temperature: float) -> float:
        """
        Calculate liquid water density.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            Density [kg/m³]
        """
        # Reference values at 20°C, 1 bar
        rho_ref = 998.2
        T_ref = 293.15  # K
        P_ref = 1e5  # Pa

        # Compressibility and thermal expansion
        beta_T = 4.5e-10  # 1/Pa
        alpha_T = 2.1e-4  # 1/K

        # Temperature correction (water density anomaly near 4°C)
        T_celsius = temperature - CELSIUS_OFFSET
        if T_celsius < 4.0:
            # Simplified correction for cold water
            temp_factor = 1.0 - alpha_T * (temperature - T_ref) * 0.5
        else:
            temp_factor = 1.0 - alpha_T * (temperature - T_ref)

        # Pressure correction
        pres_factor = 1.0 + beta_T * (pressure - P_ref)

        density = rho_ref * temp_factor * pres_factor

        return max(density, 100.0)  # Minimum bound

    @staticmethod
    def enthalpy(pressure: float, temperature: float) -> float:
        """
        Calculate liquid water specific enthalpy.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            Specific enthalpy [J/kg]
        """
        # Reference: h = 0 at triple point (273.16 K)
        T_ref = 273.16
        cp = 4186.0  # J/(kg·K), approximately constant

        # Temperature contribution
        h = cp * (temperature - T_ref)

        # Pressure contribution (small for liquid)
        rho = WaterEOS.density(pressure, temperature)
        h += (pressure - 611.657) / rho  # From triple point pressure

        return h

    @staticmethod
    def viscosity(temperature: float) -> float:
        """
        Calculate liquid water dynamic viscosity.

        Uses Vogel equation correlation.

        Args:
            temperature: Temperature [K]

        Returns:
            Dynamic viscosity [Pa·s]
        """
        T_celsius = temperature - CELSIUS_OFFSET

        if T_celsius < 0:
            T_celsius = 0.0
        elif T_celsius > 350:
            T_celsius = 350.0

        # Vogel equation parameters
        A = 2.414e-5
        B = 247.8
        C = 140.0

        mu = A * 10**(B / (T_celsius + C))

        return mu

    @staticmethod
    def internal_energy(pressure: float, temperature: float) -> float:
        """
        Calculate liquid water specific internal energy.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            Specific internal energy [J/kg]
        """
        h = WaterEOS.enthalpy(pressure, temperature)
        rho = WaterEOS.density(pressure, temperature)
        u = h - pressure / rho
        return u

    @staticmethod
    def get_state(pressure: float, temperature: float) -> FluidState:
        """
        Get complete thermodynamic state for liquid water.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            FluidState object
        """
        rho = WaterEOS.density(pressure, temperature)
        h = WaterEOS.enthalpy(pressure, temperature)
        mu = WaterEOS.viscosity(temperature)
        u = h - pressure / rho

        return FluidState(
            density=rho,
            enthalpy=h,
            viscosity=mu,
            internal_energy=u
        )


class SteamEOS:
    """
    Equation of state for water vapor (steam).

    Uses ideal gas law with compressibility corrections
    for the superheated and saturated vapor regions.
    """

    @staticmethod
    def density(pressure: float, temperature: float) -> float:
        """
        Calculate steam density.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            Density [kg/m³]
        """
        # Specific gas constant for water
        R_steam = R_GAS / WATER_MOLECULAR_WEIGHT

        # Compressibility factor (simplified correlation)
        T_r = temperature / WATER_CRITICAL_TEMPERATURE
        P_r = pressure / WATER_CRITICAL_PRESSURE

        # Simple compressibility correlation
        Z = 1.0 - 0.5 * P_r / T_r

        if Z < 0.1:
            Z = 0.1

        # Ideal gas with compressibility correction
        rho = pressure / (Z * R_steam * temperature)

        return max(rho, EPSILON)

    @staticmethod
    def enthalpy(pressure: float, temperature: float) -> float:
        """
        Calculate steam specific enthalpy.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            Specific enthalpy [J/kg]
        """
        # Reference: saturated vapor enthalpy at 100°C
        T_ref = 373.15  # K
        h_ref = 2676.0e3  # J/kg (at 1 atm)

        # Specific heat at constant pressure (function of T)
        # cp ≈ 1850 + 0.6*(T-373) for steam
        T_celsius = temperature - CELSIUS_OFFSET
        cp = 1850.0 + 0.6 * (T_celsius - 100.0)
        cp = max(cp, 1500.0)  # Minimum bound

        # Enthalpy from reference state
        h = h_ref + cp * (temperature - T_ref)

        return h

    @staticmethod
    def viscosity(temperature: float) -> float:
        """
        Calculate steam dynamic viscosity.

        Uses Sutherland's law.

        Args:
            temperature: Temperature [K]

        Returns:
            Dynamic viscosity [Pa·s]
        """
        # Sutherland parameters for steam
        mu_ref = 1.12e-5  # Pa·s at T_ref
        T_ref = 373.15  # K
        S = 961.0  # Sutherland constant

        mu = mu_ref * (temperature / T_ref)**1.5 * (T_ref + S) / (temperature + S)

        return mu

    @staticmethod
    def internal_energy(pressure: float, temperature: float) -> float:
        """
        Calculate steam specific internal energy.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            Specific internal energy [J/kg]
        """
        h = SteamEOS.enthalpy(pressure, temperature)
        rho = SteamEOS.density(pressure, temperature)
        u = h - pressure / rho
        return u

    @staticmethod
    def get_state(pressure: float, temperature: float) -> FluidState:
        """
        Get complete thermodynamic state for steam.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            FluidState object
        """
        rho = SteamEOS.density(pressure, temperature)
        h = SteamEOS.enthalpy(pressure, temperature)
        mu = SteamEOS.viscosity(temperature)
        u = h - pressure / rho

        return FluidState(
            density=rho,
            enthalpy=h,
            viscosity=mu,
            internal_energy=u
        )


class AirEOS:
    """
    Equation of state for air (treated as ideal gas).
    """

    @staticmethod
    def density(pressure: float, temperature: float) -> float:
        """
        Calculate air density.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            Density [kg/m³]
        """
        R_air = R_GAS / AIR_MOLECULAR_WEIGHT
        rho = pressure / (R_air * temperature)
        return max(rho, EPSILON)

    @staticmethod
    def enthalpy(temperature: float) -> float:
        """
        Calculate air specific enthalpy.

        Args:
            temperature: Temperature [K]

        Returns:
            Specific enthalpy [J/kg]
        """
        # Reference: h = 0 at 0°C
        T_ref = 273.15
        cp = 1005.0  # J/(kg·K)
        h = cp * (temperature - T_ref)
        return h

    @staticmethod
    def viscosity(temperature: float) -> float:
        """
        Calculate air dynamic viscosity.

        Uses Sutherland's law.

        Args:
            temperature: Temperature [K]

        Returns:
            Dynamic viscosity [Pa·s]
        """
        # Sutherland parameters for air
        mu_ref = 1.716e-5  # Pa·s at T_ref
        T_ref = 273.15  # K
        S = 110.4  # Sutherland constant

        mu = mu_ref * (temperature / T_ref)**1.5 * (T_ref + S) / (temperature + S)

        return mu

    @staticmethod
    def get_state(pressure: float, temperature: float) -> FluidState:
        """
        Get complete thermodynamic state for air.

        Args:
            pressure: Pressure [Pa]
            temperature: Temperature [K]

        Returns:
            FluidState object
        """
        rho = AirEOS.density(pressure, temperature)
        h = AirEOS.enthalpy(temperature)
        mu = AirEOS.viscosity(temperature)
        u = h - pressure / rho

        return FluidState(
            density=rho,
            enthalpy=h,
            viscosity=mu,
            internal_energy=u
        )


def saturation_pressure(temperature: float) -> float:
    """
    Calculate water saturation (vapor) pressure.

    Uses Antoine equation approximation.

    Args:
        temperature: Temperature [K]

    Returns:
        Saturation pressure [Pa]
    """
    T_celsius = temperature - CELSIUS_OFFSET

    # Bounds check
    if T_celsius < 0.01:
        T_celsius = 0.01
    elif T_celsius > 373.946:  # Critical point
        return WATER_CRITICAL_PRESSURE

    # Antoine equation coefficients (for water, T in °C, P in mmHg)
    # Valid for 1-100°C: A=8.07131, B=1730.63, C=233.426
    # Valid for 100-374°C: A=8.14019, B=1810.94, C=244.485

    if T_celsius <= 100.0:
        A = 8.07131
        B = 1730.63
        C = 233.426
    else:
        A = 8.14019
        B = 1810.94
        C = 244.485

    # Pressure in mmHg
    log_P_mmHg = A - B / (C + T_celsius)
    P_mmHg = 10**log_P_mmHg

    # Convert to Pa (1 mmHg = 133.322 Pa)
    P_sat = P_mmHg * 133.322

    return P_sat


def saturation_temperature(pressure: float) -> float:
    """
    Calculate water saturation (boiling) temperature.

    Inverse of saturation_pressure using Newton-Raphson.

    Args:
        pressure: Pressure [Pa]

    Returns:
        Saturation temperature [K]
    """
    if pressure <= 611.657:  # Triple point
        return 273.16
    if pressure >= WATER_CRITICAL_PRESSURE:
        return WATER_CRITICAL_TEMPERATURE

    # Initial guess using simplified correlation
    T = 373.15 * (pressure / 101325.0)**0.25

    # Newton-Raphson iteration
    for _ in range(20):
        P_sat = saturation_pressure(T)
        dP_dT = (saturation_pressure(T + 0.1) - P_sat) / 0.1

        if abs(dP_dT) < EPSILON:
            break

        dT = (pressure - P_sat) / dP_dT
        T += dT

        if abs(dT) < 0.001:
            break

    # Bounds
    T = max(T, 273.16)
    T = min(T, WATER_CRITICAL_TEMPERATURE)

    return T


def latent_heat(temperature: float) -> float:
    """
    Calculate latent heat of vaporization for water.

    Args:
        temperature: Temperature [K]

    Returns:
        Latent heat [J/kg]
    """
    T_celsius = temperature - CELSIUS_OFFSET

    # Watson correlation
    T_c = WATER_CRITICAL_TEMPERATURE - CELSIUS_OFFSET  # °C
    T_ref = 100.0  # °C
    L_ref = 2257.0e3  # J/kg at 100°C

    if T_celsius >= T_c:
        return 0.0

    L = L_ref * ((T_c - T_celsius) / (T_c - T_ref))**0.38

    return max(L, 0.0)


def water_vapor_mass_fraction_in_gas(
    pressure: float,
    temperature: float,
    gas_pressure: float
) -> float:
    """
    Calculate mass fraction of water vapor in gas phase.

    Assumes ideal gas mixture.

    Args:
        pressure: Total pressure [Pa]
        temperature: Temperature [K]
        gas_pressure: Partial pressure of non-condensable gas [Pa]

    Returns:
        Mass fraction of water vapor [-]
    """
    P_sat = saturation_pressure(temperature)
    P_vapor = min(P_sat, pressure - gas_pressure)

    if P_vapor <= 0:
        return 0.0

    # Mole fractions
    y_vapor = P_vapor / pressure
    y_gas = 1.0 - y_vapor

    # Mass fractions
    M_vapor = WATER_MOLECULAR_WEIGHT
    M_gas = AIR_MOLECULAR_WEIGHT

    denom = y_vapor * M_vapor + y_gas * M_gas
    if denom < EPSILON:
        return 0.0

    x_vapor = y_vapor * M_vapor / denom

    return x_vapor


def gas_mixture_density(
    pressure: float,
    temperature: float,
    water_mass_fraction: float
) -> float:
    """
    Calculate density of gas mixture (steam + air).

    Args:
        pressure: Total pressure [Pa]
        temperature: Temperature [K]
        water_mass_fraction: Mass fraction of water vapor [-]

    Returns:
        Mixture density [kg/m³]
    """
    # Individual densities
    rho_steam = SteamEOS.density(pressure, temperature)
    rho_air = AirEOS.density(pressure, temperature)

    # Mixture density (using volume fractions)
    x_w = water_mass_fraction
    x_a = 1.0 - x_w

    if x_w < EPSILON:
        return rho_air
    if x_a < EPSILON:
        return rho_steam

    # Volume fractions from mass fractions
    v_w = x_w / rho_steam
    v_a = x_a / rho_air
    v_total = v_w + v_a

    rho_mix = 1.0 / v_total

    return rho_mix


def gas_mixture_viscosity(
    temperature: float,
    water_mass_fraction: float
) -> float:
    """
    Calculate viscosity of gas mixture (steam + air).

    Uses simple mass-weighted mixing rule.

    Args:
        temperature: Temperature [K]
        water_mass_fraction: Mass fraction of water vapor [-]

    Returns:
        Mixture viscosity [Pa·s]
    """
    mu_steam = SteamEOS.viscosity(temperature)
    mu_air = AirEOS.viscosity(temperature)

    x_w = water_mass_fraction
    x_a = 1.0 - x_w

    # Simple mixing (Wilke's equation could be used for more accuracy)
    mu_mix = x_w * mu_steam + x_a * mu_air

    return mu_mix
