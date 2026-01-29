"""
Physical constants for porous media flow simulation.

All units are in SI:
- Pressure: Pa
- Temperature: K (or °C where noted)
- Length: m
- Mass: kg
- Time: s
- Energy: J
"""

# Universal constants
R_GAS = 8.314462618  # Universal gas constant [J/(mol·K)]
GRAVITY = 9.80665  # Standard gravitational acceleration [m/s²]

# Water properties at reference conditions
WATER_MOLECULAR_WEIGHT = 18.01528e-3  # [kg/mol]
WATER_CRITICAL_TEMPERATURE = 647.096  # [K]
WATER_CRITICAL_PRESSURE = 22.064e6  # [Pa]
WATER_CRITICAL_DENSITY = 322.0  # [kg/m³]
WATER_TRIPLE_POINT_TEMPERATURE = 273.16  # [K]
WATER_TRIPLE_POINT_PRESSURE = 611.657  # [Pa]

# Reference state for water (IAPWS-IF97)
WATER_REFERENCE_TEMPERATURE = 273.15  # [K] (0°C)
WATER_REFERENCE_PRESSURE = 101325.0  # [Pa] (1 atm)
WATER_DENSITY_REF = 999.84  # [kg/m³] at 0°C, 1 atm

# Air properties
AIR_MOLECULAR_WEIGHT = 28.9647e-3  # [kg/mol]
AIR_DENSITY_STP = 1.225  # [kg/m³] at STP (15°C, 101325 Pa)

# Steam/vapor properties
STEAM_SPECIFIC_GAS_CONSTANT = R_GAS / WATER_MOLECULAR_WEIGHT  # [J/(kg·K)]

# Standard conditions
STANDARD_TEMPERATURE = 288.15  # [K] (15°C)
STANDARD_PRESSURE = 101325.0  # [Pa] (1 atm)

# Absolute zero
ABSOLUTE_ZERO = 0.0  # [K]
CELSIUS_OFFSET = 273.15  # K = °C + CELSIUS_OFFSET

# Default simulation parameters
DEFAULT_POROSITY = 0.1
DEFAULT_PERMEABILITY = 1.0e-15  # [m²] (about 1 millidarcy)
DEFAULT_ROCK_DENSITY = 2650.0  # [kg/m³]
DEFAULT_ROCK_SPECIFIC_HEAT = 1000.0  # [J/(kg·K)]
DEFAULT_ROCK_THERMAL_CONDUCTIVITY = 2.0  # [W/(m·K)]

# Numerical tolerances
NEWTON_TOLERANCE = 1.0e-4  # Relaxed for two-phase flow (was 1e-6)
MAX_NEWTON_ITERATIONS = 30
MIN_TIME_STEP = 1.0e-10  # [s]
MAX_TIME_STEP_INCREASE = 2.0
TIME_STEP_SAFETY_FACTOR = 0.9

# Small numbers to avoid division by zero
EPSILON = 1.0e-10
SATURATION_EPSILON = 1.0e-8

# Phase identifiers
PHASE_LIQUID = 0
PHASE_GAS = 1
PHASE_AQUEOUS = 0  # alias for liquid
PHASE_GASEOUS = 1  # alias for gas

# Component identifiers
COMPONENT_WATER = 0
COMPONENT_AIR = 1

# Primary variable identifiers
PVAR_PRESSURE = 0
PVAR_SATURATION = 1
PVAR_TEMPERATURE = 2
PVAR_MASS_FRACTION = 3

# Boundary condition types
BC_DIRICHLET = 0  # Fixed value
BC_NEUMANN = 1  # Fixed flux
BC_NO_FLOW = 2  # No flow (default for outer boundaries)

# Unit conversions
DARCY_TO_M2 = 9.869233e-13  # 1 Darcy = 9.869233e-13 m²
M2_TO_DARCY = 1.0 / DARCY_TO_M2
BAR_TO_PA = 1.0e5
PA_TO_BAR = 1.0e-5
MPA_TO_PA = 1.0e6
PA_TO_MPA = 1.0e-6
ATM_TO_PA = 101325.0
PA_TO_ATM = 1.0 / ATM_TO_PA
DAY_TO_S = 86400.0
S_TO_DAY = 1.0 / DAY_TO_S
YEAR_TO_S = 365.25 * DAY_TO_S
S_TO_YEAR = 1.0 / YEAR_TO_S

# Latent heat of vaporization at 100°C (approximate)
LATENT_HEAT_VAPORIZATION = 2.257e6  # [J/kg]

# Specific heats (approximate values)
WATER_SPECIFIC_HEAT = 4186.0  # [J/(kg·K)] at 25°C
STEAM_SPECIFIC_HEAT_CP = 2010.0  # [J/(kg·K)] at 100°C
AIR_SPECIFIC_HEAT_CP = 1005.0  # [J/(kg·K)] at STP
