"""
Capillary pressure models for multiphase flow.

Provides constitutive relationships between saturation and
capillary pressure including van Genuchten and Brooks-Corey models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

from ..utils.constants import SATURATION_EPSILON


class CapillaryPressure(ABC):
    """
    Abstract base class for capillary pressure models.

    Capillary pressure is defined as: Pc = Pg - Pl
    (gas pressure minus liquid pressure)

    All capillary pressure models should inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    def pressure(self, sl: float) -> float:
        """
        Calculate capillary pressure.

        Args:
            sl: Liquid saturation [-]

        Returns:
            Capillary pressure [Pa]
        """
        pass

    @abstractmethod
    def derivative(self, sl: float) -> float:
        """
        Calculate derivative of capillary pressure with respect to Sl.

        Args:
            sl: Liquid saturation [-]

        Returns:
            dPc/dSl [Pa]
        """
        pass

    def saturation(self, pc: float) -> float:
        """
        Calculate liquid saturation from capillary pressure (inverse).

        Default implementation uses Newton-Raphson iteration.

        Args:
            pc: Capillary pressure [Pa]

        Returns:
            Liquid saturation [-]
        """
        # Initial guess
        sl = 0.5

        for _ in range(20):
            pc_calc = self.pressure(sl)
            dpc_dsl = self.derivative(sl)

            if abs(dpc_dsl) < SATURATION_EPSILON:
                break

            dsl = (pc - pc_calc) / dpc_dsl
            sl += dsl

            sl = max(0.01, min(0.99, sl))

            if abs(dsl) < 1e-8:
                break

        return sl


class VanGenuchtenCapillary(CapillaryPressure):
    """
    Van Genuchten capillary pressure model.

    Pc = (1/alpha) * (Se^(-1/m) - 1)^(1-m)

    where Se = (Sl - Slr) / (Sls - Slr) is effective saturation.

    Attributes:
        slr: Residual liquid saturation [-]
        sls: Maximum liquid saturation [-]
        alpha: van Genuchten alpha parameter [1/Pa]
        m: van Genuchten m parameter [-]
        pmax: Maximum capillary pressure [Pa]
    """

    def __init__(self, slr: float = 0.1, sls: float = 1.0,
                 alpha: float = 1e-4, m: float = 0.45,
                 pmax: float = 1e7):
        """
        Initialize van Genuchten capillary pressure model.

        Args:
            slr: Residual liquid saturation
            sls: Maximum liquid saturation
            alpha: van Genuchten alpha [1/Pa]
            m: van Genuchten m parameter
            pmax: Maximum capillary pressure [Pa]
        """
        self.slr = slr
        self.sls = sls
        self.alpha = alpha
        self.m = m
        self.pmax = pmax
        self._sld = sls - slr

    @classmethod
    def from_params(cls, params: Dict[str, float]) -> 'VanGenuchtenCapillary':
        """Create instance from parameter dictionary."""
        return cls(
            slr=params.get('slr', 0.1),
            sls=params.get('sls', 1.0),
            alpha=params.get('alpha', 1e-4),
            m=params.get('m', 0.45),
            pmax=params.get('pmax', 1e7)
        )

    def _effective_saturation(self, sl: float) -> float:
        """Calculate effective saturation."""
        if self._sld <= SATURATION_EPSILON:
            return 0.5
        se = (sl - self.slr) / self._sld
        return max(SATURATION_EPSILON, min(1.0 - SATURATION_EPSILON, se))

    def pressure(self, sl: float) -> float:
        """Calculate capillary pressure."""
        se = self._effective_saturation(sl)

        if se >= 1.0 - SATURATION_EPSILON:
            return 0.0

        # Van Genuchten formula
        # Pc = (1/alpha) * (Se^(-1/m) - 1)^(1-m)
        se_inv_m = se ** (-1.0 / self.m)
        pc = (1.0 / self.alpha) * (se_inv_m - 1.0) ** (1.0 - self.m)

        return min(pc, self.pmax)

    def derivative(self, sl: float) -> float:
        """Calculate dPc/dSl."""
        se = self._effective_saturation(sl)

        if se <= SATURATION_EPSILON or se >= 1.0 - SATURATION_EPSILON:
            return 0.0
        if self._sld <= SATURATION_EPSILON:
            return 0.0

        # Derivative of van Genuchten formula
        # dPc/dSl = dPc/dSe * dSe/dSl
        # dSe/dSl = 1 / (Sls - Slr)

        se_inv_m = se ** (-1.0 / self.m)
        term = se_inv_m - 1.0

        if term <= SATURATION_EPSILON:
            return 0.0

        # dPc/dSe
        dpc_dse = (1.0 / self.alpha) * (1.0 - self.m) * (term ** (-self.m))
        dpc_dse *= (-1.0 / self.m) * se ** (-1.0 / self.m - 1.0)

        # dPc/dSl
        dpc_dsl = dpc_dse / self._sld

        return dpc_dsl

    def saturation(self, pc: float) -> float:
        """Calculate liquid saturation from capillary pressure (analytical)."""
        if pc <= 0:
            return self.sls
        if pc >= self.pmax:
            return self.slr

        # Inverse van Genuchten formula
        # Se = [1 + (alpha * Pc)^(1/(1-m))]^(-m)
        term = (self.alpha * pc) ** (1.0 / (1.0 - self.m))
        se = (1.0 + term) ** (-self.m)

        sl = se * self._sld + self.slr

        return max(self.slr, min(self.sls, sl))


class BrooksCoreyCapillary(CapillaryPressure):
    """
    Brooks-Corey capillary pressure model.

    Pc = Pe * Se^(-1/lambda)

    where Se = (Sl - Slr) / (Sls - Slr) is effective saturation.

    Attributes:
        slr: Residual liquid saturation [-]
        sls: Maximum liquid saturation [-]
        pe: Entry pressure [Pa]
        lambda_: Brooks-Corey lambda parameter [-]
        pmax: Maximum capillary pressure [Pa]
    """

    def __init__(self, slr: float = 0.1, sls: float = 1.0,
                 pe: float = 1e4, lambda_: float = 2.0,
                 pmax: float = 1e7):
        """
        Initialize Brooks-Corey capillary pressure model.

        Args:
            slr: Residual liquid saturation
            sls: Maximum liquid saturation
            pe: Entry pressure [Pa]
            lambda_: Brooks-Corey lambda parameter
            pmax: Maximum capillary pressure [Pa]
        """
        self.slr = slr
        self.sls = sls
        self.pe = pe
        self.lambda_ = lambda_
        self.pmax = pmax
        self._sld = sls - slr

    @classmethod
    def from_params(cls, params: Dict[str, float]) -> 'BrooksCoreyCapillary':
        """Create instance from parameter dictionary."""
        return cls(
            slr=params.get('slr', 0.1),
            sls=params.get('sls', 1.0),
            pe=params.get('pe', 1e4),
            lambda_=params.get('lambda', 2.0),
            pmax=params.get('pmax', 1e7)
        )

    def _effective_saturation(self, sl: float) -> float:
        """Calculate effective saturation."""
        if self._sld <= SATURATION_EPSILON:
            return 0.5
        se = (sl - self.slr) / self._sld
        return max(SATURATION_EPSILON, min(1.0 - SATURATION_EPSILON, se))

    def pressure(self, sl: float) -> float:
        """Calculate capillary pressure."""
        se = self._effective_saturation(sl)

        if se >= 1.0 - SATURATION_EPSILON:
            return self.pe

        # Brooks-Corey formula
        pc = self.pe * se ** (-1.0 / self.lambda_)

        return min(pc, self.pmax)

    def derivative(self, sl: float) -> float:
        """Calculate dPc/dSl."""
        se = self._effective_saturation(sl)

        if se <= SATURATION_EPSILON or se >= 1.0 - SATURATION_EPSILON:
            return 0.0
        if self._sld <= SATURATION_EPSILON:
            return 0.0

        # Derivative
        # dPc/dSe = -Pe/lambda * Se^(-1/lambda - 1)
        dpc_dse = -self.pe / self.lambda_ * se ** (-1.0 / self.lambda_ - 1.0)

        # dPc/dSl
        dpc_dsl = dpc_dse / self._sld

        return dpc_dsl

    def saturation(self, pc: float) -> float:
        """Calculate liquid saturation from capillary pressure (analytical)."""
        if pc <= self.pe:
            return self.sls
        if pc >= self.pmax:
            return self.slr

        # Inverse Brooks-Corey formula
        # Se = (Pc / Pe)^(-lambda)
        se = (pc / self.pe) ** (-self.lambda_)

        sl = se * self._sld + self.slr

        return max(self.slr, min(self.sls, sl))


class LinearCapillary(CapillaryPressure):
    """
    Simple linear capillary pressure model.

    Pc = Pmax * (1 - Se)

    Useful for simple cases and debugging.

    Attributes:
        slr: Residual liquid saturation [-]
        sls: Maximum liquid saturation [-]
        pmax: Maximum capillary pressure [Pa]
    """

    def __init__(self, slr: float = 0.0, sls: float = 1.0,
                 pmax: float = 1e5):
        """
        Initialize linear capillary pressure model.

        Args:
            slr: Residual liquid saturation
            sls: Maximum liquid saturation
            pmax: Maximum capillary pressure [Pa]
        """
        self.slr = slr
        self.sls = sls
        self.pmax = pmax
        self._sld = sls - slr

    @classmethod
    def from_params(cls, params: Dict[str, float]) -> 'LinearCapillary':
        """Create instance from parameter dictionary."""
        return cls(
            slr=params.get('slr', 0.0),
            sls=params.get('sls', 1.0),
            pmax=params.get('pmax', 1e5)
        )

    def _effective_saturation(self, sl: float) -> float:
        """Calculate effective saturation."""
        if self._sld <= SATURATION_EPSILON:
            return 0.5
        se = (sl - self.slr) / self._sld
        return max(0.0, min(1.0, se))

    def pressure(self, sl: float) -> float:
        """Calculate capillary pressure."""
        se = self._effective_saturation(sl)
        pc = self.pmax * (1.0 - se)
        return max(0.0, pc)

    def derivative(self, sl: float) -> float:
        """Calculate dPc/dSl."""
        if self._sld <= SATURATION_EPSILON:
            return 0.0
        return -self.pmax / self._sld

    def saturation(self, pc: float) -> float:
        """Calculate liquid saturation from capillary pressure."""
        if pc >= self.pmax:
            return self.slr
        if pc <= 0:
            return self.sls

        se = 1.0 - pc / self.pmax
        sl = se * self._sld + self.slr

        return max(self.slr, min(self.sls, sl))


class NoCapillary(CapillaryPressure):
    """
    No capillary pressure model (Pc = 0).

    Useful for cases where capillary effects are negligible.
    """

    def __init__(self):
        """Initialize no capillary pressure model."""
        pass

    @classmethod
    def from_params(cls, params: Dict[str, float]) -> 'NoCapillary':
        """Create instance from parameter dictionary."""
        return cls()

    def pressure(self, sl: float) -> float:
        """Capillary pressure is always zero."""
        return 0.0

    def derivative(self, sl: float) -> float:
        """Derivative is always zero."""
        return 0.0

    def saturation(self, pc: float) -> float:
        """Saturation is undefined for zero capillary pressure."""
        return 0.5


def create_capillary_pressure(model_name: str,
                              params: Dict[str, Any]) -> CapillaryPressure:
    """
    Factory function to create capillary pressure model.

    Args:
        model_name: Model name ('van_genuchten', 'brooks_corey', 'linear', 'none')
        params: Model parameters

    Returns:
        CapillaryPressure instance
    """
    models = {
        'van_genuchten': VanGenuchtenCapillary,
        'brooks_corey': BrooksCoreyCapillary,
        'linear': LinearCapillary,
        'none': NoCapillary,
    }

    model_name_lower = model_name.lower()
    if model_name_lower not in models:
        raise ValueError(f"Unknown capillary pressure model: {model_name}")

    return models[model_name_lower].from_params(params)
