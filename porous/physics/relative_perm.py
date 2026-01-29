"""
Relative permeability models for multiphase flow.

Provides various constitutive relationships between saturation
and relative permeability including Corey and van Genuchten models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

from ..utils.constants import SATURATION_EPSILON


class RelativePermeability(ABC):
    """
    Abstract base class for relative permeability models.

    All relative permeability models should inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    def liquid(self, sl: float) -> float:
        """
        Calculate liquid relative permeability.

        Args:
            sl: Liquid saturation [-]

        Returns:
            Liquid relative permeability [-]
        """
        pass

    @abstractmethod
    def gas(self, sl: float) -> float:
        """
        Calculate gas relative permeability.

        Args:
            sl: Liquid saturation [-]

        Returns:
            Gas relative permeability [-]
        """
        pass

    def both(self, sl: float) -> Tuple[float, float]:
        """
        Calculate both liquid and gas relative permeabilities.

        Args:
            sl: Liquid saturation [-]

        Returns:
            Tuple of (krl, krg)
        """
        return self.liquid(sl), self.gas(sl)

    @abstractmethod
    def liquid_derivative(self, sl: float) -> float:
        """
        Calculate derivative of liquid relative permeability with respect to Sl.

        Args:
            sl: Liquid saturation [-]

        Returns:
            dkrl/dSl
        """
        pass

    @abstractmethod
    def gas_derivative(self, sl: float) -> float:
        """
        Calculate derivative of gas relative permeability with respect to Sl.

        Args:
            sl: Liquid saturation [-]

        Returns:
            dkrg/dSl
        """
        pass


class CoreyRelPerm(RelativePermeability):
    """
    Corey relative permeability model.

    krl = ((Sl - Slr) / (1 - Slr - Sgr))^nl
    krg = ((1 - Sl - Sgr) / (1 - Slr - Sgr))^ng

    Attributes:
        slr: Residual liquid saturation [-]
        sgr: Residual gas saturation [-]
        nl: Liquid Corey exponent [-]
        ng: Gas Corey exponent [-]
    """

    def __init__(self, slr: float = 0.1, sgr: float = 0.05,
                 nl: float = 2.0, ng: float = 2.0):
        """
        Initialize Corey model.

        Args:
            slr: Residual liquid saturation
            sgr: Residual gas saturation
            nl: Liquid Corey exponent
            ng: Gas Corey exponent
        """
        self.slr = slr
        self.sgr = sgr
        self.nl = nl
        self.ng = ng
        self._sld = 1.0 - slr - sgr  # Denominator for effective saturation

    @classmethod
    def from_params(cls, params: Dict[str, float]) -> 'CoreyRelPerm':
        """Create instance from parameter dictionary."""
        return cls(
            slr=params.get('slr', 0.1),
            sgr=params.get('sgr', 0.05),
            nl=params.get('nl', 2.0),
            ng=params.get('ng', 2.0)
        )

    def _effective_liquid_saturation(self, sl: float) -> float:
        """Calculate effective liquid saturation."""
        if self._sld <= SATURATION_EPSILON:
            return 0.0
        se = (sl - self.slr) / self._sld
        return max(0.0, min(1.0, se))

    def _effective_gas_saturation(self, sl: float) -> float:
        """Calculate effective gas saturation."""
        if self._sld <= SATURATION_EPSILON:
            return 0.0
        se = (1.0 - sl - self.sgr) / self._sld
        return max(0.0, min(1.0, se))

    def liquid(self, sl: float) -> float:
        """Calculate liquid relative permeability."""
        se = self._effective_liquid_saturation(sl)
        if se <= 0:
            return 0.0
        if se >= 1:
            return 1.0
        return se ** self.nl

    def gas(self, sl: float) -> float:
        """Calculate gas relative permeability."""
        se = self._effective_gas_saturation(sl)
        if se <= 0:
            return 0.0
        if se >= 1:
            return 1.0
        return se ** self.ng

    def liquid_derivative(self, sl: float) -> float:
        """Calculate dkrl/dSl."""
        se = self._effective_liquid_saturation(sl)
        if se <= SATURATION_EPSILON or se >= 1.0 - SATURATION_EPSILON:
            return 0.0
        if self._sld <= SATURATION_EPSILON:
            return 0.0
        return self.nl * (se ** (self.nl - 1)) / self._sld

    def gas_derivative(self, sl: float) -> float:
        """Calculate dkrg/dSl."""
        se = self._effective_gas_saturation(sl)
        if se <= SATURATION_EPSILON or se >= 1.0 - SATURATION_EPSILON:
            return 0.0
        if self._sld <= SATURATION_EPSILON:
            return 0.0
        # Note: dSeg/dSl = -1/sld
        return -self.ng * (se ** (self.ng - 1)) / self._sld


class VanGenuchtenRelPerm(RelativePermeability):
    """
    Van Genuchten - Mualem relative permeability model.

    krl = sqrt(Se) * [1 - (1 - Se^(1/m))^m]^2
    krg = sqrt(1-Se) * (1 - Se^(1/m))^(2m)

    where Se = (Sl - Slr) / (Sls - Slr) is effective saturation.

    Attributes:
        slr: Residual liquid saturation [-]
        sls: Maximum liquid saturation [-]
        m: van Genuchten parameter [-]
    """

    def __init__(self, slr: float = 0.1, sls: float = 1.0, m: float = 0.45):
        """
        Initialize van Genuchten model.

        Args:
            slr: Residual liquid saturation
            sls: Maximum liquid saturation
            m: van Genuchten m parameter
        """
        self.slr = slr
        self.sls = sls
        self.m = m
        self._sld = sls - slr

    @classmethod
    def from_params(cls, params: Dict[str, float]) -> 'VanGenuchtenRelPerm':
        """Create instance from parameter dictionary."""
        return cls(
            slr=params.get('slr', 0.1),
            sls=params.get('sls', 1.0),
            m=params.get('m', 0.45)
        )

    def _effective_saturation(self, sl: float) -> float:
        """Calculate effective saturation."""
        if self._sld <= SATURATION_EPSILON:
            return 0.0
        se = (sl - self.slr) / self._sld
        return max(0.0, min(1.0, se))

    def liquid(self, sl: float) -> float:
        """Calculate liquid relative permeability."""
        se = self._effective_saturation(sl)
        if se <= SATURATION_EPSILON:
            return 0.0
        if se >= 1.0 - SATURATION_EPSILON:
            return 1.0

        # Van Genuchten - Mualem formula
        se_inv_m = se ** (1.0 / self.m)
        term = 1.0 - (1.0 - se_inv_m) ** self.m
        krl = np.sqrt(se) * term ** 2

        return max(0.0, min(1.0, krl))

    def gas(self, sl: float) -> float:
        """Calculate gas relative permeability."""
        se = self._effective_saturation(sl)
        if se >= 1.0 - SATURATION_EPSILON:
            return 0.0
        if se <= SATURATION_EPSILON:
            return 1.0

        # Van Genuchten formula for gas
        se_inv_m = se ** (1.0 / self.m)
        term = (1.0 - se_inv_m) ** self.m
        krg = np.sqrt(1.0 - se) * term ** 2

        return max(0.0, min(1.0, krg))

    def liquid_derivative(self, sl: float) -> float:
        """Calculate dkrl/dSl using numerical differentiation."""
        se = self._effective_saturation(sl)
        if se <= SATURATION_EPSILON or se >= 1.0 - SATURATION_EPSILON:
            return 0.0
        if self._sld <= SATURATION_EPSILON:
            return 0.0

        # Use finite difference
        dsl = 1e-8
        krl_plus = self.liquid(sl + dsl)
        krl_minus = self.liquid(sl - dsl)
        return (krl_plus - krl_minus) / (2 * dsl)

    def gas_derivative(self, sl: float) -> float:
        """Calculate dkrg/dSl using numerical differentiation."""
        se = self._effective_saturation(sl)
        if se <= SATURATION_EPSILON or se >= 1.0 - SATURATION_EPSILON:
            return 0.0
        if self._sld <= SATURATION_EPSILON:
            return 0.0

        # Use finite difference
        dsl = 1e-8
        krg_plus = self.gas(sl + dsl)
        krg_minus = self.gas(sl - dsl)
        return (krg_plus - krg_minus) / (2 * dsl)


class LinearRelPerm(RelativePermeability):
    """
    Simple linear relative permeability model.

    krl = (Sl - Slr) / (1 - Slr - Sgr)
    krg = (1 - Sl - Sgr) / (1 - Slr - Sgr)

    Useful for simple cases and debugging.
    """

    def __init__(self, slr: float = 0.0, sgr: float = 0.0):
        """
        Initialize linear model.

        Args:
            slr: Residual liquid saturation
            sgr: Residual gas saturation
        """
        self.slr = slr
        self.sgr = sgr
        self._sld = 1.0 - slr - sgr

    @classmethod
    def from_params(cls, params: Dict[str, float]) -> 'LinearRelPerm':
        """Create instance from parameter dictionary."""
        return cls(
            slr=params.get('slr', 0.0),
            sgr=params.get('sgr', 0.0)
        )

    def liquid(self, sl: float) -> float:
        """Calculate liquid relative permeability."""
        if self._sld <= SATURATION_EPSILON:
            return 0.0
        krl = (sl - self.slr) / self._sld
        return max(0.0, min(1.0, krl))

    def gas(self, sl: float) -> float:
        """Calculate gas relative permeability."""
        if self._sld <= SATURATION_EPSILON:
            return 0.0
        krg = (1.0 - sl - self.sgr) / self._sld
        return max(0.0, min(1.0, krg))

    def liquid_derivative(self, sl: float) -> float:
        """Calculate dkrl/dSl."""
        if self._sld <= SATURATION_EPSILON:
            return 0.0
        if sl <= self.slr or sl >= 1.0 - self.sgr:
            return 0.0
        return 1.0 / self._sld

    def gas_derivative(self, sl: float) -> float:
        """Calculate dkrg/dSl."""
        if self._sld <= SATURATION_EPSILON:
            return 0.0
        if sl <= self.slr or sl >= 1.0 - self.sgr:
            return 0.0
        return -1.0 / self._sld


def create_relative_permeability(model_name: str,
                                 params: Dict[str, Any]) -> RelativePermeability:
    """
    Factory function to create relative permeability model.

    Args:
        model_name: Model name ('corey', 'van_genuchten', or 'linear')
        params: Model parameters

    Returns:
        RelativePermeability instance
    """
    models = {
        'corey': CoreyRelPerm,
        'van_genuchten': VanGenuchtenRelPerm,
        'linear': LinearRelPerm,
    }

    model_name_lower = model_name.lower()
    if model_name_lower not in models:
        raise ValueError(f"Unknown relative permeability model: {model_name}")

    return models[model_name_lower].from_params(params)
