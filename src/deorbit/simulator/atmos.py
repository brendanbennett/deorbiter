from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from io import StringIO
from typing import Callable

import numpy as np
from ambiance import Atmosphere as _IcaoAtmosphere

# Supress random stdout messages from this import
with redirect_stdout(StringIO()):
    from pyatmos import coesa76 as _coesa76

from deorbit.utils.constants import AIR_DENSITY_SEA_LEVEL, EARTH_RADIUS


class AtmosphereModel(ABC):
    """Abstract base class for Atmosphere model implementations.
    Attributes:
        name: abstract; must be set as a class variable in any subclass

    Methods:
        density(state, time) -> float: abstract; must be implemented in any subclass
        model_kwargs() -> dict: returns a dictionary of model parameters
    """

    @property
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def density(self, state: np.ndarray, time: float) -> float:
        ...

    def model_kwargs(self) -> dict:
        kwargs = self.__dict__
        # Filter out private variables
        # Private variables marked with '_' are not included in the model_kwargs dictionary.
        # There should be an equal number of keyword arguments to __init__() as there
        # are non-private instance variables
        return {key: kwargs[key] for key in kwargs if key[0] != "_"}


class SimpleAtmos(AtmosphereModel):
    """Generate simple atmospheric model

    Attributes:
        earth_radius (float, optional): Earth's radius in metres. Defaults to EARTH_RADIUS.
        surf_density (float, optional): Air density at Earth's surface in kgm^-3. Defaults to AIR_DENSITY_SEA_LEVEL.

    Methods:
        density(state: np.ndarray, time: float) -> float: Density function taking state and time as input
        model_kwargs() -> dict: Returns model parameters
    """

    name = "simple_atmos"

    def __init__(
        self,
        earth_radius: float = EARTH_RADIUS,
        surf_density: float = AIR_DENSITY_SEA_LEVEL,
    ) -> None:
        """
        Args:
            earth_radius (float, optional): Earth's radius in metres. Defaults to EARTH_RADIUS.
            surf_density (float, optional): Air density at Earth's surface in kgm^-3. Defaults to AIR_DENSITY_SEA_LEVEL.
        """
        # Document the keywords used in the atmosphere model. This is required.
        self.earth_radius = earth_radius
        self.surf_density = surf_density

    ## This function can be changed.
    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)

        return self.surf_density * np.exp(
            (-(np.linalg.norm(state[:dim]) / self.earth_radius) + 1)
        )


class IcaoAtmos(AtmosphereModel):
    name = "icao_standard_atmos"

    def __init__(self, earth_radius: float = EARTH_RADIUS):
        self.earth_radius = earth_radius
        self._max_height = 81020
        self._density_at_max_height = _IcaoAtmosphere(self._max_height).density

    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)
        position = state[:dim]

        height = np.linalg.norm(position) - self.earth_radius
        if height <= self._max_height:
            return _IcaoAtmosphere(height).density
        else:
            # TODO make better high altitude approx
            return self._density_at_max_height * np.exp(
                height - self._max_height
            )


class CoesaAtmos(AtmosphereModel):
    name = "coesa_atmos"

    def __init__(self, earth_radius: float = EARTH_RADIUS):
        self.earth_radius = earth_radius

    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)
        position = state[:dim]

        height = np.linalg.norm(position) - self.earth_radius
        height_in_km = height * 1e-3
        return _coesa76(height_in_km).rho


class CoesaAtmosFast(AtmosphereModel):
    """Uses a lookup table of atmosphere densities"""

    name = "coesa_atmos_fast"

    def __init__(self, earth_radius: float = EARTH_RADIUS, precision: int = 2):
        assert (
            precision >= 0 and int(precision) == precision
        ), "Precision must be a non-negative integer"
        self.earth_radius = earth_radius
        self.precision = precision

        start, end = -611, 1000000
        rounded_start = np.round(start, decimals=-precision)
        self._start = (
            rounded_start
            if rounded_start >= start
            else rounded_start + 10**precision
        )
        sample_heights = np.arange(
            self._start, end + 1, step=10**precision, dtype=np.float64
        )
        sampled_densities = _coesa76(sample_heights * 1e-3).rho
        self._samples = dict(zip(sample_heights, sampled_densities))

    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)
        position = state[:dim]

        height = np.linalg.norm(position) - self.earth_radius
        rounded_height = np.round(height, decimals=-self.precision)
        rounded_height = (
            rounded_height
            if rounded_height >= self._start
            else rounded_height + 10**self.precision
        )
        try:
            rho = self._samples[rounded_height]
        except KeyError:
            raise Exception(
                f"Height {height} is not supported by the COESA76-fast atmosphere model!"
            )
        return rho
