from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from ambiance import Atmosphere as _IcaoAtmosphere

from deorbit.utils.constants import AIR_DENSITY_SEA_LEVEL, EARTH_RADIUS
from deorbit.data_models.atmos import AtmosKwargs, SimpleAtmosKwargs, IcaoKwargs, CoesaKwargs, CoesaFastKwargs


class AtmosphereModel(ABC):
    """Abstract base class for Atmosphere model implementations.
    Attributes:
        name: abstract; must be set as a class variable in any subclass

    Methods:
        density(state, time) -> float: abstract; must be implemented in any subclass
        kwargs() -> AtmosKwargs: returns a pydantic data model of model parameters
    """
    def __init__(self) -> None:
        super().__init__()
        self.kwargs: AtmosKwargs = None
        
    @property
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def density(self, state: np.ndarray, time: float) -> float:
        ...


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

    def __init__(self, **kwargs) -> None:
        """
        Args:
            earth_radius (float, optional): Earth's radius in metres. Defaults to EARTH_RADIUS.
            surf_density (float, optional): Air density at Earth's surface in kgm^-3. Defaults to AIR_DENSITY_SEA_LEVEL.
        """
        # Document the keywords used in the atmosphere model. This is required.
        self.kwargs: SimpleAtmosKwargs = SimpleAtmosKwargs(**kwargs)

    ## This function can be changed.
    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)

        return self.kwargs.surf_density * np.exp(
            (-(np.linalg.norm(state[:dim]) / self.kwargs.earth_radius) + 1)
        )


class IcaoAtmos(AtmosphereModel):
    name = "icao_standard_atmos"

    def __init__(self, **kwargs):
        self.kwargs: IcaoKwargs = IcaoKwargs(**kwargs)
        self._max_height = 81020
        self._density_at_max_height = _IcaoAtmosphere(self._max_height).density

    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)
        position = state[:dim]

        height = np.linalg.norm(position) - self.kwargs.earth_radius
        if height <= self._max_height:
            return _IcaoAtmosphere(height).density
        else:
            # TODO make better high altitude approx
            return self._density_at_max_height * np.exp(
                height - self._max_height
            )


class CoesaAtmos(AtmosphereModel):
    name = "coesa_atmos"

    def __init__(self, **kwargs):
        # Lazy import of coesa76
        if "_coesa76" not in dir():
            global _coesa76
            # Supress random stdout messages from this import
            with redirect_stdout(StringIO()):
                from pyatmos import coesa76 as _coesa76

        self.kwargs: CoesaKwargs = CoesaKwargs(**kwargs)

    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)
        position = state[:dim]

        height = np.linalg.norm(position) - self.kwargs.earth_radius
        height_in_km = height * 1e-3
        return _coesa76(height_in_km).rho


class CoesaAtmosFast(AtmosphereModel):
    """Uses a lookup table of atmosphere densities"""

    name = "coesa_atmos_fast"

    def __init__(self, **kwargs):
        self.kwargs: CoesaFastKwargs = CoesaFastKwargs(**kwargs)
        assert (
            self.kwargs.precision >= 0 and int(self.kwargs.precision) == self.kwargs.precision
        ), "Precision must be a non-negative integer"

        # Lazy import of coesa76
        if "_coesa76" not in dir():
            global _coesa76
            with redirect_stdout(StringIO()):
                from pyatmos import coesa76 as _coesa76

        start, end = -611, 1000000
        rounded_start = np.round(start, decimals=-self.kwargs.precision)
        self._start = (
            rounded_start
            if rounded_start >= start
            else rounded_start + 10**self.kwargs.precision
        )
        sample_heights = np.arange(
            self._start, end + 1, step=10**self.kwargs.precision, dtype=np.float64
        )
        sampled_densities = _coesa76(sample_heights * 1e-3).rho
        self._samples = dict(zip(sample_heights, sampled_densities))

    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)
        position = state[:dim]

        height = np.linalg.norm(position) - self.kwargs.earth_radius
        rounded_height = np.round(height, decimals=-self.kwargs.precision)
        rounded_height = (
            rounded_height
            if rounded_height >= self._start
            else rounded_height + 10**self.kwargs.precision
        )
        try:
            rho = self._samples[rounded_height]
        except KeyError:
            raise Exception(
                f"Height {height}m at time {time} is not supported by the COESA76-fast atmosphere model!"
            )
        return rho
