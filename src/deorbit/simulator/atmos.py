from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from io import StringIO
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere as _IcaoAtmosphere

from deorbit.data_models.atmos import (
    AtmosKwargs,
    CoesaFastKwargs,
    CoesaKwargs,
    IcaoKwargs,
    SimpleAtmosKwargs,
)
from deorbit.utils.constants import AIR_DENSITY_SEA_LEVEL, EARTH_RADIUS


class AtmosphereModel(ABC):
    """Abstract base class for Atmosphere model implementations.
    Attributes:
        kwargs (AtmosKwargs): A pydantic data model of model parameters

    Methods:
        density(state, time) -> float: abstract; must be implemented in any subclass
    """

    _models = {}

    def __init_subclass__(cls, model_name: str = None, **kwargs):
        # This special method is called when a _subclass_ is defined in the code.
        # This allows the `model_name` to be passed as an argument to the subclass instantiator
        if model_name is None:
            raise SyntaxError(
                "'model_name' must be supplied as an argument when defining a subclass of AtmosphereModel"
            )
        super().__init_subclass__(**kwargs)
        cls._models[model_name] = cls

    def __new__(cls, kwargs: AtmosKwargs):
        model_name = kwargs.atmos_name
        model_cls = cls._models[model_name]
        return super().__new__(model_cls)

    def __init__(self) -> None:
        super().__init__()
        self.kwargs: AtmosKwargs = None

    @abstractmethod
    def density(self, state: np.ndarray, time: float) -> float: ...

    def plot(
        self, height_bounds_meters: tuple[float, float], num_points: int = 100, ax: plt.Axes = None, label: str = None
    ) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        heights = np.linspace(*height_bounds_meters, num_points)
        state_samples = [(0, EARTH_RADIUS + h, 0, 0) for h in heights]
        densities = [self.density(s, 0) for s in state_samples]
        ax.plot(densities, heights, label=label)
        return fig, ax


class SimpleAtmos(AtmosphereModel, model_name="simple_atmos"):
    """Generate simple atmospheric model

    Attributes:
        earth_radius (float, optional): Earth's radius in metres. Defaults to EARTH_RADIUS.
        surf_density (float, optional): Air density at Earth's surface in kgm^-3. Defaults to AIR_DENSITY_SEA_LEVEL.

    Methods:
        density(state: np.ndarray, time: float) -> float: Density function taking state and time as input
        model_kwargs() -> dict: Returns model parameters
    """

    def __init__(self, kwargs: SimpleAtmosKwargs) -> None:
        """
        Args:
            earth_radius (float, optional): Earth's radius in metres. Defaults to EARTH_RADIUS.
            surf_density (float, optional): Air density at Earth's surface in kgm^-3. Defaults to AIR_DENSITY_SEA_LEVEL.
        """
        # Document the keywords used in the atmosphere model. This is required.
        self.kwargs: SimpleAtmosKwargs = kwargs

    ## This function can be changed.
    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)

        return self.kwargs.surf_density * np.exp(
            (-(np.linalg.norm(state[:dim]) / self.kwargs.earth_radius) + 1)
        )


class IcaoAtmos(AtmosphereModel, model_name="icao_standard_atmos"):
    def __init__(self, kwargs: IcaoKwargs):
        self.kwargs: IcaoKwargs = kwargs
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
            return self._density_at_max_height * np.exp(height - self._max_height)


class CoesaAtmos(AtmosphereModel, model_name="coesa_atmos"):
    def __init__(self, kwargs: CoesaKwargs):
        # Lazy import of coesa76
        if "_coesa76" not in dir():
            global _coesa76
            # Supress random stdout messages from this import
            with redirect_stdout(StringIO()):
                from pyatmos import coesa76 as _coesa76

        self.kwargs: CoesaKwargs = kwargs

    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)
        position = state[:dim]

        height = np.linalg.norm(position) - self.kwargs.earth_radius
        height_in_km = height * 1e-3
        return _coesa76(height_in_km).rho


class CoesaAtmosFast(AtmosphereModel, model_name="coesa_atmos_fast"):
    """Uses a lookup table of atmosphere densities"""

    def __init__(self, kwargs: CoesaFastKwargs):
        self.kwargs: CoesaFastKwargs = kwargs
        assert (
            self.kwargs.precision >= 0
            and int(self.kwargs.precision) == self.kwargs.precision
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
            self._start,
            end + 1,
            step=10**self.kwargs.precision,
            dtype=np.float64,
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


def raise_for_invalid_atmos_model(atmos_model: str) -> None:
    """Raises ValueError if the given simulation method name is not defined"""
    available_models = list(get_available_atmos_models().keys())
    if atmos_model not in available_models:
        raise ValueError(
            f"Atmosphere model {atmos_model} is not supported. Supported models are: {available_models}"
        )


def get_available_atmos_models() -> dict[str, type[AtmosphereModel]]:
    """Find available atmosphere models in atmos.py

    Returns:
        dict[str, type[AtmosphereModel]]: Dictionary of {model name: subclass of AtmosphereModel}
    """
    return AtmosphereModel._models


if __name__ == "__main__":
    breakpoint()
