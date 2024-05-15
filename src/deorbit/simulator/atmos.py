from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from io import StringIO
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from ambiance import Atmosphere as _IcaoAtmosphere

from deorbit.data_models.atmos import (
    AtmosKwargs,
    CoesaFastKwargs,
    CoesaKwargs,
    IcaoKwargs,
    SimpleAtmosKwargs,
    ZeroAtmosKwargs,
)
from deorbit.utils.constants import (
    AIR_DENSITY_SEA_LEVEL,
    EARTH_RADIUS,
    EARTH_ROTATIONAL_SPEED,
)


class AtmosphereModel(ABC):
    """Abstract base class for Atmosphere model implementations.
    
    .. note:: The canonical way of initializing an AtmosphereModel is by using the function :func:`deorbit.data_models.get_model_for_atmos` to obtain the correct kwargs model for the desired atmosphere model, and then passing the kwargs model to the AtmosphereModel constructor.
        For example:
        
        .. code-block:: python
        
            from from deorbit.simulator.atmos import AtmosphereModel
            from deorbit.data_models.atmos import get_model_for_atmos
            atmos = AtmosphereModel(get_model_for_atmos("coesa_atmos_fast"))
    
    :param kwargs: The parameter values for the atmosphere model.
    :type kwargs: AtmosKwargs
    """

    _models = {}
    _rot_2d_ccw = np.array([[0, -1], [1, 0]])

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
    def density(self, state: np.ndarray, time: float) -> float:
        """Calculate the density of the atmosphere at a given state (and time)
        
        :param state: The state array of the object in the atmosphere
        :param time: The time at which the density is calculated. This is currently not used in any model.
        :return: The density of the atmosphere at the given state and time
        """
        ...

    def velocity(self, state: np.ndarray, time: float) -> np.ndarray:
        """Calculate the velocity of the atmosphere as a result of the Earth's rotation at a given state (and time)

        :param state: The state array of the object in the atmosphere
        :param time: The time at which the velocity is calculated. This is currently not used in any model.
        :return: The velocity of the atmosphere at the given state and time
        """
        dim = int(len(state) / 2)
        position = state[:dim]
        if dim == 2:
            pos_norm = np.linalg.norm(position)
            rot_radius = pos_norm
            vel_direction = AtmosphereModel._rot_2d_ccw @ position / pos_norm
        if dim == 3:
            rot_radius = np.sqrt(np.sum(position**2) + state[2] ** 2)
            vel_direction = np.array(
                [
                    *(
                        AtmosphereModel._rot_2d_ccw
                        @ position[:2]
                        / (np.linalg.norm(position[:2]))
                    ),
                    0,
                ]
            )
        speed = EARTH_ROTATIONAL_SPEED * rot_radius
        return speed * vel_direction

    @abstractmethod
    def derivative(self, state: np.ndarray, time: float) -> float:
        """Calculate the derivative of the density of the atmosphere at a given state (and time) with respect to height
        
        :param state: The state array of the object in the atmosphere
        :param time: The time at which the derivative is calculated. This is currently not used in any model.
        :return: The derivative of the density of the atmosphere at the given state and time with respect to height
        """
        ...

    def plot(
        self,
        height_bounds_meters: tuple[float, float],
        num_points: int = 100,
        ax: plt.Axes = None,
        label: str = None,
        derivative: bool = False,
    ) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        heights = np.linspace(*height_bounds_meters, num_points)
        state_samples = [(0, EARTH_RADIUS + h, 0, 0) for h in heights]
        densities = [self.density(s, 0) for s in state_samples]
        ax.plot(densities, heights, label=label)
        if derivative:
            try:
                derivatives = [self.derivative(s, 0) for s in state_samples]
                ax.plot(derivatives, heights, label=label + " derivative")
            except NotImplementedError:
                pass
        return fig, ax


class ZeroAtmos(AtmosphereModel, model_name="zero_atmos"):
    """Identically zero atmosphere model"""

    def __init__(self, kwargs: ZeroAtmosKwargs) -> None:
        self.kwargs: ZeroAtmosKwargs = kwargs

    def derivative(self, state: np.ndarray, time: float) -> float:
        """:meta private:"""
        raise NotImplementedError(
            f"Derivative not implemented for {self.__class__.__name__}"
        )

    def density(self, state: np.ndarray, time: float) -> float:
        """Identically zero density function
        
        :param state: Unused
        :param time: Unused
        :return: 0
        """
        return 0


class SimpleAtmos(AtmosphereModel, model_name="simple_atmos"):
    """Generate simple atmospheric model
    """

    def __init__(self, kwargs: SimpleAtmosKwargs) -> None:
        # Document the keywords used in the atmosphere model. This is required.
        self.kwargs: SimpleAtmosKwargs = kwargs

    def derivative(self, state: np.ndarray, time: float) -> float:
        """:meta private:"""
        raise NotImplementedError(
            f"Derivative not implemented for {self.__class__.__name__}"
        )
        
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
        
    def derivative(self, state: np.ndarray, time: float) -> float:
        """:meta private:"""
        raise NotImplementedError(
            f"Derivative not implemented for {self.__class__.__name__}"
        )

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
    """Uses the COESA76 model to calculate atmospheric density. The most accurate model. This uses lazy importing of the coesa76 model from :mod:`pyatmos`.
    """
    def __init__(self, kwargs: CoesaKwargs):
        # Lazy import of coesa76
        if "_coesa76" not in dir():
            global _coesa76
            # Supress random stdout messages from this import
            with redirect_stdout(StringIO()):
                from pyatmos import coesa76 as _coesa76

        self.kwargs: CoesaKwargs = kwargs
    
    def derivative(self, state: np.ndarray, time: float) -> float:
        """:meta private:"""
        raise NotImplementedError(
            f"Derivative not implemented for {self.__class__.__name__}"
        )

    def density(self, state: np.ndarray, time: float) -> float:
        dim = int(len(state) / 2)
        position = state[:dim]

        height = np.linalg.norm(position) - self.kwargs.earth_radius
        height_in_km = height * 1e-3
        return _coesa76(height_in_km).rho


class CoesaAtmosFast(AtmosphereModel, model_name="coesa_atmos_fast"):
    """Uses a lookup table of COESA76 atmosphere densities to calculate quicker. Marginally less accurate than CoesaAtmos."""

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

        sampled_derivatives = np.gradient(sampled_densities, 10**self.kwargs.precision)
        self._derivatives = dict(zip(sample_heights, sampled_derivatives))

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
        if rounded_height > 1000000:
            return 0.0
        try:
            rho = self._samples[rounded_height]
        except KeyError:
            raise Exception(
                f"Height {height}m at time {time} is not supported by the COESA76-fast atmosphere model!"
            )
        return rho

    def derivative(self, state: np.ndarray, time: float) -> float:
        # TODO: Fix this: has a bump
        dim = int(len(state) / 2)
        position = state[:dim]

        height = np.linalg.norm(position) - self.kwargs.earth_radius
        rounded_height = np.round(height, decimals=-self.kwargs.precision)
        rounded_height = (
            rounded_height
            if rounded_height >= self._start
            else rounded_height + 10**self.kwargs.precision
        )
        if rounded_height > 1000000:
            return 0.0
        try:
            return self._derivatives[rounded_height]
        except KeyError:
            raise Exception(
                f"Height {height}m at time {time} is not supported by the COESA76-fast atmosphere model!"
            )


def raise_for_invalid_atmos_model(atmos_model: str) -> None:
    """Raises ValueError if the given simulation method name is not defined
    
    :param atmos_model: The name of the atmosphere model
    :raises ValueError: If the atmosphere model is not supported
    """
    available_models = list(get_available_atmos_models().keys())
    if atmos_model not in available_models:
        raise ValueError(
            f"Atmosphere model {atmos_model} is not supported. Supported models are: {available_models}"
        )


def get_available_atmos_models() -> dict[str, type[AtmosphereModel]]:
    """Find available atmosphere models in atmos.py

    :return: A dictionary of available atmosphere models
    """
    return AtmosphereModel._models


if __name__ == "__main__":
    breakpoint()
