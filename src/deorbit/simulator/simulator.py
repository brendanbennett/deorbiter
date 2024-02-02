from abc import ABC, abstractmethod
from pathlib import Path
from time import thread_time_ns
from typing import Callable

import numpy as np
import numpy.typing as npt
from numpy._typing import ArrayLike
from tqdm import tqdm

from deorbit.data_models.atmos import AtmosKwargs, get_model_for_atmos
from deorbit.data_models.methods import MethodKwargs, get_model_for_sim
from deorbit.data_models.sim import SimConfig, SimData
from deorbit.simulator.atmos import (
    AtmosphereModel,
    raise_for_invalid_atmos_model,
    get_available_atmos_models
)
from deorbit.utils.constants import (
    EARTH_RADIUS,
    GM_EARTH,
    MEAN_DRAG_COEFF,
    MEAN_XSECTIONAL_AREA,
    SATELLITE_MASS,
)
from deorbit.utils.dataio import save_sim_data


class Simulator(ABC):
    """

    Usage:
    todo UPDATE
    """

    _methods: dict = {}

    def __init_subclass__(cls, method_name: str = None, **kwargs):
        if method_name is None:
            raise SyntaxError(
                "'method_name' must be supplied as an argument when defining a subclass of Simulator"
            )
        super().__init_subclass__(**kwargs)
        cls._methods[method_name] = cls

    def __new__(cls, config: SimConfig):
        method_name = config.simulation_method_kwargs.method_name
        raise_for_invalid_sim_method(method_name)
        method_cls = cls._methods[method_name]
        return super().__new__(method_cls)

    def __init__(self, config: SimConfig) -> None:
        self.states: list[np.ndarray] = list()
        self.times: list[float] = list()
        self._atmosphere_model: Callable = None
        self.sim_method_kwargs: MethodKwargs = config.simulation_method_kwargs

        self.set_atmosphere_model(config.atmosphere_model_kwargs)
        self.set_initial_conditions(config.initial_state, config.initial_time)

    def export_config(self) -> SimConfig:
        """
        Returns:
            SimConfig: Config object which can be used to recreate this simulation
        """
        assert len(self.states) > 0 and len(self.times) > 0
        initial_values = list(self.states[0]), self.times[0]
        config = SimConfig(
            initial_values=initial_values,
            simulation_method_kwargs=self.sim_method_kwargs,
            atmosphere_model_kwargs=self._atmosphere_model.kwargs,
        )
        return config

    def _reset_state_and_time(self) -> None:
        self.states = list()
        self.times = list()

    def set_initial_conditions(self, state: np.ndarray, time: float):
        """Resets the simulation and initialises values with the given state vector and time"""
        # Makes sure state is a numpy array
        state = np.array(state, dtype=float)
        assert state.shape == (
            2 * self.dim,
        ), f"Incorrect shape for initial state {state}. Expected {(2*self.dim,)}, got {state.shape}"
        self._reset_state_and_time()
        self.states.append(state)
        self.times.append(time)

    def set_atmosphere_model(self, model_kwargs: AtmosKwargs) -> None:
        model_name = model_kwargs.atmos_name
        raise_for_invalid_atmos_model(model_name)
        self._atmosphere_model: AtmosphereModel = AtmosphereModel(model_kwargs)

    def _pos_from_state(self, state: np.ndarray) -> np.ndarray:
        return state[: self.dim]

    def _vel_from_state(self, state: np.ndarray) -> np.ndarray:
        return state[self.dim :]

    def atmosphere(self, state: np.ndarray, time: float) -> float:
        return self._atmosphere_model.density(state, time)

    def _gravity_accel(self, state: np.ndarray) -> np.ndarray:
        """Calculate acceleration by gravity.

        Args:
            state: (np.ndarray)

        Returns:
            np.ndarray: Acceleration by gravity vector
        """
        position = state[: self.dim]
        radius = np.linalg.norm(position)
        return -position * GM_EARTH / (radius**3)

    def _drag_accel(self, state: np.ndarray, time: float) -> np.ndarray:
        """Calculates acceleration on the satellite due to drag in a particular state.
        Uses the chosen atmosphere model to calculate air density.

        Args:
            state (np.ndarray)
            time (float)

        Returns:
            np.ndarray: Acceleration by drag vector
        """
        accel = (
            -(1 / (2 * SATELLITE_MASS))
            * self.atmosphere(state, time)
            * MEAN_XSECTIONAL_AREA
            * MEAN_DRAG_COEFF
            * state[self.dim :]
            * np.linalg.norm(state[self.dim :])
        )
        return accel

    def _calculate_accel(self, state: np.ndarray, time: float) -> float:
        drag_accel = self._drag_accel(state=state, time=time)
        grav_accel = self._gravity_accel(state=state)
        # print(f"state {state} at time {time} has drag accel {np.linalg.norm(drag_accel)} \
        # and gravity accel {np.linalg.norm(grav_accel)}")
        return drag_accel + grav_accel

    def _step_time(self) -> None:
        self.times.append(self.times[-1] + self.time_step)

    def _objective_function(self, state: np.ndarray, time: float) -> np.ndarray:
        """The function that gives the derivative our state vector x' = f(x,t).
        Returns a flat array (x1', x2', x3', v1', v2', v3')"""
        accel = self._calculate_accel(state, time)
        return np.concatenate((state[self.dim :], accel))

    def is_terminal(self, state: np.ndarray) -> bool:
        return np.linalg.norm(self._pos_from_state(state)) <= EARTH_RADIUS

    @abstractmethod
    def _run_method(self, steps: int | None) -> None:
        ...

    def run(self, steps: int = None):
        start_time = thread_time_ns()

        # Run with selected simulation method
        self._run_method(steps)

        elapsed_time = (thread_time_ns() - start_time) * 1e-9

        if self.is_terminal(self.states[-1]):
            print(
                f"Impacted at {self.states[-1][:self.dim]} at velocity {self.states[-1][self.dim:]} at simulated time {self.times[-1]}s."
            )

        print(f"Simulation finished in {elapsed_time:.5f} seconds")

    # def check_set_up(self) -> None:
    #     """Check all required modules are initialised"""
    #     errors = []
    #     if self._atmosphere_model is None:
    #         errors.append(
    #             'Atmosphere model hasn\'t been set! Set with set_atmosphere_model("[name]", model_kwargs)'
    #         )

    #     if self._simulation_method is None:
    #         errors.append(
    #             'Simulation method hasn\'t been set! Set with set_simulation_method("[name]")'
    #         )
    #     elif self._simulation_method not in self.available_sim_methods:
    #         errors.append(
    #             f"{self._simulation_method} is not an implemented simulation method! Must be one of: {list(self.available_sim_methods.keys())}"
    #         )

    #     if len(self.states) == 0 or len(self.times) == 0:
    #         errors.append(
    #             "Initial conditions not set! Set with set_initial_conditions(state, time)"
    #         )

    # if errors:
    #     raise NotImplementedError(" | ".join(errors))

    @property
    def time_step(self):
        return self.sim_method_kwargs.time_step

    @property
    def dim(self):
        return self.sim_method_kwargs.dimension

    @property
    def x1(self):
        return [xt[0] for xt in self.states]

    @property
    def x2(self):
        return [xt[1] for xt in self.states]

    @property
    def x3(self):
        # assert (
        #     self.dim >= 3
        # ), "Attempted to access x3 coordinate from 2D simulator"
        return [xt[2] for xt in self.states]

    def gather_data(self) -> SimData:
        """Generates a portable data object containing all the simulation data reqiured to save.

        Returns:
            SimData: pydantic data model containing both simulated data and config.
        """
        config = self.export_config()
        if self.dim == 2:
            data = SimData(
                x1=self.x1, x2=self.x2, times=self.times, sim_config=config
            )
        elif self.dim == 3:
            data = SimData(
                x1=self.x1,
                x2=self.x2,
                x3=self.x3,
                times=self.times,
                sim_config=config,
            )
        else:
            raise Exception("Sim dimension is not 2 or 3!")
        return data

    def save_data(self, save_dir_path: str) -> Path:
        """Saves simulation data to [save_dir_path] directory as defined in the SimData data model.

        File name format: sim_data_[unix time in ms].json

        Args:
            save_dir_path (Path like): Data directory to save json file.
        """
        save_sim_data(self.gather_data(), dir_path_string=save_dir_path)


class EulerSimulator(Simulator, method_name="euler"):
    def _step_state_euler(self) -> None:
        self._step_time()
        next_state = np.array(self.states[-1], dtype=float)
        next_state += (
            self._objective_function(self.states[-1], self.times[-1])
            * self.time_step
        )
        self.states.append(next_state)

    def _run_method(self, steps: int | None) -> None:
        """Simple forward euler integration technique"""
        print("Running simulation with Euler integrator")
        # Boilerplate code for stepping the simulation
        if steps is None:
            iters = 0
            while not self.is_terminal(self.states[-1]):
                self._step_state_euler()
                iters += 1
        else:
            for i in tqdm(range(steps)):
                if self.is_terminal(self.states[-1]):
                    iters = i
                    break
                self._step_state_euler()
            else:
                iters = steps

        print(
            f"Ran {iters} iterations at time step of {self.time_step} seconds"
        )


class AdamsBashforthSimulator(Simulator, method_name="adams_bashforth"):
    def _step_state_euler(self) -> None:
        self._step_time()
        next_state = np.array(self.states[-1], dtype=float)
        next_state += (
            self._objective_function(self.states[-1], self.times[-1])
            * self.time_step
        )
        self.states.append(next_state)

    def _step_state_adams_bashforth(self, buffer: list) -> None:
        func_n_minus_2, func_n_minus_1 = buffer
        # Update with two step Adams-Bashforth
        next_state = (
            self.states[-1]
            + (3 / 2) * self.time_step * func_n_minus_1
            - (1 / 2) * self.time_step * func_n_minus_2
        )
        # Update buffer with next function evaluation f(xn, tn)
        self._step_time()
        buffer.append(self._objective_function(next_state, self.times[-1]))
        del buffer[0]
        self.states.append(next_state)

    def _run_method(self, steps: int | None) -> None:
        """Two-step Adams-Bashforth integration technique.

        A linear multistep method that only samples the function at the same time steps as are output.
        This contrasts with the Runge-Kutta methods which take intermediatesamples between time steps.
        This allows buffering of previous calls to the right-hand-side function of the ODE which is
        fairly expensive."""
        print("Running simulation with Two-step Adams-Bashforth integrator")
        function_buffer = list()
        iters = 0
        # Initialise function buffer with f(x0, t0) and f(x1, t1)
        function_buffer.append(
            self._objective_function(self.states[-1], self.times[-1])
        )
        # We calculate f(x1, t1) using the forward euler method.
        self._step_state_euler()
        function_buffer.append(
            self._objective_function(self.states[-1], self.times[-1])
        )
        # Boilerplate code for stepping the simulation
        if steps is None:
            iters = 0
            while not self.is_terminal(self.states[-1]):
                self._step_state_adams_bashforth(function_buffer)
                iters += 1
        else:
            for i in tqdm(range(steps)):
                if self.is_terminal(self.states[-1]):
                    iters = i
                    break
                self._step_state_adams_bashforth(function_buffer)
            else:
                iters = steps

        print(
            f"Ran {iters} iterations at time step of {self.time_step} seconds"
        )


class RK4Simulator(Simulator, method_name="RK4"):
    def _step_state_RK4(self) -> None:
        self._step_time()
        next_state = np.array(self.states[-1])
        k1 = self._objective_function(self.states[-1], self.times[-1])
        k2 = self._objective_function(
            (self.states[-1] + (self.time_step * k1) / 2),
            (self.times[-1] + self.time_step / 2),
        )
        k3 = self._objective_function(
            (self.states[-1] + (self.time_step * k2) / 2),
            (self.times[-1] + self.time_step / 2),
        )
        k4 = self._objective_function(
            (self.states[-1] + self.time_step * k3),
            (self.times[-1] + self.time_step),
        )
        next_state += self.time_step * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.states.append(next_state)

    def _run_method(self, steps: int | None) -> None:
        """4th order Runge Kutta Numerical Integration Method"""
        print("Running simulation with RK4 integrator")
        iters = 0
        # Boilerplate code for stepping the simulation
        if steps is None:
            iters = 0
            while not self.is_terminal(self.states[-1]):
                self._step_state_RK4()
                iters += 1
        else:
            for i in tqdm(range(steps)):
                if self.is_terminal(self.states[-1]):
                    iters = i
                    break
                self._step_state_RK4()
            else:
                iters = steps

        print(
            f"Ran {iters} iterations at time step of {self.time_step} seconds"
        )


def raise_for_invalid_sim_method(sim_method: str) -> None:
    """Raises ValueError if the given simulation method name is not defined"""
    available_methods = list(get_available_sim_methods().keys())
    if sim_method not in available_methods:
        raise ValueError(
            f"Simulation method {sim_method} is not supported. Supported methods are: {available_methods}"
        )


def get_available_sim_methods() -> dict[str, type[Simulator]]:
    """Python magic to find the names of implemented simulation methods.

    Returns:
        dict[str, subclass(Simulator)]: a dictionary of `{name: method class}`
    """
    return Simulator._methods


def generate_sim_config(
    sim_method: str,
    atmos_model: str,
    initial_state: npt.ArrayLike,
    initial_time: float = 0.0,
    time_step: float = 0.1,
    sim_method_kwargs: dict | type[MethodKwargs] | None = None,
    atmos_kwargs: dict | type[AtmosKwargs] | None = None,
):
    assert len(initial_state) % 2 == 0

    raise_for_invalid_sim_method(sim_method)
    raise_for_invalid_atmos_model(atmos_model)

    dimension: int = int(len(initial_state) / 2)
    method_kwargs_model: type[MethodKwargs] = get_model_for_sim(sim_method)
    atmos_kwargs_model: type[AtmosKwargs] = get_model_for_atmos(atmos_model)

    if sim_method_kwargs is None:
        # Use the defaults set by the data model
        sim_method_kwargs = method_kwargs_model(
            dimension=dimension, time_step=time_step
        )
    elif type(sim_method_kwargs) is dict:
        # If a user supplies time_step in this dictionary, prefer it over the one supplied as an argument
        if "time_step" in sim_method_kwargs:
            time_step = sim_method_kwargs.pop("time_step")
        sim_method_kwargs = method_kwargs_model(
            dimension=dimension, time_step=time_step, **sim_method_kwargs
        )
    elif (
        type(sim_method_kwargs) is not method_kwargs_model
        and type(sim_method_kwargs) in get_available_sim_methods().values()
    ):
        raise ValueError(
            f"Mismatched kwargs object provided. Expected kwargs for {sim_method}, got kwargs for {sim_method_kwargs.method_name}"
        )
    else:
        raise ValueError(
            "Simulation method kwargs are invalid. Must either be a dict, a MethodKwargs instance or None"
        )

    if atmos_kwargs is None:
        # Use the defaults set by the data model
        atmos_kwargs = atmos_kwargs_model()
    elif type(atmos_kwargs) is dict:
        atmos_kwargs = atmos_kwargs_model(**atmos_kwargs)
    elif (
        type(atmos_kwargs) is not atmos_kwargs_model
        and type(atmos_kwargs) in get_available_atmos_models().values()
    ):
        raise ValueError(
            f"Mismatched kwargs object provided. Expected kwargs for {atmos_model}, got kwargs for {atmos_kwargs.atmos_name}"
        )
    else:
        raise ValueError(
            "Atmosphere model kwargs are invalid. Must either be a dict, a AtmosKwargs instance or None"
        )


    config = SimConfig(
        initial_state=initial_state,
        initial_time=initial_time,
        simulation_method=sim_method,
        simulation_method_kwargs=sim_method_kwargs,
        atmosphere_model=atmos_model,
        atmosphere_model_kwargs=atmos_kwargs,
    )
    return config


def run_with_config(
    config: SimConfig,
    steps: int | None = None,
) -> Simulator:
    sim = Simulator(config)
    sim.run(steps=steps)
    return sim


def run(
    sim_method: str,
    atmos_model: str,
    initial_state: npt.ArrayLike,
    initial_time: float = 0.0,
    time_step: float = 0.1,
    sim_method_kwargs: dict | type[MethodKwargs] | None = None,
    atmos_kwargs: dict | type[AtmosKwargs] | None = None,
    steps: int | None = None,
) -> Simulator:
    config = generate_sim_config(
        sim_method=sim_method,
        atmos_model=atmos_model,
        initial_state=initial_state,
        initial_time=initial_time,
        time_step=time_step,
        sim_method_kwargs=sim_method_kwargs,
        atmos_kwargs=atmos_kwargs,
    )
    sim = run_with_config(config, steps)
    return sim


if __name__ == "__main__":
    # TODO implement CLI
    breakpoint()
    pass
