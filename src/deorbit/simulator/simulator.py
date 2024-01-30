from pathlib import Path
from time import thread_time_ns
from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
from numpy._typing import ArrayLike
import numpy.typing as npt
from tqdm import tqdm

from deorbit.data_models.sim import SimConfig, SimData
from deorbit.data_models.atmos import AtmosKwargs
from deorbit.data_models.methods import (
    MethodKwargs,
    EulerKwargs,
    RK4Kwargs,
    AdamsBashforthKwargs,
)
from deorbit.simulator.atmos import AtmosphereModel, get_available_atmos_models
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

    @property
    @abstractmethod
    def name(self):
        """Simulation method name"""
        ...

    def __init__(
        self,
        sim_method_kwargs: MethodKwargs,
        atmosphere_model: str,
        atmosphere_model_kwargs: AtmosKwargs,
        initial_values: tuple[npt.ArrayLike, float],
    ) -> None:
        self.states: list[np.ndarray] = list()
        self.times: list[float] = list()
        self._atmosphere_model: Callable = None
        self.sim_method_kwargs: MethodKwargs = sim_method_kwargs

        self.set_atmosphere_model(atmosphere_model, atmosphere_model_kwargs)
        initial_state, initial_time = initial_values
        self.set_initial_conditions(initial_state, initial_time)

    def export_config(self) -> SimConfig:
        """
        Returns:
            SimConfig: Config object which can be used to recreate this simulation
        """
        assert len(self.states) > 0 and len(self.times) > 0
        initial_values = list(self.states[0]), self.times[0]
        config = SimConfig(
            initial_values=initial_values,
            simulation_method=self.name,
            simulation_method_kwargs=self.sim_method_kwargs,
            atmosphere_model=self._atmosphere_model.name,
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

    def set_atmosphere_model(
        self, model_string: str, model_kwargs: AtmosKwargs
    ) -> None:
        models = get_available_atmos_models()
        if model_string in models:
            model_class = models[model_string]
            self._atmosphere_model: AtmosphereModel = model_class(
                **model_kwargs.model_dump()
            )
            model_kwargs: AtmosKwargs = self._atmosphere_model.kwargs
        else:
            raise ValueError(
                f"Model {model_string} is not defined in atmos.py! Defined models are {list(models.keys())}"
            )

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
                sim_config=self.config,
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


class EulerSimulator(Simulator):
    name = "euler"

    def __init__(
        self,
        sim_method_kwargs: EulerKwargs,
        atmosphere_model: str,
        atmosphere_model_kwargs: AtmosKwargs,
        initial_values: tuple[ArrayLike],
    ) -> None:
        super().__init__(
            sim_method_kwargs,
            atmosphere_model,
            atmosphere_model_kwargs,
            initial_values,
        )

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


class AdamsBashforthSimulator(Simulator):
    name = "adams_bashforth"

    def __init__(
        self,
        sim_method_kwargs: AdamsBashforthKwargs,
        atmosphere_model: str,
        atmosphere_model_kwargs: AtmosKwargs,
        initial_values: tuple[ArrayLike],
    ) -> None:
        super().__init__(
            sim_method_kwargs,
            atmosphere_model,
            atmosphere_model_kwargs,
            initial_values,
        )

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


class RK4Simulator(Simulator):
    name = "RK4"

    def __init__(
        self,
        sim_method_kwargs: RK4Kwargs,
        atmosphere_model: str,
        atmosphere_model_kwargs: AtmosKwargs,
        initial_values: tuple[ArrayLike],
    ) -> None:
        super().__init__(
            sim_method_kwargs,
            atmosphere_model,
            atmosphere_model_kwargs,
            initial_values,
        )

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


def get_available_sim_methods() -> dict[str, Callable]:
    """Python magic to find the names of implemented simulation methods.

    Returns:
        dict[str, subclass(Simulator)]: a dictionary of `{name: method class}`
    """
    full_list = Simulator.__subclasses__()
    return {i.name: i for i in full_list}


def get_simulator(config: SimConfig) -> Simulator:
    assert config.atmosphere_model in get_available_atmos_models()
    assert config.simulation_method in get_available_sim_methods()
    sim_cls: type[Simulator] = get_available_sim_methods()[
        config.simulation_method
    ]
    sim = sim_cls(
        sim_method_kwargs=config.simulation_method_kwargs,
        atmosphere_model=config.atmosphere_model,
        atmosphere_model_kwargs=config.atmosphere_model_kwargs,
        initial_values=config.initial_values,
    )
    return sim


if __name__ == "__main__":
    # TODO implement CLI
    breakpoint()
    pass
