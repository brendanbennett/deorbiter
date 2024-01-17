from inspect import getmembers, isclass
from pathlib import Path
from time import thread_time_ns
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import deorbit.simulator.atmos as atmos
from deorbit.data_models import SimConfig, SimData
from deorbit.simulator.atmos import AtmosphereModel
from deorbit.utils.constants import (EARTH_RADIUS, GM_EARTH, MEAN_DRAG_COEFF,
                                     MEAN_XSECTIONAL_AREA, SATELLITE_MASS)
from deorbit.utils.dataio import save_sim_data


def sim_method(name: str) -> Callable:
    """Decorator to mark a Simulator class method as one which runs a simulation method.
    A name must be provided as an argument. This name string will be the one used by
    the end user to specify the simulation method at run time.

    Usage:
    ```
    @sim_method("forward_euler")
    def _run_forward_euler(...):
        ...
    ```"""

    # Check that the programmer has provided a name.
    if not isinstance(name, str):
        raise SyntaxError(
            "Simulation method decorator must be supplied with a name!"
        )

    def wrapper(method: Callable) -> Callable:
        method.__sim_method_name__: str = name
        return method

    return wrapper


class Simulator:
    """Simulator class used to generate satellite simulation data.
    Must be initialised with a SimConfig instance. This config may be empty on initialisation,
    but the

    Usage:
    ```
    sim_config = SimConfig(
        time_step=0.1,
        atmosphere_model="coesa_atmos",
        simulation_method="euler",
    )
    sim = Simulator(sim_config)
    sim.run(steps=10000)
    ```
    """

    def __init__(self, config: SimConfig) -> None:
        self.states: list[np.ndarray] = list()
        self.times: list[float] = list()
        self._atmosphere_model: Callable = None
        self._simulation_method: str = None
        self.config: SimConfig = None
        self.available_sim_methods = get_available_sim_methods()

        self.load_config(config)

    def load_config(self, config: SimConfig):
        self.config = config

        # Initialise atmosphere model if supplied
        if self.config.atmosphere_model is not None:
            self.set_atmosphere_model(
                self.config.atmosphere_model,
                self.config.atmosphere_model_kwargs,
            )

        if self.config.simulation_method is not None:
            self.set_simulation_method(self.config.simulation_method)

        if self.config.initial_values is not None:
            initial_state, initial_time = self.config.initial_values
            self.set_initial_conditions(initial_state, initial_time)

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
        self, model_string: str = None, model_kwargs: dict = dict()
    ) -> None:
        models = get_available_atmos_models()
        if model_string in models:
            model_class = models[model_string]
            self._atmosphere_model: AtmosphereModel = model_class(
                **model_kwargs
            )
            model_kwargs = self._atmosphere_model.model_kwargs()
            self.config.atmosphere_model = model_string
            self.config.atmosphere_model_kwargs = model_kwargs
        else:
            raise ValueError(
                f"Model {model_string} is not defined in atmos.py! Defined models are {list(models.keys())}"
            )

    def set_simulation_method(self, sim_method_name: str = None):
        if sim_method_name not in self.available_sim_methods:
            raise NotImplementedError(
                f"{self._simulation_method} is not an implemented simulation method! Must be one of: {list(self.available_sim_methods.keys())}"
            )
        self._simulation_method = sim_method_name
        self.config.simulation_method = sim_method_name

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

    def is_terminal(self, state: np.ndarray) -> bool:
        return np.linalg.norm(self._pos_from_state(state)) <= EARTH_RADIUS

    @sim_method("euler")
    def _run_euler(self, steps: int | None) -> None:
        """Simple forward euler integration technique"""
        print("Running simulation with Euler integrator")
        iters = 0
        while not self.is_terminal(self.states[-1]):
            if steps is not None and iters >= steps:
                break
            self._step_state_euler()
            iters += 1

        print(
            f"Ran {iters} iterations at time step of {self.time_step} seconds"
        )

    @sim_method("adams_bashforth")
    def _run_adams_bashforth(self, steps: int | None) -> None:
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

        while not self.is_terminal(self.states[-1]):
            if steps is not None and iters >= steps:
                break
            self._step_state_adams_bashforth(function_buffer)
            iters += 1

        print(
            f"Ran {iters} iterations at time step of {self.time_step} seconds"
        )

    def run(self, steps: int = None):
        self.check_set_up()

        start_time = thread_time_ns()

        # Run with selected simulation method
        getattr(self, self.available_sim_methods[self._simulation_method])(
            steps
        )

        elapsed_time = (thread_time_ns() - start_time) * 1e-9

        if self.is_terminal(self.states[-1]):
            print(
                f"Impacted at {self.states[-1][:self.dim]} at velocity {self.states[-1][self.dim:]} at simulated time {self.times[-1]}s."
            )

        print(f"Simulation finished in {elapsed_time:.5f} seconds")

    def check_set_up(self) -> None:
        """Check all required modules are initialised"""
        errors = []
        if self._atmosphere_model is None:
            errors.append(
                'Atmosphere model hasn\'t been set! Set with set_atmosphere_model("[name]", model_kwargs)'
            )

        if self.config.time_step is None:
            errors.append("Time step hasn't been set!")

        if self._simulation_method is None:
            errors.append(
                'Simulation method hasn\'t been set! Set with set_simulation_method("[name]")'
            )
        elif self._simulation_method not in self.available_sim_methods:
            errors.append(
                f"{self._simulation_method} is not an implemented simulation method! Must be one of: {list(self.available_sim_methods.keys())}"
            )

        if len(self.states) == 0 or len(self.times) == 0:
            errors.append(
                "Initial conditions not set! Set with set_initial_conditions(state, time)"
            )

        if errors:
            raise NotImplementedError(" | ".join(errors))

    @property
    def time_step(self):
        return self.config.time_step

    @property
    def dim(self):
        return self.config.dimension

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
        if self.dim == 2:
            data = SimData(
                x1=self.x1, x2=self.x2, times=self.times, sim_config=self.config
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


def get_available_atmos_models() -> dict[str:Callable]:
    """Find available atmosphere models in atmos.py

    Returns:
        dict[str, subclass(AtmosphereModel)]: Dictionary of {model name: subclass of AtmosphereModel}
    """
    full_list = getmembers(atmos, isclass)
    # Atmosphere function factories have the __atmos__ attribute set to True.
    # This is so this function doesn't get confused with other functions defined
    # in atmos. This can be avoided by defining these functions with an _underscore
    # at the beginning but this check makes this unnecessary.
    full_list = [
        cls
        for cls in full_list
        if issubclass(cls[1], AtmosphereModel) and cls[1] != AtmosphereModel
    ]
    return {i[1].name: i[1] for i in full_list}


def get_available_sim_methods() -> dict[str, str]:
    """Python magic to find the names of implemented simulation methods marked with `@sim_method("[name]")`.

    Returns:
        dict[str, str]: a dictionary of `{human readable name: method name}`"""
    return {
        getattr(Simulator, i)
        .__sim_method_name__: getattr(Simulator, i)
        .__name__
        for i in dir(Simulator)
        if hasattr(getattr(Simulator, i), "__sim_method_name__")
    }


if __name__ == "__main__":
    # TODO implement CLI
    pass
