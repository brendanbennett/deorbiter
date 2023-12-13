from inspect import getmembers, isfunction
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

import src.simulator.atmos as atmos
from src.data_models import SimConfig, SimData
from src.utils.constants import (
    EARTH_RADIUS,
    GM_EARTH,
    SATELLITE_MASS,
    MEAN_DRAG_COEFF,
    MEAN_XSECTIONAL_AREA,
)
from src.utils.dataio import save_sim_data


class Simulator:
    def __init__(self, config: SimConfig) -> None:
        self.states: list[np.ndarray] = None
        self.times: list[float] = list()
        self._atmosphere_model: Callable = None
        self.config: SimConfig = config

        self.load_config(config)

    def load_config(self, config: SimConfig):
        self.states = list()
        self.config = config

        # Initialise atmosphere model if supplied
        if self.config.atmosphere_model is not None:
            self.set_atmosphere_model(
                self.config.atmosphere_model,
                self.config.atmosphere_model_kwargs,
            )

    def set_atmosphere_model(
        self, model_string: str = None, model_kwargs: dict = dict()
    ) -> None:
        models = get_available_atmos_models()
        if model_string in models:
            model_factory: Callable = models[model_string]
            self._atmosphere_model, model_kwargs = model_factory(**model_kwargs)
            self.config.atmosphere_model = model_string
            self.config.atmosphere_model_kwargs = model_kwargs
        else:
            raise ValueError(
                f"Model {model_string} is not defined in atmos.py!"
            )

    def _pos_from_state(self, state: np.ndarray) -> np.ndarray:
        return state[: self.dim]

    def _vel_from_state(self, state: np.ndarray) -> np.ndarray:
        return state[self.dim :]

    def atmosphere(self, state: np.ndarray, time: float) -> float:
        return self._atmosphere_model(state, time)

    def _gravity_accel(self, state: np.ndarray) -> np.ndarray:
        """Calculate acceleration by gravity"""
        position = state[: self.dim]
        radius = np.linalg.norm(position)
        return -position * GM_EARTH / (radius**3)

    def _drag_accel(self, state: np.ndarray, time: float) -> np.ndarray:
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
        # print(f"state {state} at time {time} has drag accel {np.linalg.norm(drag_accel)} and gravity accel {np.linalg.norm(grav_accel)}")
        return drag_accel + grav_accel

    def _step_time(self) -> None:
        self.times.append(self.times[-1] + self.config.time_step)

    def _step_state(self) -> None:
        """Super janky state step function"""
        # TODO make easily extendable.
        accel = self._calculate_accel(self.states[-1], self.times[-1])
        self._step_time()
        next_state = np.array(self.states[-1])
        # update position according to previus velocity
        next_state[: self.dim] += (
            self.states[-1][self.dim :] * self.config.time_step
        )
        # update velocity according to acceleration at previous state
        next_state[self.dim :] += accel * self.config.time_step
        self.states.append(next_state)

    def is_terminal(self, state: np.ndarray) -> bool:
        return np.linalg.norm(self._pos_from_state(state)) < EARTH_RADIUS

    def run(self, steps: int = None):
        self.check_set_up()

        iters = 0
        while not self.is_terminal(self.states[-1]):
            if steps is not None and iters >= steps:
                break
            self._step_state()
            iters += 1
        else:
            print(f"Impacted at {self.states[-1][:self.dim]} at velocity {self.states[-1][self.dim:]} at time {self.times[-1]} seconds.")
        print(
            f"Ran {iters} iterations at time step of {self.config.time_step} seconds"
        )

    def check_set_up(self) -> None:
        """Check all required modules are initialised"""
        errors = []
        if self._atmosphere_model is None:
            errors.append("Atmosphere model hasn't been set!")

        if self.config.time_step is None:
            errors.append("Time step hasn't been set!")

        if errors:
            raise NotImplementedError(" | ".join(errors))

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
        assert (
            self.dim >= 3
        ), "Attempted to access x3 coordinate from 2D simulator"
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

    def save_data(self, path: str) -> None:
        """Saves simulation data to [path] as defined in the SimData data model.

        Args:
            path (str): Path to save json data file
        """
        save_sim_data(self.gather_data(), path=path)


def get_available_atmos_models() -> dict[str:Callable]:
    """Find available atmosphere models in atmos.py

    Returns:
        dict[str, Callable]: Dictionary of model name keys and function values
    """
    full_list = getmembers(atmos, isfunction)
    # Atmosphere function factories have the __atmos__ attribute set to True.
    # This is so this function doesn't get confused with other functions defined
    # in atmos. This can be avoided by defining these functions with an _underscore
    # at the beginning but this check makes this unnecessary.
    full_list = [
        func
        for func in full_list
        if hasattr(func[1], "__atmos__") and func[1].__atmos__ == True
    ]
    return {i[0]: i[1] for i in full_list}


if __name__ == "__main__":
    # Run me with
    # mir-orbiter$ python -m src.simulator.simulator

    sim = Simulator(SimConfig(time_step=0.1, atmosphere_model="coesa_atmos"))
    # Initial conditions
    sim.states.append(
        np.array([EARTH_RADIUS + 85000, 0, 0, 8000], dtype=np.dtype("float64"))
    )
    sim.times.append(0)

    sim.run(10000)
    fig, ax = plt.subplots()
    ax.plot(sim.x1, sim.x2)
    earth = plt.Circle((0, 0), radius=EARTH_RADIUS, fill=False)
    ax.add_patch(earth)
    ax.set_xlim([5e6, 6.5e6])
    ax.set_ylim([-2e5, 4e6])
    plt.show()

    sim.save_data("sim_data.json")
