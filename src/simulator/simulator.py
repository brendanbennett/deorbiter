import numpy as np

from typing import Callable
from inspect import getmembers, isfunction

from src.utils.dataio import save_sim_data
from src.data_models import SimData, SimConfig

import src.simulator.atmos as atmos


class Simulator:
    def __init__(self, config: SimConfig) -> None:
        self.x: list[tuple] = None
        self.times: list[float] = list()
        self.atmosphere_model: Callable = None
        self.config: SimConfig = config

        self.load_config(config)

    def load_config(self, config: SimConfig):
        self.x = list()
        self.config = config
        
        # Initialise atmosphere model if supplied
        if self.config.atmosphere_model is not None:
            self.set_atmosphere_model(
                self.config.atmosphere_model,
                self.config.atmosphere_model_kwargs,
            )

    def set_atmosphere_model(
        self, model_string: str, model_kwargs: dict
    ) -> None:
        models = get_available_atmos_models()
        if model_string in models:
            model_factory: Callable = models[model_string]
            self.atmosphere_model, model_kwargs = model_factory(**model_kwargs)
            self.config.atmosphere_model = model_string
            self.config.atmosphere_model_kwargs = model_kwargs
        else:
            raise ValueError(
                f"Model {model_string} is not defined in atmos.py!"
            )

    def atmosphere(self, state: list[float], time: float) -> float:
        return self.atmosphere_model(state, time)
        
    def run(self):
        self.check_set_up()
        
    def check_set_up(self) -> None:
        """Check all required modules are initialised"""
        errors = []
        if self.atmosphere_model is None:
            errors.append("Atmosphere model hasn't been set!")
            
        if errors:
            raise NotImplementedError(" ".join(errors))

    @property
    def x1(self):
        return [xt[0] for xt in self.x]

    @property
    def x2(self):
        return [xt[1] for xt in self.x]

    @property
    def x3(self):
        assert (
            self.config.dimension >= 3
        ), "Attempted to access x3 coordinate from 3D simulator"
        return [xt[2] for xt in self.x]

    def gather_data(self) -> SimData:
        """Generates a portable data object containing all the simulation data reqiured to save.

        Returns:
            SimData: pydantic data model containing both simulated data and config.
        """
        if self.config.dimension == 2:
            data = SimData(
                x1=self.x1, x2=self.x2, times=self.times, sim_config=self.config
            )
        elif self.config.dimension == 3:
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
    return {i[0]: i[1] for i in full_list}


if __name__ == "__main__":
    # Run me with
    # mir-orbiter$ python -m src.simulator.simulator

    sim = Simulator(SimConfig(time_step=1))
    sim.x = np.array([np.linspace(0, 20, 20), np.random.normal(size=20)]).T
    sim.times = np.linspace(0, 100, 20)

    sim.save_data("sim_data.json")

    # Raises unset atmos error
    # print(sim.atmosphere([1,2]))

    print(get_available_atmos_models())
    sim.set_atmosphere_model(
        "simple_atmos",
        model_kwargs={"earth_radius": 6e6, "surf_density": 1.225},
    )
    print(sim.atmosphere([10000, 7000], 10))
