import numpy as np

from utils.dataio import save_sim_data
from data_models import SimData, SimConfig


class Simulator:
    def __init__(self, config: SimConfig) -> None:
        self.x: list[tuple] = None
        self.times: list[float] = list()
        self.config: SimConfig = config

        self.load_config(config)

    def load_config(self, config: SimConfig):
        self.x = list()
        self.config = config
    
    @property
    def x1(self):
        return [xt[0] for xt in self.x]
    
    @property
    def x2(self):
        return [xt[1] for xt in self.x]
    
    @property
    def x3(self):
        assert self.config.dimension >= 3, "Attempted to access x3 coordinate from 3D simulator"
        return [xt[2] for xt in self.x]
        
    def gather_data(self) -> SimData:
        if self.config.dimension == 2:
            data = SimData(x1=self.x1, x2=self.x2, times=self.times, sim_config=self.config)
        elif self.config.dimension == 3:
            data = SimData(x1=self.x1, x2=self.x2, x3=self.x3, times=self.times, sim_config=self.config)
        else:
            raise Exception("Sim dimension is not 2 or 3!")
        return data

    def save_data(self, path: str) -> None:
        save_sim_data(self.gather_data(), path=path)


if __name__ == "__main__":
    sim = Simulator(SimConfig(time_step=1))
    sim.x = np.array([np.linspace(0, 20, 20), np.random.normal(size=20)]).T
    sim.times = np.linspace(0, 100, 20)

    sim.save_data("sim_data.json")
