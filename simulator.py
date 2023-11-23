import numpy as np

from abc import ABC, abstractmethod
from utils.dataio import save_sim_data
from data_models import SimData


class SimulatorBase(ABC):
    @abstractmethod
    def gather_data(self) -> SimData:
        pass

    def save_data(self, path: str) -> None:
        save_sim_data(self.gather_data(), path=path)


class EmptySim(SimulatorBase):
    def __init__(self) -> None:
        super().__init__()
        self.x1: list = []
        self.x2: list = []
        self.times: list = []

    def gather_data(self) -> SimData:
        return SimData(x1=self.x1, x2=self.x2, times=self.times)


if __name__ == "__main__":
    sim = EmptySim()
    sim.x1 = np.linspace(0, 20, 20)
    sim.x2 = np.random.normal(size=20)
    sim.times = np.linspace(0, 100, 20)

    sim.save_data("sim_data.json")
