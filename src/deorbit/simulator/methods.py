import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from tqdm import tqdm

from deorbit.data_models.methods import MethodKwargs, EulerKwargs
from deorbit.utils.constants import EARTH_RADIUS
from deorbit.simulator.simulator import Simulator

class SimMethod(ABC):
    """Abstract base class for Atmosphere model implementations.
    Attributes:
        name: abstract; must be set as a class variable in any subclass

    Methods:
        density(state, time) -> float: abstract; must be implemented in any subclass
        kwargs() -> AtmosKwargs: returns a pydantic data model of model parameters
    """
    def __init__(self) -> None:
        super().__init__()
        self.kwargs: MethodKwargs = None
        
    @property
    @abstractmethod
    def name(self):
        ...
    
    def pos_from_state(self, state: np.ndarray) -> np.ndarray:
        return state[: self.kwargs.dimension]
    
    def is_terminal(self, state: np.ndarray) -> bool:
        return np.linalg.norm(self.pos_from_state(state)) <= EARTH_RADIUS
    
    def calculate_accel(self, state: np.ndarray, time: float) -> float:
        drag_accel = self._drag_accel(state=state, time=time)
        grav_accel = self._gravity_accel(state=state)
        # print(f"state {state} at time {time} has drag accel {np.linalg.norm(drag_accel)} \
        # and gravity accel {np.linalg.norm(grav_accel)}")
        return drag_accel + grav_accel
    
    def objective_function(self, state: np.ndarray, time: float) -> np.ndarray:
        """The function that gives the derivative our state vector x' = f(x,t).
        Returns a flat array (x1', x2', x3', v1', v2', v3')"""
        accel = self.calculate_accel(state, time)
        return np.concatenate((state[self.dim :], accel))
    

class Euler(SimMethod):
    """Simple forward euler integration technique"""
    name = "euler"
    
    def __init__(self, sim: Simulator, **kwargs) -> None:
        super().__init__()
        self.kwargs: EulerKwargs = EulerKwargs(**kwargs)
        self.states: list[ArrayLike] = list()
        self.times: list[float] = list()
        
    
    def step_time(self) -> None:
        self.times.append(self.times[-1] + self.kwargs.time_step)
        
    def step_state(self) -> None:
        self.step_time()
        next_state = np.array(self.states[-1], dtype=float)
        next_state += (
            self.objective_function(self.states[-1], self.times[-1])
            * self.kwargs.time_step
        )
        self.states.append(next_state)
        
    def run(self, steps: int | None) -> None:
        print("Running simulation with Euler integrator")
        # Boilerplate code for stepping the simulation
        if steps is None:
            iters = 0
            while not self.is_terminal(self.states[-1]):
                self.step_state()
                iters += 1
        else:
            for i in tqdm(range(steps)):
                if self.is_terminal(self.states[-1]):
                    iters = i
                    break
                self.step_state()
            else:
                iters = steps
                

        
        