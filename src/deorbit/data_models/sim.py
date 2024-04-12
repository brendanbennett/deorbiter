from typing import Optional

from pydantic import BaseModel

from deorbit.data_models.atmos import AtmosKwargs
from deorbit.data_models.methods import MethodKwargs


class SimConfig(BaseModel):
    initial_state: list

    initial_time: float = 0.0

    simulation_method_kwargs: MethodKwargs

    atmosphere_model_kwargs: AtmosKwargs


class SimData(BaseModel):
    # Data Entries
    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None  # Defaults to dim = 2
    
    v1: list[float]
    v2: list[float]
    v3: Optional[list[float]] = None
    
    times: list[float]
    
