from typing import Optional

from pydantic import BaseModel
from deorbit.data_models.atmos import AtmosKwargs
from deorbit.data_models.methods import MethodKwargs


class SimConfig(BaseModel):
    initial_values: Optional[tuple[tuple,float]] = None

    simulation_method: Optional[str] = None
    
    simulation_method_kwargs: Optional[MethodKwargs] = None

    atmosphere_model: Optional[str] = None

    atmosphere_model_kwargs: Optional[AtmosKwargs] = None


class SimData(BaseModel):
    # Data Entries
    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None  # Defaults to dim = 2
    times: list[float]

    # Metadata
    sim_config: SimConfig
