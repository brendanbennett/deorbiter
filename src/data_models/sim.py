from typing import Optional

from pydantic import BaseModel


class SimConfig(BaseModel):
    # By default, our simulation is 2 dimensional
    dimension: int = 2

    time_step: Optional[float] = None

    initial_state: Optional[tuple[float]] = None
    
    initial_time: Optional[float] = None

    simulation_method: Optional[str] = None

    atmosphere_model: Optional[str] = None

    atmosphere_model_kwargs: dict = dict()


class SimData(BaseModel):
    # Data Entries
    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None  # Defaults to dim = 2
    times: list[float]

    # Metadata
    sim_config: SimConfig
