from typing import Optional

from pydantic import BaseModel


class SimConfig(BaseModel):
    # By default, our simulation is 2 dimensional
    dimension: int = 2

    time_step: Optional[float] = None

    initial_values: Optional[tuple[tuple,float]] = None

    simulation_method: Optional[str] = None

    atmosphere_model: Optional[str] = None

    atmosphere_model_kwargs: dict = dict()

    adaptive_time_stepping: Optional[bool] = None


class SimData(BaseModel):
    # Data Entries
    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None  # Defaults to dim = 2
    times: list[float]

    # Metadata
    sim_config: SimConfig
