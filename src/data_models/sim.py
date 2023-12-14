from typing import Optional

from pydantic import BaseModel, NonNegativeFloat


class SimConfig(BaseModel):
    # By default, our simulation 2 dimensional
    dimension: int = 2

    time_step: Optional[float] = None

    # TODO make Enums so we only have a few options
    simulation_method: Optional[str] = None

    atmosphere_model: Optional[str] = None

    atmosphere_model_kwargs: dict = dict()


class SimData(BaseModel):
    # Data Entries
    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None  # Defaults to dim = 2
    times: list[NonNegativeFloat]

    # Metadata
    sim_config: SimConfig
