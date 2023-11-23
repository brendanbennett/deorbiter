from typing import Optional

from pydantic import BaseModel, NonNegativeFloat


class SimData(BaseModel):
    # Data Entries
    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None  # Defaults to dim = 2
    times: list[NonNegativeFloat]

    # Metadata
    # TODO Add more useful information

    # By default, our simulation 2 dimensional
    dimension: int = 2

    time_step: Optional[float] = None

    simulation_technique: Optional[str] = None

    atmosphere_model: Optional[str] = None
