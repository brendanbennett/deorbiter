from typing import Optional

from pydantic import BaseModel


class ObsData(BaseModel):
    """
    Output of the Observer which is a sparser copy of SimData from the Simulator.
    Only includes the states at times where the satellite is in view of a ground radar station
    """

    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None
    times: list[float]
