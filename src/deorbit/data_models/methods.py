from typing import Optional

from pydantic import BaseModel


class MethodKwargs(BaseModel):
    # By default, our simulation is 2 dimensional
    dimension: int = 2


class RK4Kwargs(MethodKwargs):
    time_step: float


class AdamsBashforthKwargs(MethodKwargs):
    time_step: float


class EulerKwargs(MethodKwargs):
    time_step: float
