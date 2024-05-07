from typing import Optional
import numpy as np
from pydantic import BaseModel, SerializeAsAny

from deorbit.data_models.atmos import AtmosKwargs
from deorbit.data_models.methods import MethodKwargs


class SimConfig(BaseModel):
    initial_state: list

    initial_time: float = 0.0

    # SerializeAsAny required to include all fields in children models of MethodKwargs and AtmosKwargs
    # See https://github.com/pydantic/pydantic/issues/7093
    simulation_method_kwargs: SerializeAsAny[MethodKwargs]

    atmosphere_model_kwargs: SerializeAsAny[AtmosKwargs]


class SimData(BaseModel):
    # Data Entries
    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None  # Defaults to dim = 2

    v1: list[float]
    v2: list[float]
    v3: Optional[list[float]] = None

    times: list[float]


   # Jacobian: list[np.ndarray]
