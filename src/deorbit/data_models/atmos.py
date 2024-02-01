from typing import ClassVar

from pydantic import BaseModel

from deorbit.utils.constants import AIR_DENSITY_SEA_LEVEL, EARTH_RADIUS


class AtmosKwargs(BaseModel):
    """Point of truth for atmosphere model parameters"""

    atmos_name: ClassVar[str]


# Children of AtmosKwargs should have usable defaults for every attribute
class CoesaKwargs(AtmosKwargs):
    atmos_name = "coesa_atmos"
    earth_radius: float = EARTH_RADIUS


class CoesaFastKwargs(AtmosKwargs):
    atmos_name = "coesa_atmos_fast"
    earth_radius: float = EARTH_RADIUS
    precision: int = 2


class SimpleAtmosKwargs(AtmosKwargs):
    atmos_name = "simple_atmos"
    earth_radius: float = EARTH_RADIUS
    surf_density: float = AIR_DENSITY_SEA_LEVEL


class IcaoKwargs(AtmosKwargs):
    atmos_name = "icao_standard_atmos"
    earth_radius: float = EARTH_RADIUS


def get_model_for_atmos(atmos_model_name: str) -> type[AtmosKwargs]:
    """Returns the correct kwargs model for the given atmosphere model"""

    for model in AtmosKwargs.__subclasses__():
        if model.atmos_name == atmos_model_name:
            return model
    else:
        raise ValueError(
            f"Atmosphere model {atmos_model_name} has no supporting kwargs model!"
        )
