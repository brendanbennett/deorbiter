from typing import ClassVar

from pydantic import BaseModel

from deorbit.utils.constants import AIR_DENSITY_SEA_LEVEL, EARTH_RADIUS, EARTH_ROTATIONAL_SPEED


class AtmosKwargs(BaseModel):
    """Point of truth for atmosphere model parameters"""

    atmos_name: ClassVar[str]
    earth_radius: float = EARTH_RADIUS
    earth_angular_velocity: float = EARTH_ROTATIONAL_SPEED


# Children of AtmosKwargs should have usable defaults for every attribute
class CoesaKwargs(AtmosKwargs):
    atmos_name = "coesa_atmos"


class CoesaFastKwargs(AtmosKwargs):
    atmos_name = "coesa_atmos_fast"
    precision: int = 2


class SimpleAtmosKwargs(AtmosKwargs):
    atmos_name = "simple_atmos"
    surf_density: float = AIR_DENSITY_SEA_LEVEL


class IcaoKwargs(AtmosKwargs):
    atmos_name = "icao_standard_atmos"


class ZeroAtmosKwargs(AtmosKwargs):
    atmos_name = "zero_atmos"


def get_model_for_atmos(atmos_model_name: str) -> type[AtmosKwargs]:
    """
    Returns the correct kwargs model for the given atmosphere model.

    Args:
        atmos_model_name (str): The name of the atmosphere model.

    Returns:
        type[AtmosKwargs]: The kwargs model corresponding to the given atmosphere model.

    Raises:
        ValueError: If the atmosphere model has no supporting kwargs model.
    """
    for model in AtmosKwargs.__subclasses__():
        if model.atmos_name == atmos_model_name:
            return model
    else:
        raise ValueError(
            f"Atmosphere model {atmos_model_name} has no supporting kwargs model!"
        )
