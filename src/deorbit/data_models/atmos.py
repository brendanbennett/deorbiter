from pydantic import BaseModel
from deorbit.utils.constants import EARTH_RADIUS, AIR_DENSITY_SEA_LEVEL


class AtmosKwargs(BaseModel):
    """ Point of truth for atmosphere model parameters
    """
    ...


class CoesaKwargs(AtmosKwargs):
    earth_radius: float = EARTH_RADIUS


class CoesaFastKwargs(AtmosKwargs):
    earth_radius: float = EARTH_RADIUS
    precision: int = 2


class SimpleAtmosKwargs(AtmosKwargs):
    earth_radius: float = EARTH_RADIUS
    surf_density: float = AIR_DENSITY_SEA_LEVEL


class IcaoKwargs(AtmosKwargs):
    earth_radius: float = EARTH_RADIUS