from typing import Callable

import numpy as np
from ambiance import Atmosphere as IcaoAtmosphere
from pyatmos import coesa76 as _coesa76

from src.utils.constants import AIR_DENSITY_SEA_LEVEL, EARTH_RADIUS


# Marks atmosphere model factory functions
def atmosphere_model(func):
    func.__atmos__ = True
    return func

# TODO Make easier to use by expanding decorator function above.
# For now, in order to implement new density models, follow this layout. 
# To change keyword arguments simple change them in the outer function.
# Any additional constants should be assigned in src.utils.constants and imported
@atmosphere_model
def simple_atmos(
    earth_radius: float = EARTH_RADIUS,
    surf_density: float = AIR_DENSITY_SEA_LEVEL,
) -> Callable:
    """Generate simple atmospheric model

    Args:
        earth_radius (float, optional): Earth's radius in metres. Defaults to EARTH_RADIUS.
        surf_density (float, optional): Air density at Earth's surface in kgm^-3. Defaults to AIR_DENSITY_SEA_LEVEL.

    Returns:
        density (Callable): Density function taking state and time as input
        model_kwargs (dict): Model parameters
    """
    # Document the keywords used in the atmosphere model. This is required.
    model_kwargs: dict = locals()

    ## This function can be changed.
    def density(state, time):
        assert len(state) % 2 == 0
        dim = int(len(state) / 2)

        return surf_density * np.exp(
            (-(np.linalg.norm(state[:dim]) / earth_radius) + 1)
        )

    # Returning a tuple of (density function, model keyword argments) is required too.
    return density, model_kwargs

@atmosphere_model
def icao_standard_atmos(earth_radius: float = EARTH_RADIUS):
    model_kwargs: dict = locals()
    
    max_height = 81020
    density_at_max_height = IcaoAtmosphere(max_height).density
    def density(state: tuple[float], time: float):
        assert len(state) % 2 == 0
        dim = int(len(state) / 2)
        position = state[:dim]
        
        height = np.linalg.norm(position) - earth_radius
        if height <= max_height:
            return IcaoAtmosphere(height).density
        else:
            # TODO make better high altitude approx
            return density_at_max_height * np.exp(height-max_height)
    
    return density, model_kwargs

@atmosphere_model
def coesa_atmos(earth_radius: float = EARTH_RADIUS):
    model_kwargs: dict = locals()
    
    def density(state: tuple[float], time: float):
        assert len(state) % 2 == 0
        dim = int(len(state) / 2)
        position = state[:dim]
        
        height = np.linalg.norm(position) - earth_radius
        height_in_km = height*1e-3
        return _coesa76(height_in_km).rho
    
    return density, model_kwargs
