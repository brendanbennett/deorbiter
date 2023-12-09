import numpy as np
from typing import Callable
from src.utils.constants import EARTH_RADIUS, AIR_DENSITY_SEA_LEVEL

# TODO Make easier to use
def simple_atmos(earth_radius: float=EARTH_RADIUS, surf_density: float=AIR_DENSITY_SEA_LEVEL) -> Callable:
    """Generate simple atmospheric model

    Args:
        earth_radius (float, optional): Earth's radius in metres. Defaults to EARTH_RADIUS.
        surf_density (float, optional): Air density at Earth's surface in kgm^-3. Defaults to AIR_DENSITY_SEA_LEVEL.

    Returns:
        density (Callable): Density function taking state and time as input
        model_kwargs (dict): Model parameters
    """
    # Document the keywords used in the atmosphere model
    model_kwargs: dict = locals()
    def density(state, time):
        assert len(state) % 2 == 0
        dim = int(len(state)/2)
        
        return surf_density * np.exp((-(np.linalg.norm(state[:dim])/earth_radius) + 1))
    return density, model_kwargs