import numpy as np
from typing import Callable
from src.utils.constants import EARTH_RADIUS, AIR_DENSITY_SEA_LEVEL


# TODO Make easier to use
# For now, in order to implement new density models, follow this layout. 
# To change keyword arguments simple change them in the outer function.
# Any additional constants should be assigned in src.utils.constants and imported
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
