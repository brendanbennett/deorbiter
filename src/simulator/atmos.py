import numpy as np
from src.utils.constants import EARTH_RADIUS, AIR_DENSITY_SEA_LEVEL

# TODO Make easier to use
def simple_atmos(earth_radius=EARTH_RADIUS, surf_density=AIR_DENSITY_SEA_LEVEL):
    # Document the keywords used in the atmosphere model
    model_kwargs = locals()
    def density(state, time):
        """_summary_

        Args:
            state (tuple[float]): state vector (x1, x2, [x3], v1, v2, [v3])
            time (float): time of state
            earth_radius (float): radius of earth's surface
            surf_density (float): density of earth's surface
        """
        assert len(state) % 2 == 0
        dim = int(len(state)/2)
        
        return surf_density * np.exp((-(np.linalg.norm(state[:dim])/earth_radius) + 1))
    return density, model_kwargs