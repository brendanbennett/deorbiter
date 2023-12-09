import numpy as np

# TODO Replace with factory
def simple_atmos(state, time, earth_radius, surf_density):
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