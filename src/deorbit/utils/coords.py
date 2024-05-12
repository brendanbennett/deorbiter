import numpy as np

from deorbit.utils.constants import EARTH_RADIUS

def xyz_from_latlong(latlong, radius: float = EARTH_RADIUS):
    """
    Gives x and y on the Earth surface at a specified latitude and longitude.
    This does not account for the rotation of the Earth; we assume 0N, 0E is at (1, 0, 0).
    """
    if not len(latlong) == 2:
        raise ValueError("latlong must be of length 2")
    if not radius > 0:
        raise ValueError("radius must be positive")
    lat, long = latlong
    if not -np.pi/2 <= lat <= np.pi/2:
        raise ValueError("latitude must be between -pi/2 and pi/2")
    x_earth = radius * np.cos(lat) * np.cos(long)
    y_earth = radius * np.cos(lat) * np.sin(long)
    z_earth = radius * np.sin(lat)

    return np.array([x_earth, y_earth, z_earth])

def latlong_from_xyz(xyz):
    """
    Gives latitude and longitude from x and y on the Earth surface. 
    This does not account for the rotation of the Earth; we assume 0N, 0E is at (1, 0, 0).
    """
    if not len(xyz) == 3:
        raise ValueError("xyz must be of length 3")
    x, y, z = xyz
    radius = np.linalg.norm(xyz)
    long = np.arctan2(y, x)
    lat = np.arcsin(z / radius)

    return np.array([lat, long])