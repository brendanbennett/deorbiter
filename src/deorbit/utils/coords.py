from typing import Sequence

import numpy as np

from deorbit.utils.constants import EARTH_RADIUS, EARTH_ROTATIONAL_SPEED


def cart_from_latlong(latlong, radius: float = EARTH_RADIUS):
    """
    Given a sequence of latitude and longitude, calculates the cartesian coordinates of a point on the Earth's surface.
    This does not account for the rotation of the Earth; we assume 0N, 0E is at (1, 0, 0).
    If working in 2D, latlong should be a scalar representing longitude.
    """
    if not radius > 0:
        raise ValueError("radius must be positive")
    if isinstance(latlong, float) or len(latlong) == 1:
        return np.array([radius * np.cos(latlong), radius * np.sin(latlong)])
    elif len(latlong) == 2:
        lat, long = latlong
        if not -np.pi / 2 <= lat <= np.pi / 2:
            raise ValueError("latitude must be between -pi/2 and pi/2")
        x_earth = radius * np.cos(lat) * np.cos(long)
        y_earth = radius * np.cos(lat) * np.sin(long)
        z_earth = radius * np.sin(lat)

        return np.array([x_earth, y_earth, z_earth])
    else:
        raise ValueError("latlong must be of length 1 or 2")


def latlong_from_cart(cart, return_radius: bool = False):
    """
    Given a sequence of cartesian coordinates, calculates the latitude and longitude of a point on the Earth's surface.
    This point is the intersection of the line from the origin to the point and the Earth's surface.
    This does not account for the rotation of the Earth; we assume 0N, 0E is at (1, 0, 0).
    If working in 2D, only returns longitude.
    """
    if len(cart) == 3:
        x, y, z = cart
        radius = np.linalg.norm(cart)
        long = np.arctan2(y, x)
        lat = np.arcsin(z / radius)
        if return_radius:
            return np.array([lat, long, radius])
        return np.array([lat, long])
    elif len(cart) == 2:
        x, y = cart
        long = np.arctan2(y, x)
        if return_radius:
            return np.array([long, np.linalg.norm(cart)])
        return long
    else:
        raise ValueError("cart must be of length 2 or 3")

def earth_rotate_cart(cart_vector: Sequence[float], time: float) -> np.ndarray:
    """Calculate the real latitude and longitude of a cartesian point above Earth's surface due to the Earth's rotation underneath it.
    
    :param cart_vector: The cartesian coordinates of the point.
    :param time: The time in seconds since the start of the simulation.
    """
    if len(cart_vector) != 3:
        long, r = latlong_from_cart(cart_vector, return_radius=True)
        long -= EARTH_ROTATIONAL_SPEED * time
        return cart_from_latlong((long,), radius=r)
    else:
        lat, long, r = latlong_from_cart(cart_vector, return_radius=True)
        long -= EARTH_ROTATIONAL_SPEED * time
        return cart_from_latlong((lat, long), radius=r)
