from typing import Sequence

import numpy as np

from deorbit.utils.constants import EARTH_RADIUS, EARTH_ROTATIONAL_SPEED


def cart_from_latlong(latlong, radius: float = EARTH_RADIUS):
    """
    Given a sequence of latitude and longitude, calculates the cartesian coordinates of a point on the Earth's surface.
    This does not account for the rotation of the Earth; we assume 0N, 0E is at (1, 0, 0).
    If working in 2D, latlong should be a scalar representing longitude.

    :param latlong: The latitude and longitude (or just longitude for 2D) of the point.
    :type latlong: float or sequence of floats
    :param radius: The radius of the Earth (or other body), defaults to EARTH_RADIUS.
    :type radius: float
    :return: The Cartesian coordinates of the point.
    :rtype: np.ndarray
    :raises ValueError: If the radius is not positive or if the latitude is out of bounds.
    :raises ValueError: If `latlong` is not of length 1 or 2.
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

    :param cart: The Cartesian coordinates of the point.
    :type cart: sequence of floats
    :param return_radius: Whether to return the radius as well, defaults to False.
    :type return_radius: bool
    :return: The latitude and longitude (and radius if specified) of the point.
    :rtype: np.ndarray
    :raises ValueError: If `cart` is not of length 2 or 3.
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


def earth_rotation(
    cart_vector: Sequence[float], 
    time: float, 
    return_radius: bool = False
) -> np.ndarray:
    """Calculate the real latitude and longitude of a cartesian point above Earth's surface due to the Earth's rotation underneath it.

    :param cart_vector: The cartesian coordinates of the point.
    :param time: The time in seconds since the start of the simulation.
    :param return_radius: Whether to return the radius of the point.
    :return: The longitude and latitude of the point.
    """
    if len(cart_vector) != 3:
        long, r = latlong_from_cart(cart_vector, return_radius=True)
        long -= EARTH_ROTATIONAL_SPEED * time
        if return_radius:
            return np.array([long, r])
        return np.array([long])
    else:
        lat, long, r = latlong_from_cart(cart_vector, return_radius=True)
        long -= EARTH_ROTATIONAL_SPEED * time
        if return_radius:
            return np.array([lat, long, r])
        return np.array([lat, long])


def earth_rotation_array(
    cart_positions: np.ndarray, 
    times: float, 
    return_radius: bool = True
) -> np.ndarray:
    """Calculate the real latitude and longitude of points above Earth's surface due to the Earth's rotation underneath it over time.
    Useful for plotting trajectories on a real map.

    :param cart_vectors: The cartesian positions of the points.
    :param time: The time in seconds since the start of the simulation.
    :param return_radius: Whether to return the radius of the points.
    :return: The longitude and latitude of the points.
    """
    return np.array([earth_rotation(v, t, return_radius=return_radius) for v, t in zip(cart_positions, times)])
