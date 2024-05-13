import pytest
import numpy as np

import deorbit.utils.coords as coords


def test_xyz_from_latlong():
    expected = {
        (0, 0, 1): (1, 0, 0),
        (0, np.pi, 1): (-1, 0, 0),
        (np.pi / 2, 0, 1): (0, 0, 1),
        (0, np.pi / 2, 1): (0, 1, 0),
        (np.pi / 4, np.pi / 2, 2): (0, np.sqrt(2), np.sqrt(2)),
    }

    for (lat, long, radius), xyz in expected.items():
        assert (
            np.max(np.abs(coords.cart_from_latlong((lat, long), radius) - xyz)) <= 1e-10
        )


def test_latlong_from_xyz():
    expected = {
        (1, 0, 0): (0, 0),
        (-1, 0, 0): (0, np.pi),
        (0, 0, 1): (np.pi / 2, 0),
        (0, 1, 0): (0, np.pi / 2),
        (0, np.sqrt(2), np.sqrt(2)): (np.pi / 4, np.pi / 2),
    }

    for xyz, latlong in expected.items():
        assert np.max(np.abs(coords.latlong_from_cart(xyz) - latlong)) <= 1e-10
