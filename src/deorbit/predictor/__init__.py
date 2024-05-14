"""This module contains the :class:`EKF` class, which is used to predict the future position of a satellite.

:synopsis: A module for predicting the future position of a satellite.
"""

from .EKF import EKF, EKFOnline

__all__ = ["EKF", "EKFOnline"]