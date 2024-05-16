"""This module contains the implementation of various simulation techniques for satellite deorbiting scenarios, employing different numerical methods.

:synopsis: functions and classes for simulating satellite deorbiting.

Module Contents
---------------

The module includes the following main components:

- **Simulator**: An abstract base class for all simulators.
- **EulerSimulator**, **AdamsBashforthSimulator**, **RK4Simulator**: Subclasses of Simulator that implement specific numerical methods.
- **AtmosphereModel**: An abstract base class for all atmosphere models.
- **CoesaAtmosFast**, **CoesaAtmos**, **ZeroAtmos**: Subclasses of AtmosphereModel that implement specific atmosphere models.
- **Utility functions**: A set of functions to validate and retrieve simulation settings and run simulations with specific configurations.
"""

import deorbit.simulator.atmos as atmos

from .atmos import get_available_atmos_models
from .simulator import Simulator, generate_sim_config, get_available_sim_methods, run, run_with_config

__all__ = [
    "Simulator",
    "get_available_atmos_models",
    "get_available_sim_methods",
    "generate_sim_config",
    "run",
    "run_with_config",
]
