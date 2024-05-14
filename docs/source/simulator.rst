Simulator
=========

.. module:: deorbit.simulator
   :platform: Unix, Windows
   :synopsis: Provides functions and classes for simulating satellite deorbiting.

This module contains the implementation of various simulation techniques for satellite deorbiting scenarios, employing different numerical methods.

Classes
-------

Simulator
---------

.. autoclass:: deorbit.simulator.Simulator
   :members:

   Base class for simulation classes. Defines the framework within which different numerical methods can be implemented.

Functions
---------

.. autofunction:: deorbit.simulator.get_available_sim_methods
.. autofunction:: deorbit.simulator.generate_sim_config
.. autofunction:: deorbit.simulator.run_with_config
.. autofunction:: deorbit.simulator.run

Module Contents
---------------

The module includes the following main components:

- **Simulator**: An abstract base class for all simulators.
- **EulerSimulator**, **AdamsBashforthSimulator**, **RK4Simulator**: Subclasses of Simulator that implement specific numerical methods.
- **Utility functions**: A set of functions to validate and retrieve simulation settings and run simulations with specific configurations.

Examples
--------

To run a simulation with an Euler integrator:

.. code-block:: python

    config = deorbit.data_models.sim.SimConfig(
        initial_state=(deorbit.constants.EARTH_RADIUS + 100000, 0, 0, 8000),
        simulation_method_kwargs=deorbit.data_models.methods.RK4Kwargs(time_step=0.1),
        atmosphere_model_kwargs=deorbit.data_models.atmos.CoesaFastKwargs()
    )
    sim = simulator.EulerSimulator(config)
    sim.run(150000)
