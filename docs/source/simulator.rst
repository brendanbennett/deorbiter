simulator
=========

.. automodule:: deorbit.simulator
    :members:

.. automodule:: deorbit.simulator.atmos
    :members:

Examples
--------

To run a simulation with an Euler integrator:

.. code-block:: python

    config = deorbit.data_models.sim.SimConfig(
        initial_state=(deorbit.constants.EARTH_RADIUS + 100000, 0, 0, 8000),
        atmosphere_model_kwargs=deorbit.data_models.atmos.CoesaFastKwargs()
    )
    sim = simulator.EulerSimulator(config)
    sim.run(150000)
