import numpy as np
import pytest

from deorbit.data_models.atmos import SimpleAtmosKwargs, get_model_for_atmos
from deorbit.data_models.methods import EulerKwargs, get_model_for_sim
from deorbit.data_models.sim import SimConfig
from deorbit.simulator import (
    Simulator,
    generate_sim_config,
    get_available_sim_methods,
    run,
)
from deorbit.simulator.atmos import get_available_atmos_models
from deorbit.utils.constants import AIR_DENSITY_SEA_LEVEL, EARTH_RADIUS


@pytest.mark.parametrize(
    "method, atmos",
    zip(
        list(get_available_sim_methods().keys()),
        list(get_available_atmos_models().keys()),
    ),
)
def test_generate_config(method, atmos):
    initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
    config = generate_sim_config(method, atmos, initial_state)
    assert config.atmosphere_model_kwargs.atmos_name == atmos
    assert config.simulation_method_kwargs.method_name == method
    assert np.all(config.initial_state == initial_state)
    assert config.initial_time == 0.0


@pytest.mark.parametrize("method", list(get_available_sim_methods().keys()))
def test_sample_simulation(method):
    initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
    run(method, "coesa_atmos_fast", initial_state, steps=1000)


def test_defining_sim_class_no_name():
    """Simulator subclasses need a name defined"""
    with pytest.raises(SyntaxError):

        class MySimulator(Simulator):
            def _run_method(self):
                pass


def test_defining_sim_class_no_run_method():
    """Simulator subclasses need a `_run_method` method"""

    class MySimulator(Simulator, method_name="mysim"):
        pass

    with pytest.raises(TypeError):
        MySimulator()


def test_adding_sim_subclass():
    """Creating a subclass should update the name dictionary in the parent class"""
    method_name = "mysimulator"

    class MySimulator(Simulator, method_name=method_name):
        pass

    assert method_name in Simulator._methods
    assert method_name in get_available_sim_methods()


def test_raise_for_invalid_sim_method():
    initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
    sim_kwargs = EulerKwargs(time_step=0.1)
    EulerKwargs.method_name = "not_a_method"
    config = SimConfig(
        initial_state=initial_state,
        simulation_method_kwargs=sim_kwargs,
        atmosphere_model_kwargs=SimpleAtmosKwargs(),
    )
    with pytest.raises(ValueError):
        Simulator(config)


@pytest.mark.parametrize("noise_types", [["gaussian"], None, ["impulse", "gaussian"]])
def test_gaussian_noise(noise_types):
    initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
    config = generate_sim_config("RK4", "coesa_atmos_fast", initial_state)
    sim = Simulator(config)
    sim._calculate_accel(sim.states[0], 0)

    random_config = generate_sim_config(
        "RK4",
        "coesa_atmos_fast",
        initial_state,
        noise_strengths=[100] * len(noise_types) if noise_types is not None else None,
        noise_types=noise_types,
    )
    random_sim = Simulator(random_config)

    if noise_types is None:
        assert np.all(sim._calculate_accel(
            sim.states[0], 0
        ) == random_sim._calculate_accel(random_sim.states[0], 0))
    else:
        assert np.all(sim._calculate_accel(
            sim.states[0], 0
        ) != random_sim._calculate_accel(random_sim.states[0], 0))
        
        
def test_impulse_noise(monkeypatch):
    # The random function is replaced with a function that always returns 0, 
    # so 'collision' always occurs.
    with monkeypatch.context() as m:
        m.setattr(np.random, "random", lambda *args: 0)
        initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
        config = generate_sim_config("RK4", "coesa_atmos_fast", initial_state)
        sim = Simulator(config)
        sim._calculate_accel(sim.states[0], 0)

        random_config = generate_sim_config(
            "RK4",
            "coesa_atmos_fast",
            initial_state,
            noise_strengths=[100],
            noise_types=["impulse"],
        )
        random_sim = Simulator(random_config)
        
        assert np.all(sim._calculate_accel(
            sim.states[0], 0
        ) != random_sim._calculate_accel(random_sim.states[0], 0))

