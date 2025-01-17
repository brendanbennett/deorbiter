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
    assert len(config.simulation_method_kwargs.noise_types) == 0
    assert np.all(config.initial_state == initial_state)
    assert config.initial_time == 0.0


def test_export_config():
    initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
    config = generate_sim_config(
        "euler",
        "coesa_atmos_fast",
        initial_state,
        noise_types={
            "impulse": {"impulse_probability": 0.5},
            "gaussian": {"noise_strength": 0.1},
        },
    )
    sim = Simulator(config)

    assert sim.export_config() == config


def test_dict_config():
    initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
    config = generate_sim_config("RK4", "coesa_atmos_fast", initial_state)
    method = config.simulation_method_kwargs.method_name
    method_kwargs = config.simulation_method_kwargs.model_dump()
    atmos = config.atmosphere_model_kwargs.atmos_name
    atmos_kwargs = config.atmosphere_model_kwargs.model_dump()
    new_config = generate_sim_config(
        method,
        atmos,
        initial_state,
        sim_method_kwargs=method_kwargs,
        atmos_kwargs=atmos_kwargs,
    )
    assert new_config == config


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


@pytest.mark.parametrize(
    "noise_types",
    [
        {"gaussian": {}},
        None,
        {"impulse": {"impulse_probability": 0.5}, "gaussian": {"noise_strength": 0.1}},
    ],
)
def test_gaussian_noise(noise_types):
    initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
    config = generate_sim_config("RK4", "coesa_atmos_fast", initial_state)
    sim = Simulator(config)
    sim._calculate_accel(sim.states[0], 0)

    random_config = generate_sim_config(
        "RK4",
        "coesa_atmos_fast",
        initial_state,
        noise_types=noise_types,
    )
    random_sim = Simulator(random_config)

    if noise_types is None:
        assert np.all(
            sim._calculate_accel(sim.states[0], 0)
            == random_sim._calculate_accel(random_sim.states[0], 0)
        )
    else:
        assert np.all(
            sim._calculate_accel(sim.states[0], 0)
            != random_sim._calculate_accel(random_sim.states[0], 0)
        )


def test_impulse_noise():
    initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
    config = generate_sim_config("RK4", "coesa_atmos_fast", initial_state)
    sim = Simulator(config)
    sim._calculate_accel(sim.states[0], 0)

    random_config = generate_sim_config(
        "RK4",
        "coesa_atmos_fast",
        initial_state,
        noise_types={"impulse": {"impulse_probability": 1.0}},
    )
    random_sim = Simulator(random_config)

    assert np.all(
        sim._calculate_accel(sim.states[0], 0)
        != random_sim._calculate_accel(random_sim.states[0], 0)
    )
