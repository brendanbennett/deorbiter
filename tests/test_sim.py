import pytest

from src.data_models.sim import SimConfig
from src.simulator.atmos import simple_atmos
from src.simulator.simulator import Simulator, get_available_atmos_models
from src.utils.constants import AIR_DENSITY_SEA_LEVEL, EARTH_RADIUS


def test_simple_atmos():
    state = (200, 0, -3, 20)
    time = 0.1
    model_kwargs = {"earth_radius": 200, "surf_density": 1}
    density_func, returned_model_kwargs = simple_atmos(
        earth_radius=200, surf_density=1
    )
    density = density_func(state=state, time=time)
    assert density == 1
    assert model_kwargs == returned_model_kwargs


def test_simple_atmos_defaults():
    state = (EARTH_RADIUS, 0, -3, 20)
    time = 0.1
    density_func, returned_model_kwargs = simple_atmos()
    density = density_func(state=state, time=time)
    assert density == AIR_DENSITY_SEA_LEVEL
    assert set(returned_model_kwargs.values()) == set(
        [EARTH_RADIUS, AIR_DENSITY_SEA_LEVEL]
    )


@pytest.mark.parametrize("model", list(get_available_atmos_models().keys()))
def test_set_atmos_model_with_config(model):
    sim = Simulator(
        SimConfig(
            atmosphere_model=model, time_step=0.1, simulation_method="euler"
        )
    )
    assert sim.config.atmosphere_model == model


@pytest.mark.parametrize("model", list(get_available_atmos_models().keys()))
def test_set_atmos_model_with_method(model):
    sim = Simulator(SimConfig(time_step=0.1, simulation_method="euler"))
    sim.set_atmosphere_model(model)
    assert sim.config.atmosphere_model == model


def test_set_simple_atmos_defaults():
    sim = Simulator(SimConfig(atmosphere_model="simple_atmos"))
    state = (EARTH_RADIUS, 0, -3, 20)
    time = 0.1
    density = sim.atmosphere(state=state, time=time)
    assert density == AIR_DENSITY_SEA_LEVEL
    assert set(sim.config.atmosphere_model_kwargs.values()) == set(
        [EARTH_RADIUS, AIR_DENSITY_SEA_LEVEL]
    )


@pytest.mark.parametrize("model", list(get_available_atmos_models().keys()))
def test_atmos_eval(model):
    sim = Simulator(SimConfig())
    sim.set_atmosphere_model(model)
    state = (EARTH_RADIUS + 8000, 10000, 0, 0)
    density = sim.atmosphere(state=state, time=1)
    assert density


def test_mandatory_fields():
    full_dict = {
        "atmosphere_model": "coesa_atmos",
        "time_step": 0.1,
        "simulation_method": "euler",
        "initial_values": ((EARTH_RADIUS + 8000, 10000, 0, 0), 0.0),
    }
    sim = Simulator(SimConfig(**full_dict))
    sim.check_set_up()

    for k in full_dict.keys():
        part_dict = {i: full_dict[i] for i in full_dict if i != k}
        sim = Simulator(SimConfig(**part_dict))
        with pytest.raises(NotImplementedError):
            sim.check_set_up()
