import pytest

from src.simulator.simulator import Simulator, get_available_atmos_models
from src.data_models.sim import SimConfig
from src.simulator.atmos import simple_atmos
from src.utils.constants import EARTH_RADIUS, AIR_DENSITY_SEA_LEVEL

def test_atmos_model_not_set():
    sim = Simulator(SimConfig())
    with pytest.raises(NotImplementedError):
        sim.check_set_up()
        
def test_simple_atmos():
    state = (200, 0, -3, 20)
    time = 0.1
    model_kwargs = {"earth_radius":200, "surf_density":1}
    density_func, returned_model_kwargs = simple_atmos(earth_radius=200, surf_density=1)
    density = density_func(state=state, time=time)
    assert density == 1
    assert model_kwargs == returned_model_kwargs
    
def test_simple_atmos_defaults():
    state = (EARTH_RADIUS, 0, -3, 20)
    time = 0.1
    density_func, returned_model_kwargs = simple_atmos()
    density = density_func(state=state, time=time)
    assert density == AIR_DENSITY_SEA_LEVEL
    assert set(returned_model_kwargs.values()) == set([EARTH_RADIUS, AIR_DENSITY_SEA_LEVEL])
    
def test_atmos_model_set():
    models = get_available_atmos_models()
    sim = Simulator(SimConfig(atmosphere_model=next(iter(models))))
    sim.check_set_up()
    