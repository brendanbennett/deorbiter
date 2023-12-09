import pytest

from src.simulator.simulator import Simulator, get_available_atmos_models
from src.data_models.sim import SimConfig
from src.simulator.atmos import simple_atmos

def test_atmos_model_not_set():
    sim = Simulator(SimConfig())
    with pytest.raises(NotImplementedError):
        sim.check_set_up()
        
def test_simple_atmos():
    state = (200, 0, -3, 20)
    time = 0.1
    density = simple_atmos(state=state, time=time, earth_radius=200, surf_density=1)
    assert density == 1

def test_atmos_model_set():
    models = get_available_atmos_models()
    sim = Simulator(SimConfig(atmosphere_model=next(iter(models))))
    sim.check_set_up()
    