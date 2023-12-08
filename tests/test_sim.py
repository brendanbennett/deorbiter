import pytest
import sys
sys.path.append("../src/")
from ..src.simulator.simulator import Simulator
from ..src.data_models.sim import SimConfig

def test_atmos_model_not_set():
    sim = Simulator(SimConfig())
    with pytest.raises(NotImplementedError):
        sim.run()
    