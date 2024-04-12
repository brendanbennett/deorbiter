import pytest
import numpy as np
from deorbit.utils.constants import EARTH_RADIUS
from deorbit.simulator import (
    Simulator,
    generate_sim_config,
)
from deorbit.utils.dataio import formats

@pytest.mark.parametrize("format", formats.keys())
def test_save_simdata(tmpdir, format):
    initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
    config = generate_sim_config("euler", "coesa_atmos_fast", initial_state)
    sim = Simulator(config)
    sim.run(10)
    sim.save_data(tmpdir, format)