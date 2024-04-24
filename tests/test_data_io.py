import numpy as np
import pytest

from deorbit.simulator import Simulator, generate_sim_config
from deorbit.utils.constants import EARTH_RADIUS
from deorbit.utils.dataio import formats, load_sim_data


@pytest.mark.parametrize("format", formats.keys())
def test_save_simdata(tmpdir, format):
    initial_state = np.array((EARTH_RADIUS + 100000, 0, 0, 8000))
    config = generate_sim_config("euler", "coesa_atmos_fast", initial_state)
    sim = Simulator(config)
    sim.run(10)
    pre_save_data = sim.gather_data()
    save_path = sim.save_data(tmpdir, format)

    loaded_data = load_sim_data(save_path)

    assert pre_save_data == loaded_data
