import numpy as np
import pytest
from pathlib import Path

from deorbit.simulator import Simulator, generate_sim_config
from deorbit.utils.constants import EARTH_RADIUS
from deorbit.utils.dataio import formats, load_sim_data, load_sim_config


@pytest.mark.parametrize("format", formats.keys())
def test_save_simdata(tmpdir, format):
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
    sim.run(10)
    pre_save_data = sim.gather_data()
    pre_save_config = sim.export_config()
    
    test_save_name = "test_dir"
    save_path = sim.save_data(tmpdir / Path(test_save_name), format)
    
    assert save_path == tmpdir / Path(test_save_name)

    loaded_data = load_sim_data(save_path)
    loaded_config = load_sim_config(save_path)

    assert pre_save_data == loaded_data
    assert pre_save_config == loaded_config
    
    loaded_data_2 = load_sim_data(tmpdir / Path(test_save_name))
    loaded_config_2 = load_sim_config(tmpdir / Path(test_save_name))
    
    assert pre_save_data == loaded_data_2
    assert pre_save_config == loaded_config_2
