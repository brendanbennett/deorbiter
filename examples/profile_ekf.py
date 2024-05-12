import cProfile
import pstats

import numpy as np

import deorbit
import deorbit.data_models
from deorbit.predictor.EKF import EKF
from deorbit.simulator.atmos import AtmosphereModel
from deorbit.utils.dataio import load_sim_config, load_sim_data

save_path = "examples/profiling/EKF_data"

sim_data = load_sim_data(save_path)
sim_config = load_sim_config(save_path)

if sim_data is None or sim_config is None:
    sim = deorbit.simulator.run(
        "adams_bashforth",
        "coesa_atmos_fast",
        initial_state=np.array((deorbit.constants.EARTH_RADIUS + 150000, 0, 0, 7820)),
        time_step=0.1,
    )
    sim_data = sim.gather_data()
    sim_config = sim.export_config()
    sim.save_data(save_path)
else:
    print("Loaded data from file")


# Define process and measurement noise covariance matrices, think this noise should be alot bigger
Q = np.diag([0.1, 0.1, 0.01, 0.01])
R = np.diag([1, 1, 0.1, 0.1])
P = np.diag([1, 1, 1, 1])

# Measurement matrix H (assuming all states are measured directly??????) -- for now
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

dt = sim_config.simulation_method_kwargs.time_step
atmos_config = sim_config.atmosphere_model_kwargs

atmos = AtmosphereModel(atmos_config)
# Shorten data for profiling
sim_data.x1 = sim_data.x1[:30000]
sim_data.x2 = sim_data.x2[:30000]
sim_data.v1 = sim_data.v1[:30000]
sim_data.v2 = sim_data.v2[:30000]
sim_data.times = sim_data.times[:30000]


def prof():
    estimated_traj, measurements = EKF(sim_data, sim_config, atmos, dt, Q, R, P, H)


cProfile.runctx("prof()", globals(), locals(), "examples/profiling/profile_ekf.prof")
s = pstats.Stats("examples/profiling/profile_ekf.prof")
s.strip_dirs().sort_stats("time").print_stats(50)
