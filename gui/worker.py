import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
import deorbit
from deorbit.predictor import EKF
# from deorbit.utils.dataio import load_sim_data, load_sim_config
from deorbit.observer import Observer

class WorkerThread(QThread):
    update_signal = pyqtSignal(str)
    plot_first_two_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    plot_last_two_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    plot_error_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)   
    error_signal = pyqtSignal(str)

    def __init__(self, method, atmos_model, dim, number_of_radars=10, radar_position_std_per_distance=0.005):
        super().__init__()
        self.method = method
        self.atmos_model = atmos_model
        self.dim = dim
        self.number_of_radars = number_of_radars
        self.radar_position_std_per_distance = radar_position_std_per_distance

    def run(self):
        try:
            np.random.seed(0)
            if self.dim == 2:
                self.update_signal.emit("Running 2D simulation...")
                sim = deorbit.simulator.run(
                    self.method,
                    self.atmos_model,
                    initial_state=np.array((deorbit.constants.EARTH_RADIUS + 150000, 0, 0, 7820)),
                    noise_types={"gaussian": {"noise_strength": 0.005}, "impulse": {"impulse_strength": 0.03, "impulse_probability": 1e-5}},
                    time_step=2,
                )
                self.update_signal.emit('2D simulation complete! ')
                sim_data = sim.gather_data()
                sim_config = sim.export_config()
                self.update_signal.emit('2D simulation data initialized... ')
            else:
                self.update_signal.emit("Running 3D simulation...")
                start_direction = np.random.normal(size=2)
                start_direction /= np.linalg.norm(start_direction)
                sim = deorbit.simulator.run(
                    self.method,
                    self.atmos_model,
                    initial_state=np.array((deorbit.constants.EARTH_RADIUS + 150000, 0, 0, 0, *(start_direction * 7820))),
                    noise_types={"gaussian": {"noise_strength": 0.001}, "impulse": {"impulse_strength": 0.005, "impulse_probability": 1e-5}},
                    time_step=2,
                )
                self.update_signal.emit('3D simulation complete!')
                sim_data = sim.gather_data()
                sim_config = sim.export_config()
                self.update_signal.emit('3D simulation data initialized... ')

            self.update_signal.emit('Running observer... ')
            obs = Observer(number_of_radars=self.number_of_radars, dim=self.dim, radar_position_std_per_distance=self.radar_position_std_per_distance)
            obs.run(sim_states=sim_data.state_array(), sim_times=sim_data.times, checking_interval=100)
            self.update_signal.emit('Observer complete! ')

            observation_times = np.array(obs.observed_times)
            observation_states = np.array(obs.observed_states)
            observed_covariances = np.array(obs.observed_covariances)

            self.update_signal.emit('Running EKF...')
            ekf = EKF(dim=self.dim)
            estimated_traj, uncertainties, estimated_times = ekf.run(
                observations=(observation_states, observation_times),
                dt=sim_config.simulation_method_kwargs.time_step,
                Q=np.diag([0.1] * self.dim + [0.01] * self.dim),
                R=observed_covariances,
                P=np.eye(sim_data.state_array().shape[1]),
                H=np.eye(sim_data.state_array().shape[1])
            )
            self.update_signal.emit('EKF complete!')

            self.plot_first_two_signal.emit(
                np.array(sim_data.state_array()[:, :self.dim]), 
                sim_data.state_array(),
                observation_states, 
                estimated_traj, 
                uncertainties, 
                observation_times, 
                np.array(estimated_times), 
                np.array(sim_data.times),
                observed_covariances
            )
            self.plot_last_two_signal.emit(
                np.array(sim_data.state_array()[:, :self.dim]), 
                sim_data.state_array(),
                observation_states, 
                estimated_traj, 
                uncertainties, 
                observation_times, 
                np.array(estimated_times), 
                np.array(sim_data.times),
                observed_covariances
            )

            self.plot_error_signal.emit(   
                sim_data.state_array(),
                np.array(sim_data.times),
                observation_states,
                observation_times,
                observed_covariances
            )

            self.update_signal.emit('Simulation and estimation complete!')
        except Exception as e:
            self.error_signal.emit(f'Error: {str(e)}')