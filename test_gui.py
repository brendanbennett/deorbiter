import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QComboBox, QTextEdit, QGroupBox, QMessageBox, QLineEdit
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

import deorbit
from deorbit.predictor import EKF
from deorbit.utils.dataio import load_sim_data, load_sim_config
from deorbit.observer import Observer
from deorbit.utils.plotting import plot_trajectories, plot_height, plot_crash_site, slice_by_time

class WorkerThread(QThread):
    update_signal = pyqtSignal(str)
    plot_first_two_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    plot_last_two_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    
    error_signal = pyqtSignal(str)

    def __init__(self, save_path, method, atmos_model, number_of_radars=10, radar_variance_per_m=1):
        super().__init__()
        self.save_path = save_path
        self.method = method
        self.atmos_model = atmos_model
        self.number_of_radars = number_of_radars
        self.radar_variance_per_m = radar_variance_per_m

    def run(self):
        try:
            sim_data = load_sim_data(self.save_path)
            sim_config = load_sim_config(self.save_path)
            if sim_data is None or sim_config is None:
                np.random.seed(0)
                sim = deorbit.simulator.run(
                    self.method,
                    self.atmos_model,
                    initial_state=np.array((deorbit.constants.EARTH_RADIUS + 150000, 0, 0, 7820)),
                    noise_types={"gaussian": {"noise_strength": 0.005}, "impulse": {"impulse_strength": 0.03, "impulse_probability": 1e-5}},
                    time_step=2,
                )
                sim_data = sim.gather_control_data()
                sim_config = sim.export_config()
                sim.save_data(self.save_path)
                self.update_signal.emit('Data initialized and saved.')
            else:
                self.update_signal.emit('Loaded data from file.')

            obs = Observer(number_of_radars=self.number_of_radars, radar_variance_per_m=self.radar_variance_per_m)
            obs.run(sim_states=sim_data.state_array(), sim_times=sim_data.times, checking_interval=100)
            
            observation_times = np.array(obs.observed_times)
            observation_states = np.array(obs.observed_states)
            observed_covariances = np.array(obs.observed_covariances)

            ekf = EKF()
            estimated_traj, uncertainties, estimated_times = ekf.run(
                observations=(observation_states, observation_times),
                dt=sim_config.simulation_method_kwargs.time_step,
                Q=np.diag([0.1, 0.1, 0.01, 0.01]),
                R=observed_covariances,
                P=np.eye(4),
                H=np.eye(4)
            )

            self.plot_first_two_signal.emit(np.array(sim_data.state_array()[:, :2]), observation_states, estimated_traj, observation_times, np.array(estimated_times), np.array(sim_data.times))
            self.plot_last_two_signal.emit(np.array(sim_data.state_array()[:, :2]), observation_states, estimated_traj, observation_times, np.array(estimated_times), np.array(sim_data.times))
            self.update_signal.emit('Simulation and estimation complete.')
        except Exception as e:
            self.error_signal.emit(str(e))


class SatelliteSimulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Satellite Decay Simulation')
        self.setGeometry(420, 180, 800, 600)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        self.setup_options(options_layout)
        options_group.setFixedWidth(250)  # Set fixed width for options panel
        options_layout.setSpacing(5)  # Reduce spacing between widgets
        main_layout.addWidget(options_group)
        
        right_panel = QVBoxLayout()
        self.setup_controls(right_panel)
        main_layout.addLayout(right_panel)

        self.plot_layout = QGridLayout()
        right_panel.addLayout(self.plot_layout)

    def setup_options(self, layout):
        self.predictor_combo = QComboBox()
        self.predictor_combo.addItems(['Extended Kalman Filter (EKF)'])
        self.predictor_combo.setToolTip("Choose the prediction method.")
        layout.addWidget(QLabel("Predictor:"))
        layout.addWidget(self.predictor_combo)

        self.atmos_model_combo = QComboBox()
        self.atmos_model_combo.addItems(['zero_atmos', 'simple_atmos', 'icao_standard_atmos', 'coesa_atmos', 'coesa_atmos_fast'])
        self.atmos_model_combo.setToolTip("Select the atmospheric model.")
        layout.addWidget(QLabel("Atmosphere Model:"))
        layout.addWidget(self.atmos_model_combo)

        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(['Gaussian noise', 'Impulse noise'])
        self.noise_type_combo.setToolTip("Choose the type of noise to apply.")
        layout.addWidget(QLabel("Noise Type:"))
        layout.addWidget(self.noise_type_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems(['RK4', 'Euler', 'Adams-Bashforth'])
        self.method_combo.setToolTip("Select the numerical integration method.")
        layout.addWidget(QLabel("Method:"))
        layout.addWidget(self.method_combo)

        self.number_of_radars_input = QLineEdit()
        self.number_of_radars_input.setText("10")
        self.number_of_radars_input.setToolTip("Enter the number of radars for observation.")
        layout.addWidget(QLabel("Number of Radars:"))
        layout.addWidget(self.number_of_radars_input)

        self.radar_variance_input = QLineEdit()
        self.radar_variance_input.setText("1")
        self.radar_variance_input.setToolTip("Enter the radar variance per meter.")
        layout.addWidget(QLabel("Radar Variance per m:"))
        layout.addWidget(self.radar_variance_input)

    def setup_controls(self, layout):
        self.start_button = QPushButton('Start Simulation')
        self.start_button.clicked.connect(self.start_simulation)
        layout.addWidget(self.start_button)

        self.plot_first_two_button = QPushButton('Trajectory and Height')
        self.plot_first_two_button.clicked.connect(self.plot_first_two)
        self.plot_first_two_button.setEnabled(False)
        layout.addWidget(self.plot_first_two_button)

        self.plot_last_two_button = QPushButton('Slice and Crash Site')
        self.plot_last_two_button.clicked.connect(self.plot_last_two)
        self.plot_last_two_button.setEnabled(False)
        layout.addWidget(self.plot_last_two_button)
        


        self.clear_button = QPushButton('Clear Output')
        self.clear_button.clicked.connect(self.clear_output)
        layout.addWidget(self.clear_button)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        self.plot_display = QLabel()
        layout.addWidget(self.plot_display)
        self.error_display = QLabel()
        layout.addWidget(self.error_display)

    def start_simulation(self):
        try:
            number_of_radars = int(self.number_of_radars_input.text())
            radar_variance_per_m = float(self.radar_variance_input.text())
            if number_of_radars <= 0:
                raise ValueError("Number of radars must be greater than 0.")
            if radar_variance_per_m <= 0:
                raise ValueError("Radar variance per meter must be greater than 0.")
        except ValueError as e:
            self.display_error(f" {str(e)}")
            return
    
        atmos_model = self.atmos_model_combo.currentText()
        predictor_method = self.method_combo.currentText()

        self.thread = WorkerThread("eg/EKF_example_noise_2s/", 
                                    predictor_method, 
                                    atmos_model,
                                    number_of_radars, 
                                    radar_variance_per_m)
        
        self.thread.update_signal.connect(self.update_output)
        #self.thread.plot_signal.connect(self.display_plot)
        self.thread.plot_first_two_signal.connect(self.receive_first_two)
        self.thread.plot_last_two_signal.connect(self.receive_last_two)
        self.thread.error_signal.connect(self.display_error)
        self.thread.start()

    def update_output(self, message):
        self.output_text.append(message)

    def display_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        self.output_text.append(f"Error: {str(error_message)}")

    def clear_output(self):
        self.output_text.clear()
        self.plot_display.clear()
        self.error_display.clear()
        self.clear_plots()

    def clear_plots(self):
        for i in reversed(range(self.plot_layout.count())):
            widget_to_remove = self.plot_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.setParent(None)

    def receive_first_two(self, true_traj, observed_traj, estimated_traj, observation_times, estimated_times, sim_times):
        self.true_traj = true_traj
        self.observed_traj = observed_traj
        self.estimated_traj = estimated_traj
        self.observation_times = observation_times
        self.estimated_times = estimated_times
        self.sim_times = sim_times
        self.plot_first_two_button.setEnabled(True)

    def receive_last_two(self, true_traj, observed_traj, estimated_traj, observation_times, estimated_times, sim_times):
        self.true_traj = true_traj
        self.observed_traj = observed_traj
        self.estimated_traj = estimated_traj
        self.observation_times = observation_times
        self.estimated_times = estimated_times
        self.sim_times = sim_times
        self.plot_last_two_button.setEnabled(True)

    def plot_first_two(self):
        self.clear_plots()

        fig1 = Figure()
        canvas1 = FigureCanvas(fig1)
        toolbar1 = NavigationToolbar(canvas1, self)
        ax1 = fig1.add_subplot(111)
        plot_trajectories(self.true_traj, observations=self.observed_traj, estimated_traj=self.estimated_traj, ax=ax1)
        self.plot_layout.addWidget(toolbar1, 0, 0)
        self.plot_layout.addWidget(canvas1, 1, 0)

        fig2 = Figure()
        canvas2 = FigureCanvas(fig2)
        toolbar2 = NavigationToolbar(canvas2, self)
        ax2 = fig2.add_subplot(111)
        plot_height(self.true_traj, observations=self.observed_traj, estimated_traj=self.estimated_traj, observation_times=self.observation_times, estimated_times=self.estimated_times, times=self.sim_times, ax=ax2)
        self.plot_layout.addWidget(toolbar2, 0, 1)
        self.plot_layout.addWidget(canvas2, 1, 1)

    def plot_last_two(self):
        self.clear_plots()

        fig3 = Figure()
        canvas3 = FigureCanvas(fig3)
        toolbar3 = NavigationToolbar(canvas3, self)

        ax3 = fig3.add_subplot(111)
        start_time = self.observation_times[0]
        end_time = start_time + 1000

        true_traj_sliced, _ = slice_by_time(self.true_traj, self.sim_times, start_time, end_time)
        observation_states_sliced, _ = slice_by_time(self.observed_traj, self.observation_times, start_time, end_time)
        estimated_traj_sliced, _ = slice_by_time(self.estimated_traj, self.estimated_times, start_time, end_time)

        plot_trajectories(true_traj_sliced, observations=observation_states_sliced, estimated_traj=estimated_traj_sliced, ax=ax3, show=False, tight=True)

        self.plot_layout.addWidget(toolbar3, 0, 0)
        self.plot_layout.addWidget(canvas3, 1, 0)

        fig4 = Figure()
        canvas4 = FigureCanvas(fig4)
        toolbar4 = NavigationToolbar(canvas4, self)

        ax4 = fig4.add_subplot(111)
        plot_crash_site(self.true_traj, self.estimated_traj, self.observed_traj, ax=ax4, show=False)

        self.plot_layout.addWidget(toolbar4, 0, 1)
        self.plot_layout.addWidget(canvas4, 1, 1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SatelliteSimulatorGUI()
    ex.show()
    sys.exit(app.exec())
