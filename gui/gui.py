import numpy as np
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QComboBox, QTextEdit, QGroupBox, QMessageBox, QLineEdit
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from worker import WorkerThread
from deorbit.utils.plotting import plot_trajectories, plot_height, plot_crash_site, slice_by_time, plot_trajectories_on_map, plot_theoretical_empirical_observation_error

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
        options_group.setFixedWidth(250)
        options_layout.setSpacing(5)
        main_layout.addWidget(options_group)
        
        right_panel = QVBoxLayout()
        self.setup_controls(right_panel)
        main_layout.addLayout(right_panel)

        self.plot_layout = QGridLayout()
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(0)
        right_panel.addLayout(self.plot_layout)

    def setup_options(self, layout):
        self.predictor_combo = QComboBox()
        self.predictor_combo.addItems(['Extended Kalman Filter (EKF)'])
        self.predictor_combo.setToolTip("Choose the prediction method.")
        self.predictor_combo.setEnabled(False)
        layout.addWidget(QLabel("Predictor:"))
        layout.addWidget(self.predictor_combo)

        self.dim_combo = QComboBox()
        self.dim_combo.addItems(['2D', '3D'])
        self.dim_combo.setToolTip("Choose the simulation dimensionality.")
        layout.addWidget(QLabel("Dimensionality:"))
        layout.addWidget(self.dim_combo)

        self.atmos_model_combo = QComboBox()
        self.atmos_model_combo.addItems(['coesa_atmos_fast', 'zero_atmos', 'simple_atmos', 'icao_standard_atmos', 'coesa_atmos'])
        self.atmos_model_combo.setToolTip("Select the atmospheric model.")
        layout.addWidget(QLabel("Atmosphere Model:"))
        layout.addWidget(self.atmos_model_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems(['RK4', 'euler', 'adams_bashforth'])
        self.method_combo.setToolTip("Select the numerical integration method.")
        layout.addWidget(QLabel("Method:"))
        layout.addWidget(self.method_combo)

        self.number_of_radars_input = QLineEdit()
        self.number_of_radars_input.setText("10")
        self.number_of_radars_input.setToolTip("Enter the number of radars for observation.")
        layout.addWidget(QLabel("Number of Radars:"))
        layout.addWidget(self.number_of_radars_input)

        self.radar_position_std_input = QLineEdit()
        self.radar_position_std_input.setText("0.005")
        self.radar_position_std_input.setToolTip("Enter the radar position std per distance.")
        layout.addWidget(QLabel("Radar position std per distance:"))
        layout.addWidget(self.radar_position_std_input)

    def setup_controls(self, layout):
        button_layout = QHBoxLayout()

        self.start_button = QPushButton('Start Simulation')
        self.start_button.clicked.connect(self.start_simulation)
        self.start_button.setFixedHeight(30)
        self.start_button.setFixedWidth(300)
        button_layout.addWidget(self.start_button)
        button_layout.addStretch()

        self.clear_button = QPushButton('Clear Output')
        self.clear_button.clicked.connect(self.clear_output)
        self.clear_button.setFixedHeight(30)
        self.clear_button.setFixedWidth(300)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)

        centered_layout_first = QHBoxLayout()
        centered_layout_first.addStretch()

        self.plot_first_two_button = QPushButton('Trajectory and Height')
        self.plot_first_two_button.clicked.connect(self.plot_first_two)
        self.plot_first_two_button.setEnabled(False)
        self.plot_first_two_button.setFixedHeight(30)
        self.plot_first_two_button.setFixedWidth(300)
        centered_layout_first.addWidget(self.plot_first_two_button)

        centered_layout_first.addStretch()
        layout.addLayout(centered_layout_first)

        centered_layout_last = QHBoxLayout()
        centered_layout_last.addStretch()
        
        self.plot_last_two_button = QPushButton('Slice and Crash Site')
        self.plot_last_two_button.clicked.connect(self.plot_last_two)
        self.plot_last_two_button.setEnabled(False)
        self.plot_last_two_button.setFixedHeight(30)
        self.plot_last_two_button.setFixedWidth(300)
        centered_layout_last.addWidget(self.plot_last_two_button)

        centered_layout_last.addStretch()
        layout.addLayout(centered_layout_last)

        self.plot_error_button = QPushButton('Plot Errors (3D)')
        self.plot_error_button.clicked.connect(self.plot_errors)
        self.plot_error_button.setEnabled(False)
        self.plot_error_button.setFixedHeight(30)
        self.plot_error_button.setFixedWidth(300)
        centered_layout_last.addWidget(self.plot_error_button)

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
            radar_position_std_per_distance = float(self.radar_position_std_input.text())
            if number_of_radars <= 0:
                raise ValueError("Number of radars must be greater than 0.")
            if radar_position_std_per_distance <= 0:
                raise ValueError("Radar variance per meter must be greater than 0.")
        except ValueError as e:
            self.display_error(f" {str(e)}")
            return
    
        atmos_model = self.atmos_model_combo.currentText()
        predictor_method = self.method_combo.currentText()
        dim = 2 if self.dim_combo.currentText() == '2D' else 3

        self.thread = WorkerThread(predictor_method, 
                                    atmos_model,
                                    dim,
                                    number_of_radars, 
                                    radar_position_std_per_distance)
        
        self.thread.update_signal.connect(self.update_output)
        self.thread.plot_first_two_signal.connect(self.receive_first_two)
        self.thread.plot_last_two_signal.connect(self.receive_last_two)
        self.thread.plot_error_signal.connect(self.receive_error_data)
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

    def receive_first_two(self, true_traj, sim_states, observation_states, estimated_traj, uncertainties, observation_times, estimated_times, sim_times, observed_covariances):
        self.true_traj = true_traj
        self.sim_states = sim_states
        self.observation_states = observation_states
        self.estimated_traj = estimated_traj
        self.uncertainties = uncertainties
        self.observation_times = observation_times
        self.estimated_times = estimated_times
        self.sim_times = sim_times
        self.observed_covariances = observed_covariances
        self.plot_first_two_button.setEnabled(True)


    def receive_last_two(self, true_traj, sim_states, observation_states, estimated_traj, uncertainties, observation_times, estimated_times, sim_times, observed_covariances):
        self.true_traj = true_traj
        self.sim_states = sim_states
        self.observation_states = observation_states
        self.estimated_traj = estimated_traj
        self.uncertainties = uncertainties
        self.observation_times = observation_times
        self.estimated_times = estimated_times
        self.sim_times = sim_times
        self.observed_covariances = observed_covariances
        self.plot_last_two_button.setEnabled(True)
   

    def receive_error_data(self, sim_states, sim_times, observation_states, observation_times, observed_covariances):
        self.sim_states = sim_states
        self.sim_times = sim_times
        self.observation_states = observation_states
        self.observation_times = observation_times
        self.observed_covariances = observed_covariances

        if self.dim_combo.currentText() == '3D':
            self.plot_error_button.setEnabled(True)

    def plot_first_two(self):
        try:
            self.clear_plots()

            fig1 = Figure()
            canvas1 = FigureCanvas(fig1)
            toolbar1 = NavigationToolbar(canvas1, self)
            ax1 = fig1.add_subplot(111, projection='3d' if self.true_traj.shape[1] == 3 else 'rectilinear')
            plot_trajectories(self.true_traj, 
                              observations=self.observation_states, 
                              estimated_traj=self.estimated_traj, 
                              ax=ax1)
            self.plot_layout.addWidget(toolbar1, 0, 0)
            self.plot_layout.addWidget(canvas1, 1, 0)

            fig2 = Figure()
            canvas2 = FigureCanvas(fig2)
            toolbar2 = NavigationToolbar(canvas2, self)
            ax2 = fig2.add_subplot(111)
            plot_height(self.true_traj, 
                        observations=self.observation_states, 
                        estimated_traj=self.estimated_traj, 
                        observation_times=self.observation_times, 
                        estimated_times=self.estimated_times, 
                        times=self.sim_times, 
                        ax=ax2)
            self.plot_layout.addWidget(toolbar2, 0, 1)
            self.plot_layout.addWidget(canvas2, 1, 1)
        except Exception as e:
            self.display_error(str(e))

    def plot_last_two(self):
        try:
            self.clear_plots()

            if self.true_traj.shape[1] == 3:
                fig = Figure()
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, self)

                ax = fig.add_subplot(111)
                plot_trajectories_on_map(self.true_traj, 
                                         self.sim_times, 
                                         self.estimated_traj, 
                                         self.estimated_times, 
                                         uncertainties=self.uncertainties, 
                                         only_crash_sites=False, 
                                         ax=ax)
                ax.legend()
                self.plot_layout.addWidget(toolbar, 0, 0)
                self.plot_layout.addWidget(canvas, 1, 0)
            else:
                fig3 = Figure()
                canvas3 = FigureCanvas(fig3)
                toolbar3 = NavigationToolbar(canvas3, self)

                ax3 = fig3.add_subplot(111)
                start_time = self.observation_times[0]
                end_time = start_time + 1000

                true_traj_sliced, _ = slice_by_time(self.true_traj, self.sim_times, start_time, end_time)
                observation_states_sliced, _ = slice_by_time(self.observation_states, self.observation_times, start_time, end_time)
                estimated_traj_sliced, _ = slice_by_time(self.estimated_traj, self.estimated_times, start_time, end_time)

                plot_trajectories(true_traj_sliced, 
                                  observations=observation_states_sliced, 
                                  estimated_traj=estimated_traj_sliced, 
                                  ax=ax3, 
                                  show=False, 
                                  tight=True)

                self.plot_layout.addWidget(toolbar3, 0, 0)
                self.plot_layout.addWidget(canvas3, 1, 0)

                fig4 = Figure()
                canvas4 = FigureCanvas(fig4)
                toolbar4 = NavigationToolbar(canvas4, self)

                ax4 = fig4.add_subplot(111)
                plot_crash_site(self.true_traj, 
                                self.estimated_traj, 
                                self.observation_states, 
                                ax=ax4, 
                                show=False)

                self.plot_layout.addWidget(toolbar4, 0, 1)
                self.plot_layout.addWidget(canvas4, 1, 1)
        except Exception as e:
            self.display_error(str(e))

    def plot_errors(self):
        if self.dim_combo.currentText() == '3D':
            try:
                self.clear_plots()
                fig = Figure()
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, self)
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)

                plot_theoretical_empirical_observation_error(
                    self.sim_states,
                    self.sim_times,
                    self.observation_states,
                    self.observation_times,
                    self.observed_covariances,
                    ax1=ax1,
                    ax2=ax2,
                    show=False
                )

                self.plot_layout.addWidget(toolbar, 0, 0)
                self.plot_layout.addWidget(canvas, 1, 0)
            except Exception as e:
                self.display_error(str(e))
        else:
            self.display_error('This plot is only available for 3D simulations.')