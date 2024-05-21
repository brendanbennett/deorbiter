import numpy as np
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QComboBox, QTextEdit, QGroupBox, QMessageBox, QLineEdit
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from worker import WorkerThread
from deorbit.utils.plotting import plot_trajectories, plot_heatmap_gui, scatter_on_map, plot_height, plot_crash_site, slice_by_time, plot_trajectories_on_map, plot_theoretical_empirical_observation_error

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
        self.method_combo.addItems(['RK4', 'Euler', 'Adams-Bashforth'])
        self.method_combo.setToolTip("Select the numerical integration method.")
        layout.addWidget(QLabel("Method:"))
        layout.addWidget(self.method_combo)

        self.number_of_radars_input = QLineEdit()
        self.number_of_radars_input.setText("10")
        self.number_of_radars_input.setToolTip("Enter the number of radars for observation.")
        layout.addWidget(QLabel("Number of Radars:"))
        layout.addWidget(self.number_of_radars_input)

        self.radar_position_std_per_position = QLineEdit()
        self.radar_position_std_per_position.setText("5e-3")
        self.radar_position_std_per_position.setToolTip("Enter the radar position std per distance.")
        layout.addWidget(QLabel("Radar position std per distance(sigma_r):"))
        layout.addWidget(self.radar_position_std_per_position)

        self.radar_velocity_std__per_speed = QLineEdit()
        self.radar_velocity_std__per_speed.setText("5e-4")
        self.radar_velocity_std__per_speed.setToolTip("Enter the radar velocity std per meter.")
        layout.addWidget(QLabel("Radar velocity std per meter (sigma_vs):"))
        layout.addWidget(self.radar_velocity_std__per_speed)

        self.radar_velocity_std_per_distance = QLineEdit()
        self.radar_velocity_std_per_distance.setText("1e-6")
        self.radar_velocity_std_per_distance.setToolTip("Enter the radar velocity std per meter.")
        layout.addWidget(QLabel("Radar velocity std per meter (simage_vd):"))
        layout.addWidget(self.radar_velocity_std_per_distance)


    def setup_controls(self, layout):
        row1_layout = QHBoxLayout()

        self.start_button = QPushButton('Start Simulation')
        self.start_button.clicked.connect(self.start_simulation)
        self.start_button.setFixedHeight(30)
        row1_layout.addWidget(self.start_button, stretch=1)

        self.clear_button = QPushButton('Clear Output')
        self.clear_button.clicked.connect(self.clear_output)
        self.clear_button.setFixedHeight(30)
        row1_layout.addWidget(self.clear_button, stretch=1)
        
        layout.addLayout(row1_layout)

        row2_layout = QHBoxLayout()
        self.plot_first_two_button = QPushButton('Trajectory and Height')
        self.plot_first_two_button.clicked.connect(self.plot_first_two)
        self.plot_first_two_button.setEnabled(False)
        self.plot_first_two_button.setFixedHeight(30)
        row2_layout.addWidget(self.plot_first_two_button, stretch=1)

        self.plot_last_two_button = QPushButton('Slice and Crash Site')
        self.plot_last_two_button.clicked.connect(self.plot_last_two)
        self.plot_last_two_button.setEnabled(False)
        self.plot_last_two_button.setFixedHeight(30)
        row2_layout.addWidget(self.plot_last_two_button, stretch=1)
        
        layout.addLayout(row2_layout)

        row3_layout = QHBoxLayout()
        self.plot_error_button = QPushButton('Plot Errors')
        self.plot_error_button.clicked.connect(self.plot_errors)
        self.plot_error_button.setEnabled(False)
        self.plot_error_button.setFixedHeight(30)
        row3_layout.addWidget(self.plot_error_button, stretch=1)

        self.plot_heatmap_button = QPushButton('Heatmaps')
        self.plot_heatmap_button.clicked.connect(self.plot_heatmap)
        self.plot_heatmap_button.setEnabled(False)
        self.plot_heatmap_button.setFixedHeight(30)
        row3_layout.addWidget(self.plot_heatmap_button, stretch=1)
        
        layout.addLayout(row3_layout)

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
            radar_position_std_per_distance = float(self.radar_position_std_per_position.text())
            radar_velocity_std_per_speed = float(self.radar_velocity_std__per_speed.text())
            radar_velocity_std_per_distance = float(self.radar_velocity_std_per_distance.text())

            if number_of_radars <= 0:
                raise ValueError("Number of radars must be greater than 0.")
            if radar_position_std_per_distance <= 0:
                raise ValueError("Radar standard deviation of radar position per meter must be greater than 0.")
            if radar_velocity_std_per_speed <=0:
                raise ValueError("Radar standard deviation of radar velocity per speed must be greater than 0.")
            if radar_velocity_std_per_distance <=0:
                raise ValueError("Radar standard deviation of radar velocity per meter must be greater than 0.")
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
                                    radar_position_std_per_distance,
                                    radar_velocity_std_per_speed,
                                    radar_velocity_std_per_distance
                                    )
        
        self.thread.update_signal.connect(self.update_output)
        self.thread.plot_first_two_signal.connect(self.receive_first_two)
        self.thread.plot_last_two_signal.connect(self.receive_last_two)
        self.thread.plot_error_signal.connect(self.receive_error_data)
        self.thread.plot_heatmap_signal.connect(self.receive_heatmap_data)
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
        # if self.dim_combo.currentText() == '3D':
        self.plot_error_button.setEnabled(True)

    def receive_heatmap_data(self, sim_states, sim_times, estimated_traj, estimated_times, observation_to_check, observation_times, uncertainties):
        self.sim_states = sim_states
        self.sim_times = sim_times
        self.estimated_traj = estimated_traj 
        self.estimated_times = estimated_times 
        self.observation_to_check = observation_to_check
        self.observation_times = observation_times 
        self.uncertainties = uncertainties
        if self.dim_combo.currentText() == '3D':
            self.plot_heatmap_button.setEnabled(True)

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
                fig.tight_layout()
                self.plot_layout.addWidget(toolbar, 0, 0)
                self.plot_layout.addWidget(canvas, 1, 0)
            except Exception as e:
                self.display_error(str(e))
        elif self.dim_combo.currentText() == '2D':
            try:
                self.clear_plots()
                fig = Figure(figsize=(9, 6))
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, self)

                total_std = np.sqrt(np.trace(self.uncertainties, axis1=1, axis2=2))

                ax1, ax2 = fig.subplots(2, 1)
                ax1.semilogy(self.estimated_times, total_std)
                ax2.semilogy(self.estimated_times[:-400], total_std[:-400])

                ax1.set_title("Total standard deviation of state estimates")
                ax2.set_xlabel("Time (s)")
                ax1.set_ylabel("Total standard deviation")
                ax2.set_ylabel("Total standard deviation")

                fig.tight_layout()
                self.plot_layout.addWidget(toolbar, 0, 0)
                self.plot_layout.addWidget(canvas, 1, 0)
                canvas.draw()
            except Exception as e:
                self.display_error(str(e))
        else:
            self.display_error('This plot is only available for 3D simulations.')

    def plot_heatmap(self):
        if self.dim_combo.currentText() == '3D':
            try:
                self.clear_plots()  

                fig = Figure()
                canvas = FigureCanvas(fig)
                toolbar = NavigationToolbar(canvas, self)
                self.plot_layout.addWidget(toolbar, 0, 0)
                self.plot_layout.addWidget(canvas, 1, 0)

                crashes, crash_times, mean_crash, mean_crash_time = plot_heatmap_gui(
                    self.sim_states,
                    self.sim_times,
                    self.estimated_traj,
                    self.estimated_times,
                    self.observation_to_check,
                    self.observation_times,
                    self.uncertainties,
                    plot_mean=True
                )

                ax = fig.add_subplot(111)
                scatter_on_map(
                    crashes[0],  
                    crash_times[0],  
                    0.3,
                    "r",
                    20,
                    "x",
                    "Predicted Crash Sites",
                    title="Crash site heatmap",
                    ax=ax,
                    show=False,  
                    draw_lines=True
                )

                scatter_on_map(
                    [self.sim_states[-1][:3]],
                    [self.sim_times[-1]],
                    1,
                    "b",
                    60,
                    "x",
                    "True Crash Site",
                    ax=ax,
                    show=False, 
                    draw_lines=False
                )

                scatter_on_map(
                    [mean_crash],
                    [mean_crash_time],
                    1,
                    "g",
                    60,
                    "x",
                    "Mean Crash Site",
                    ax=ax,
                    show=False,  
                    draw_lines=False
                )

                ax.legend()
                canvas.draw() 

                total_std = np.mean(np.std(crashes[0], axis=0))
                final_error = np.linalg.norm(mean_crash - self.sim_states[-1, :3])
                self.output_text.append(f"Total standard deviation: {total_std}")
                self.output_text.append(f"Final error: {final_error}")

            except Exception as e:
                self.display_error(str(e))
        else:
            self.display_error('This plot is only available for 3D simulations.')