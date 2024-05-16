# Satellite Deorbit Simulator and Predictor Tutorial

This package is designed to simulate the trajectory of a deorbiting satellite around Earth and predict its impact location using radar station measurements and an Extended Kalman Filter.The package consists of 3 primary modules: <br>
The Simulator provides the functionality to allow for easy dynamic simulations of a deorbiting satellite.<br>
The Observer allows the user to configure and simulate measurements from this simulation using radar stations positioned on Earth.<br>
The Predictor uses an Extended Kalman Filter with these simulated measurements. It predicts the trajectory and final impact location of the satellite using the radar measurement data. 

The following document outlines the setup and usage guide for the package in order to reproduce the results in the report.

## Setup

### Installing Python

This package requires `Python=3.10`. 

To check your Python version with a Windows OS:
Open command prompt by typing 'cmd' into the search bar.
Type 'python' and press enter.
This should display the Python version currently installed. 
If it is not recognised, you can download Python from this source:<br>
https://www.python.org/downloads/

### Creating a New Conda Environment (RECOMMENDED)

Before installing this package it is recommended to create a new Conda environment (see "Installing Conda" below). Using a new environment before installing the required packages will ensure there are no conflicts with your current packages installed.

Open your command prompt and enter:
```
> conda create --name new_environment_name python=3.10 
```

It is recommended to name the environment "deorbit" or similar.

Activate the new environment in the prompt by entering:
```
> conda activate new_environment_name
```

and replace 'new_environment_name' with your previously chosen environment name, in this example it is "deorbit".
The prompt should now change from:
```
(base) C:\Users\User>
```

to:
```
(deorbit) C:\Users\User>
```

Which shows that the "deorbit" environment is activated.

If you encounter any issues, make sure to check your installation and path configuration of Conda. Alternatively, you can try entering the command into the Anaconda prompt by searching "Anaconda prompt" if you installed Anaconda Distribution.


### Installing Conda (RECOMMENDED)

If you do not have Conda installed it can be downloaded from: <br>
https://conda.io/projects/conda/en/latest/user-guide/install/windows.html<br>
Choose either Miniconda for a lightweight version or Anaconda Distribution.


### Installing the Deorbit Package

pip installing the Deorbit package allows the package to be imported into your Python files, enter:
```
> python -m pip install mir-satellite-deorbiter@git+https://github.com/ES98B-Mir-project-23/mir-orbiter.git@main
```

Into your command prompt or Anaconda prompt.

### Installing the Deorbit Package for Development Purposes (OPTIONAL)

To download the code for development purposes and have access to the code, you can clone the repository from GitHub to a local directory with the command line prompt:
```
> git clone https://github.com/ES98B-Mir-project-23/mir-orbiter.git
```

```
> cd mir-orbiter
```

to navigate to the mir-orbiter path

To install the required package dependencies, enter:
```
> python -m pip install -e .[dev]
```

This will install the package in editable mode, allowing the package to be modified and changes to be applied immediately in the local environment. 
<br>
After either of the above, the package will be available as 'deorbit' in the python environment and is ready to use. 

## Usage Guide

### Simulator

The Simulator is used to produce synthetic satellite deorbit data from initial conditions, until the point of impact. 

To reproduce the simulation performed in the report, create a new Python file, import the package together with Numpy and Matplotlib for complete functionality:

```
import deorbit
import numpy as np
import matplotlib.pyplot as plt
from deorbit.observer import Observer
from deorbit.utils.dataio import load_sim_data, load_sim_config
from deorbit.utils.plotting import plot_trajectories, plot_height
```

The following runs the simulation using RK4 integrator and the coesa atmos fast atmospheric model. The initial height of the satellite is 150000 metres and the simulation runs at 2 second time intervals. 

```
sim = deorbit.simulator.run(
        "RK4",
        "coesa_atmos_fast",
        initial_state=np.array((deorbit.constants.EARTH_RADIUS + 150000, 0, 0, 0, 0, 7820)),
        time_step=2,
    )
```

The simulation results can then be stored in the form of an array of states in sim_data. The simulation configuration is stored in sim_config. The results are then saved.

```
save_path = "eg/sim_example/"
sim_data = sim.gather_data()
sim_config = sim.export_config()
sim.save_data(save_path)
```

The results can then be loaded from the file using:

```
sim_data = load_sim_data(save_path)
sim_config = load_sim_config(save_path)
```

The trajectory, in this 3 dimensional example is accessed with the position vectors stored in the state array and can be used to visualise the trajectory or the height change through time:

```
traj = sim_data.state_array()[:, :3]

plot_trajectories(traj)
plot_height(traj, sim_data.times)
```

### Observation

The Observation module runs the simulation of radar measurements using the simulated data. The observer can be initialised with:
```
obs = Observer(number_of_radars=20, dim=3, radar_noise_factor = 0.5)
```
20 radars are used in 3 dimensions and the radar variance per meter is set to 0.5.
The configuration of radar stations is equally spaced around the Earth's surface and can be visualized with:
```
obs.plot_config()
```
Next, the position vectors and times are taken from the previously generated 'sim_data' to be converted into radar measurements:
```
sim_states = sim_data.state_array()
sim_times = sim_data.times
```
Now, the Observer can be run, in this example the radars check for line of sight every 100 time steps. The observed states and times are received and stored again as a trajectory to be used for plotting purposes:
```
obs.run(sim_states=sim_states, sim_times=sim_times, checking_interval=100)
obs_states = obs.observed_states
obs_times = obs.observed_times
traj = sim_states[:, :3]
```

The height over time can be plotted showing both the simulated and observed trajectories:
```
plot_height(traj, sim_times, observations=obs_states, observation_times = obs_times)
```

### Prediction

Now the observed states and times can be used by the Extended Kalman Filter (EKF) to generate a predicted trajectory and impact location:

First, the process and measurement covariance matrices Q and P are defined followed by the measurement matrix H and the time step:
```
Q = np.diag([0.1, 0.1, 0.01, 0.01])#process noise
P = np.eye(4) 
H = np.eye(4)
dt = sim_config.simulation_method_kwargs.time_step
```
Finally the EKF is initialised and run:
```
ekf = EKF()
estimated_traj, uncertainties, estimated_times = ekf.run(observations, dt=dt, Q=Q, R=observed_covariances, P=P, H=H)
```
The predicted trajectory can be plotted with the true trajectory to visualise the accuracy of the prediction:
```
true_traj = sim_data.state_array()[:, :2]
crash_coords = true_traj[-1, :]
plot_trajectories(true_traj, observations=observation_states, estimated_traj=estimated_traj)
```
To plot a close plot of the impact location:
```
plot_crash_site(true_traj, observations=observation_states, estimated_traj=estimated_traj)
```
To plot a close plot where the final crash is is outputd on a 2D map:
```
plot_crash_site_on_map(true_traj, estimated_traj=estimated_traj):
```

## Examples

For mroe detailed usage examples see [examples/](examples)

## API documentation

To build the documentation, you must first have installed the package's dev dependencies with

```
python -m pip install mir-satellite-deorbiter[dev]@git+https://github.com/ES98B-Mir-project-23/mir-orbiter.git@main
```

The following will build the documentation in [docs/build](docs/build/):

```
sphinx-build -M html docs/source/ docs/build/
