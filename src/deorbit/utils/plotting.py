"""This module encapsulates various plotting methods for visualizing trajectories, errors, and other relevant data associated with the simulation and prediction of satellite trajectories.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import deorbit
from mpl_toolkits.basemap import Basemap

__all__ = ['plot_trajectories', 'plot_height', 'plot_crash_site', 'plot_error', 'plot_position_error', 'plot_velocity_error', 'plot_absolute_error', 'plot_theoretical_empirical_observation_error', 'plot_heatmap']

def plot_trajectories(true_traj, estimated_traj = None, observations = None, title="Trajectories"):
    """
    Plots 2D or 3D trajectories for true, estimated, and noisy measurement data,
    including a representation of Earth when relevant.

    Args:
        true_traj (np.ndarray): The true trajectory data points as a NumPy array.
        estimated_traj (np.ndarray, optional): The estimated trajectory data points. Defaults to None.
        observations (np.ndarray, optional): The noisy measurements of the trajectory. Defaults to None.
        title (str): The title of the plot.

    Returns:
        A matplotlib plot of the trajectories and measurements with Earth's representation if applicable.
    """
    if len(true_traj[0]) == 2:
        fig, ax = plt.subplots()
        ax.plot(true_traj[:, 0], true_traj[:, 1], label='True Trajectory')
        if observations is not None:
            ax.scatter(observations[:, 0], observations[:, 1], marker='x', color='r', label='Noisy Measurements')
        if estimated_traj is not None:
            ax.plot(estimated_traj[:, 0], estimated_traj[:, 1], label='Estimated Trajectory', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Position X')
        ax.set_ylabel('Position Y')
        earth = plt.Circle((0, 0), radius=deorbit.constants.EARTH_RADIUS, fill=False)
        ax.add_patch(earth)
        ax.legend()
        plt.show()
        plt.close()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], label='True Trajectory')
        if observations is not None:
            ax.scatter(observations[:, 0], observations[:, 1], observations[:, 2], marker='x', color='r', label='Noisy Measurements')
        if estimated_traj is not None:
            ax.plot(estimated_traj[:, 0], estimated_traj[:, 1], estimated_traj[:, 2], label='Estimated Trajectory', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Position X')
        ax.set_ylabel('Position Y')
        ax.set_zlabel('Position Z')

        #plotting EARTH
        r = deorbit.constants.EARTH_RADIUS
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        ax.plot_surface(x, y, z, color="g", alpha=0.5)

        ax.legend()
        plt.show()
        plt.close()

def plot_height(true_traj, times, estimated_traj = None, observations = None, observation_times = None, title = 'Height'):
    """
    Plots the height over time from Earth's surface for true, estimated, and noisy measurements.

    Args:
        true_traj (np.ndarray): True trajectory data points.
        times (np.ndarray): Timestamps corresponding to the true trajectory data points.
        estimated_traj (np.ndarray, optional): Estimated trajectory data points. Defaults to None.
        observations (np.ndarray, optional): Noisy measurement data points. Defaults to None.
        observation_times (np.ndarray, optional): Timestamps for the observations. Defaults to None.
        title (str): The title of the plot.

    Returns:
        A line plot representing the height from Earth's surface for each trajectory type over time.
    """
    fig, ax = plt.subplots()
    ax.plot(np.array(times) / 60, (np.linalg.norm(true_traj, axis=1) - deorbit.constants.EARTH_RADIUS)/1000, label = 'True Height')
    if observations is not None:
        ax.scatter(np.array(observation_times)/ 60, (np.linalg.norm(observations, axis = 1) -deorbit.constants.EARTH_RADIUS)/1000, marker='x', color='r', label='Noisy Measurements')
    if estimated_traj is not None:
        ax.plot(np.array(times)/ 60, (np.linalg.norm(estimated_traj, axis = 1) -deorbit.constants.EARTH_RADIUS)/1000, label='Estimated Trajectory', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Time (m)')
    ax.set_ylabel('Height')
    ax.legend()
    plt.show()
    plt.close()

def plot_crash_site(true_traj, estimated_traj = None, observations = None, title = 'Crash Site'):
    """
    Plots the final crash site in 2D, including trajectories and observations up to the crash point.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray, optional): The estimated trajectory data points. Defaults to None.
        observations (np.ndarray, optional): Noisy measurements of the trajectory. Defaults to None.
        title (str): The title of the plot.

    Returns:
        A 2D plot centered around the crash site with trajectory and observation data.
    """
    if len(true_traj[0]) == 2:
        crash_coords = true_traj[-1, :]
        fig, ax = plt.subplots()
        ax.plot(true_traj[:, 0], true_traj[:, 1], label='True Trajectory')
        if observations is not None:
            ax.scatter(observations[:, 0], observations[:, 1], marker='x', color='r', label='Noisy Measurements')
        if estimated_traj is not None:
            ax.plot(estimated_traj[:, 0], estimated_traj[:, 1], label='Estimated Trajectory', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Position X')
        ax.set_ylabel('Position Y')
        ax.set_xlim([crash_coords[0]-5e5, crash_coords[0]+5e5])
        ax.set_ylim([crash_coords[1]-5e5, crash_coords[1]+5e5])
        earth = plt.Circle((0, 0), radius=deorbit.constants.EARTH_RADIUS, fill=False)
        ax.add_patch(earth)
        ax.legend()
        plt.show()
        plt.close()
    else: 
        print('Crash Site Visualisation works in 2D only')


    
def plot_error(true_traj, estimated_traj, title="Error in Trajectories"):
    """
    Plots the norm of the error between the true and estimated trajectories over time.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray): The estimated trajectory data points.
        title (str): The title of the plot.

    Returns:
        A line plot showing the error between the true and estimated trajectories over time.
    """
    true_traj = np.array(true_traj)
    estimated_traj = np.array(estimated_traj)
    error = np.linalg.norm(true_traj - estimated_traj, axis=1)
    fig, ax = plt.subplots()
    ax.plot(error)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Error')
    plt.show()
 
def Three_Dim_Slice_Trajectory(true_traj, estimated_traj, observation_states, observation_times, sim_times, dt, Three_Dim_crash_coords):
    """
    Plots a 3D trajectory slice for a specified time duration around the first observation time, including a crash site marker.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray): The estimated trajectory data points.
        observation_states (np.ndarray): Observed states during the trajectory.
        observation_times (np.ndarray): Timestamps for each observation.
        sim_times (np.ndarray): Simulation timestamps.
        dt (float): Time step duration.
        Three_Dim_crash_coords (tuple): 3D coordinates for the crash site.

    Returns:
        A 3D plot of the true and estimated trajectory segments and observations within the specified duration.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_duration = 1000

    start_index = np.argmax(sim_times > observation_times[0])
    start_time = sim_times[start_index]
    end_time = start_time + plot_duration

    true_traj_slice = slice(start_index, int(end_time/dt))
    est_traj_slice = slice(0, int(plot_duration/dt))
    measure_slice = slice(0, np.argmax(observation_times > end_time))

    ax.plot(true_traj[:, 0][true_traj_slice], true_traj[:, 1][true_traj_slice], true_traj[:, 2][true_traj_slice], label='True Trajectory')
    ax.scatter(observation_states[:, 0][measure_slice], observation_states[:, 1][measure_slice], observation_states[:, 2][measure_slice], marker='x', color='r', label='Noisy Measurements')
    ax.plot(estimated_traj[:, 0][est_traj_slice], estimated_traj[:, 1][est_traj_slice], estimated_traj[:, 2][est_traj_slice], label='EKF Estimated Trajectory', linestyle='--')
    ax.scatter(Three_Dim_crash_coords[0], Three_Dim_crash_coords[1], Three_Dim_crash_coords[2], color='red', marker='X', label='Crash Site')
    ax.set_title('Extended Kalman Filter for Satellite Motion')
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Position Z')
    ax.legend()
    plt.show()

def plot_position_error(true_traj, estimated_traj, observation_states):
    """
    Plots the position error between the true and estimated trajectories in 2D or 3D.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray): The estimated trajectory data points.
        observation_states (np.ndarray): Observed states during the trajectory (not used in error calculation).

    Returns:
        A plot indicating position errors in either 2D or 3D depending on the trajectory data dimensionality.
    """
    # Calculate absolute error between true and estimated trajectories
    error = np.abs(true_traj - estimated_traj)

    # Check if the data is 2D or 3D
    is_3d = len(true_traj[0]) == 3

    # Plot position error
    if is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(error[:, 0], error[:, 1], error[:, 2], label='Position Error')
        ax.set_xlabel('Error in X')
        ax.set_ylabel('Error in Y')
        ax.set_zlabel('Error in Z')
        ax.set_title('Position Error')
        ax.legend()
        plt.show()
    else:
        plt.plot(error[:, 0], error[:, 1], label='Position Error')
        plt.xlabel('Error in X')
        plt.ylabel('Error in Y')
        plt.title('Position Error')
        plt.legend()
        plt.show()

def plot_velocity_error(true_traj, estimated_traj):
    """
    Plots the velocity error between the true and estimated trajectories in 2D or 3D.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray): The estimated trajectory data points.

    Returns:
        A plot indicating velocity errors in either 2D or 3D depending on the trajectory data dimensionality.
    """
    # Calculate velocity for true and estimated trajectories
    true_velocity = np.diff(true_traj, axis=0)
    estimated_velocity = np.diff(estimated_traj, axis=0)

    # Calculate absolute error between true and estimated velocities
    velocity_error = np.abs(true_velocity - estimated_velocity)

    # Check if the data is 2D or 3D
    is_3d = len(true_traj[0]) == 3

    # Plot velocity error
    if is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(velocity_error[:, 0], velocity_error[:, 1], velocity_error[:, 2], label='Velocity Error')
        ax.set_xlabel('Error in Velocity X')
        ax.set_ylabel('Error in Velocity Y')
        ax.set_zlabel('Error in Velocity Z')
        ax.set_title('Velocity Error')
        ax.legend()
        plt.show()
    else:
        plt.plot(velocity_error[:, 0], velocity_error[:, 1], label='Velocity Error')
        plt.xlabel('Error in Velocity X')
        plt.ylabel('Error in Velocity Y')
        plt.title('Velocity Error')
        plt.legend()
        plt.show()

def plot_velocity_error(true_traj, estimated_traj, title="Error in Velocity"):
    """
    Plots the velocity error over time for true and estimated trajectories.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray): The estimated trajectory data points.
        title (str): The title of the plot.

    Returns:
        A line plot showing the velocity error over time.
    """
    true_velocities = np.diff(true_traj, axis=0)
    estimated_velocities = np.diff(estimated_traj, axis=0)
    error = np.abs(true_velocities - estimated_velocities)
    fig, ax = plt.subplots()
    ax.plot(error)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Error')
    plt.show()

def plot_absolute_error(true_traj, estimated_traj):
    """
    Plots the absolute error between true and estimated trajectories for all coordinates.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray): The estimated trajectory data points.

    Returns:
        A plot of the absolute error vectors between the trajectories, in either 2D or 3D.
    """
    # Calculate absolute error between true and estimated trajectories
    error = np.abs(true_traj - estimated_traj)

    # Check if the data is 2D or 3D
    is_3d = len(true_traj[0]) == 3

    # Plot absolute error
    if is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(error[:, 0], error[:, 1], error[:, 2], label='Absolute Error')
        ax.set_xlabel('Absolute Error in X')
        ax.set_ylabel('Absolute Error in Y')
        ax.set_zlabel('Absolute Error in Z')
        ax.set_title('Absolute Error')
        ax.legend()
        plt.show()
    else:
        plt.plot(error[:, 0], error[:, 1], label='Absolute Error')
        plt.xlabel('Absolute Error in X')
        plt.ylabel('Absolute Error in Y')
        plt.title('Absolute Error')
        plt.legend()
        plt.show()
        
def plot_theoretical_empirical_observation_error(sim_states, sim_times, observation_states, observation_times, observed_covariances):
    """
    Plots both theoretical and empirical observation errors for position and velocity.

    Args:
        sim_states (np.ndarray): Simulated states over time.
        sim_times (np.ndarray): Timestamps for simulated states.
        observation_states (np.ndarray): Observed states.
        observation_times (np.ndarray): Timestamps for observed states.
        observed_covariances (np.ndarray): Covariance matrices for the observations.

    Returns:
        Two plots in a single figure: one for velocity error and one for position error, comparing empirical and theoretical values.
    """
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,8))
    sim_times_observed = np.array([i for i, t in enumerate(sim_times) if t in observation_times])
    vel_observation_error = np.linalg.norm((observation_states - sim_states[sim_times_observed])[:, 3:], axis=1)
    vel_std = np.sqrt(observed_covariances[:,3,3])
    ax1.plot(observation_times, vel_observation_error, label="Empirical")
    ax1.plot(observation_times, vel_std*np.sqrt(3), label="Theoretical")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Velocity error [m/s]")
    ax1.set_title("Velocity measurement error")
    ax1.legend()
    
    pos_observation_error = np.linalg.norm((observation_states - sim_states[sim_times_observed])[:, :3], axis=1)
    pos_std = np.sqrt(observed_covariances[:,0,0])
    ax2.plot(observation_times, pos_observation_error, label="Empirical")
    ax2.plot(observation_times, pos_std*np.sqrt(3), label="Theoretical")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position error [m]")
    ax2.set_title("Position measurement error")
    ax2.legend()
    
    fig.set_constrained_layout(True)
    plt.show()
    
def plot_heatmap(true_traj, estimated_traj, uncertainties):
    """
    Generates a heatmap of the impact location probability based on trajectory uncertainties.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray): The estimated trajectory data points.
        uncertainties (np.ndarray): Uncertainty matrices associated with each trajectory point.

    Returns:
        A heatmap over a geographic map displaying the probability density of the impact location.
    """
    crash_coords = true_traj[-1, :]
    dim = estimated_traj.shape[1] // 2
    mean_trajectory = estimated_traj[:, :dim]  # Extract position data
    
    # Define grid --> Projecting to 2D map?
    lats = np.linspace(-90, 90, 100)
    lons = np.linspace(-180, 180, 100)
    grid_lats, grid_lons = np.meshgrid(lats, lons)
    
    # Initialize heatmap array
    heatmap = np.zeros_like(grid_lats)
    
    # Iterate over each grid point
    for i in range(len(lats-1)):
        for j in range(len(lons-1)):
            # dist between mean trajectory and grid point
            dist = np.linalg.norm(mean_trajectory[:, :dim] - [lons[j], lats[i]], axis=1)
            dist = dist[:, np.newaxis]  # Column vector
            
            # Compute PDF based on uncertainty matrix
            try:
                inv_uncertainty_matrix = np.linalg.inv(uncertainties)  # Invert uncertainty matrix
                pdf = np.exp(-0.5 * np.sum(dist @ inv_uncertainty_matrix * dist, axis=1))
            except np.linalg.LinAlgError:
                pdf = np.zeros_like(dist.flatten())  # If matrix is singular, set PDF to zero I kept getting LinAlgError: Singular matrix so had to add this condition
            
            # Update heatmap 
            heatmap[i, j] = np.mean(pdf)

    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    m = Basemap(projection='mill', lon_0=0)
    m.drawcoastlines()
    m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1])
    x, y = m(grid_lons, grid_lats)
    m.pcolormesh(x, y, heatmap, cmap='hot_r')
    plt.colorbar(label='Probability Density')
    plt.title('Impact Location Heatmap')
    plt.show()
    
    return print("Heatmap shape:", heatmap.shape), print("Length of lats:", len(lats)), print("Length of lons:", len(lons)) # Debugging 
