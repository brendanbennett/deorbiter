"""This module encapsulates various plotting methods for visualizing trajectories, errors, and other relevant data associated with the simulation and prediction of satellite trajectories.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import deorbit
from mpl_toolkits.basemap import Basemap

def plot_trajectories(true_traj, estimated_traj = None, observations = None, title="Trajectories"):
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

def plot_from_last_measurements(true_traj, estimated_traj = None, observations = None, observation_times = None, no_measurements = 1, title = 'Plot from last Measurements'):
    last_measurements = observations[-no_measurements:, :]
    last_traj = true_traj[-no_measurements:, :]
    last_estimated_traj = estimated_traj[-no_measurements:, :]
    plot_trajectories(last_traj, estimated_traj = last_estimated_traj, observations=last_measurements)

    
    
def plot_error(true_traj, estimated_traj, title="Error in Trajectories"):
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
    crash_coords = true_traj[-1, :]
    dim = estimated_traj.shape[1] // 2
    mean_trajectory = estimated_traj[:, :dim]  # Extract position data
    uncertainty_matrix = uncertainties[:, :dim, :dim]  # Extract position uncertainty

    # Define grid --> Projecting to 2D map?
    lats = np.linspace(-90, 90, 100)
    lons = np.linspace(-180, 180, 200)
    grid_lats, grid_lons = np.meshgrid(lats, lons)

    # Initialize heatmap array
    heatmap = np.zeros_like(grid_lats)

    # Iterate over each grid point
    for i in range(len(lats)):
        for j in range(len(lons)):
            # Compute distance between mean trajectory and grid point
            dist = np.linalg.norm(mean_trajectory[:, :dim] - [lons[j], lats[i]], axis=1)

            # Compute PDF based on uncertainty matrix
            pdf = np.exp(-0.5 * np.sum(dist @ np.linalg.inv(uncertainty_matrix) * dist, axis=1))

            # Update heatmap value
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
