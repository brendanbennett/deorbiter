"""
This module, EKF_Plotting, encapsulates various plotting methods for visualizing trajectories, errors, and other relevant data associated with the Extended Kalman Filter (EKF) applied to satellite motion.

Attributes:
    None

Methods:
    Two_Dim_plot_trajectories(true_traj, estimated_traj, observations, title="Trajectories"): Plots the true trajectory, estimated trajectory, and noisy measurements in 2D.
    Two_Dim_plot_error(true_traj, estimated_traj, title="Error in Trajectories"): Plots the error in trajectories in 2D.
    Three_Dim_plot_trajectories(true_traj, estimated_traj, observations, title="3D Trajectories"): Plots the true trajectory, estimated trajectory, and noisy measurements in 3D.
    Three_Dim_Slice_Trajectory(true_traj, estimated_traj, observation_states, observation_times, sim_times, dt, Three_Dim_crash_coords): Plots a sliced view of the true trajectory, estimated trajectory, and noisy measurements in 3D.
    plot_position_error(true_traj, estimated_traj, observation_states): Plots the position error.
    plot_velocity_error(true_traj, estimated_traj): Plots the velocity error.
    plot_absolute_error(true_traj, estimated_traj): Plots the absolute error. 
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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
        ax.legend()
        plt.show()
        plt.close()
    
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