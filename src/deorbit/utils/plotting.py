"""This module encapsulates various plotting methods for visualizing trajectories, errors, and other relevant data associated with the simulation and prediction of satellite trajectories.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import deorbit
from mpl_toolkits.basemap import Basemap
from deorbit.utils.coords import latlong_from_cart, earth_rotation
from matplotlib.patches import Ellipse

__all__ = [
    "plot_trajectories",
    "plot_height",
    "plot_crash_site",
    "plot_crash_site_on_map" "plot_error",
    "plot_position_error",
    "plot_velocity_error",
    "plot_absolute_error",
    "plot_theoretical_empirical_observation_error",
    "slice_by_time",
]


def plot_trajectories(
    true_traj,
    estimated_traj=None,
    observations=None,
    title="Trajectories",
    ax=None,
    show=True,
    tight=False,
):
    """
    Plots 2D or 3D trajectories for true, estimated, and noisy measurement data,
    including a representation of Earth when relevant.

    Args:
        true_traj (np.ndarray): The true trajectory data points as a NumPy array.
        estimated_traj (np.ndarray, optional): The estimated trajectory data points. Defaults to None.
        observations (np.ndarray, optional): The noisy measurements of the trajectory. Defaults to None.
        title (str): The title of the plot.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
    """
    if len(true_traj[0]) == 2:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(true_traj[:, 0], true_traj[:, 1], label="True Trajectory", alpha=0.7)
        ax.set_aspect("equal")
        if observations is not None:
            ax.scatter(
                observations[:, 0],
                observations[:, 1],
                marker="x",
                color="r",
                label="Noisy Measurements",
                alpha=0.7,
                s=10,
            )
        if estimated_traj is not None:
            ax.plot(
                estimated_traj[:, 0],
                estimated_traj[:, 1],
                label="Estimated Trajectory",
                linestyle="--",
            )
        ax.set_title(title)
        ax.set_xlabel("Position X")
        ax.set_ylabel("Position Y")
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        earth = plt.Circle((0, 0), radius=deorbit.constants.EARTH_RADIUS, fill=False)
        ax.add_patch(earth)
        if tight:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax.legend()
    else:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            true_traj[:, 0],
            true_traj[:, 1],
            true_traj[:, 2],
            label="True Trajectory",
            alpha=0.7,
        )
        if observations is not None:
            ax.scatter(
                observations[:, 0],
                observations[:, 1],
                observations[:, 2],
                marker="x",
                color="r",
                label="Noisy Measurements",
                s=10,
                alpha=0.7,
            )
        if estimated_traj is not None:
            ax.plot(
                estimated_traj[:, 0],
                estimated_traj[:, 1],
                estimated_traj[:, 2],
                label="Estimated Trajectory",
                linestyle="--",
            )
        ax.set_title(title)
        ax.set_xlabel("Position X")
        ax.set_ylabel("Position Y")
        ax.set_zlabel("Position Z")
        # ax.set_aspect('equal')

        # plotting EARTH
        r = deorbit.constants.EARTH_RADIUS
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        ax.plot_surface(x, y, z, color="g", alpha=0.5)

        ax.legend()
    if show:
        plt.show()
        plt.close()


def plot_height(
    true_traj,
    times,
    estimated_traj=None,
    observations=None,
    observation_times=None,
    estimated_times=None,
    title="Height",
    ax=None,
    show=True,
) -> None:
    """
    Plots the height over time from Earth's surface for true, estimated, and noisy measurements.

    Args:
        true_traj (np.ndarray): True trajectory data points.
        times (np.ndarray): Timestamps corresponding to the true trajectory data points.
        estimated_traj (np.ndarray, optional): Estimated trajectory data points. Defaults to None.
        observations (np.ndarray, optional): Noisy measurement data points. Defaults to None.
        observation_times (np.ndarray, optional): Timestamps for the observations. Defaults to None.
        estimated_times (np.ndarry, optional): Timestamps from the estimator. Defaults to None
        title (str): The title of the plot.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(
        np.array(times) / 60,
        (np.linalg.norm(true_traj, axis=1) - deorbit.constants.EARTH_RADIUS) / 1000,
        label="True Height",
        alpha=0.9,
        color="darkblue",
    )
    if observations is not None:
        ax.scatter(
            np.array(observation_times) / 60,
            (np.linalg.norm(observations, axis=1) - deorbit.constants.EARTH_RADIUS)
            / 1000,
            marker="x",
            color="r",
            label="Noisy Measurements",
            alpha=0.7,
            s=10,
        )
    if estimated_traj is not None:
        ax.plot(
            np.array(estimated_times) / 60,
            (np.linalg.norm(estimated_traj, axis=1) - deorbit.constants.EARTH_RADIUS)
            / 1000,
            label="Estimated Trajectory",
            linestyle="--",
            color="k",
            lw=2,
        )
    ax.set_title(title)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Height (km)")
    ax.legend()
    if show:
        plt.show()
        plt.close()


def plot_crash_site(
    true_traj,
    estimated_traj=None,
    observations=None,
    title="Crash Site",
    ax=None,
    show=True,
) -> None:
    """
    Plots the final crash site in 2D, including trajectories and observations up to the crash point.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray, optional): The estimated trajectory data points. Defaults to None.
        observations (np.ndarray, optional): Noisy measurements of the trajectory. Defaults to None.
        title (str): The title of the plot.
    """
    if len(true_traj[0]) == 2:
        crash_coords = true_traj[-1, :]
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(true_traj[:, 0], true_traj[:, 1], label="True Trajectory")
        if observations is not None:
            ax.scatter(
                observations[:, 0],
                observations[:, 1],
                marker="x",
                color="r",
                label="Noisy Measurements",
            )
        if estimated_traj is not None:
            ax.plot(
                estimated_traj[:, 0],
                estimated_traj[:, 1],
                label="Estimated Trajectory",
                linestyle="--",
            )
        ax.set_title(title)
        ax.set_xlabel("Position X")
        ax.set_ylabel("Position Y")
        ax.set_xlim([crash_coords[0] - 5e5, crash_coords[0] + 5e5])
        ax.set_ylim([crash_coords[1] - 5e5, crash_coords[1] + 5e5])
        earth = plt.Circle((0, 0), radius=deorbit.constants.EARTH_RADIUS, fill=False)
        ax.add_patch(earth)
        ax.legend()
    else:
        print("Crash Site Visualisation works in 2D only")
    if show:
        plt.show()
        plt.close()


def plot_crash_site_on_map(
    true_traj,
    true_crash_time=0.0,
    estimated_traj=None,
    estimated_crash_time=0.0,
    uncertainties=None,
    title="Crash Site",
    ax=None,
    show=True,
):
    """
    Plots the final crash site on a 2D map, including the estaimated crash site if it is supplied

    Args:
    true_traj (np.ndarray): The true trajectory data points.
    estimated_traj (np.ndarray, optional): The estimated trajectory data points. Defaults to None.
    uncertainties (np.ndarray): covariance arrays associated with trajectory data points. Defaults to None
    title (str): The title of the plot.
    """
    crash_coords = true_traj[-1, :]
    dim = len(crash_coords)

    if ax is None:
        m = Basemap(projection="cyl", lon_0=0)

    # Draw coastlines, parallels, and meridians
    m.drawcoastlines()
    m.drawparallels(np.arange(-90.0, 91.0, 30.0), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180.0, 181.0, 60.0), labels=[0, 0, 0, 1])

    if dim == 2:
        long_crash_coords = latlong_from_cart(crash_coords) / (np.pi / 180)

        # normalize longitude coordinate
        normalized_long = long_crash_coords % 360
        if normalized_long > 180:
            normalized_long -= 360  # Convert to [-180, 180] range

        # Plot crash point
        m.scatter(
            normalized_long,
            0,
            marker="x",
            color="r",
            s=10,
            label=f"True crash: ({normalized_long:.2f}, 0)",
        )
        if estimated_traj is not None:
            estimated_crash_coords = estimated_traj[-1, :2]
            estimated_crash_long = latlong_from_cart(estimated_crash_coords) / (
                np.pi / 180
            )

            # normalize longitude coordinate
            normalized_est_longitude = estimated_crash_long % 360
            if normalized_est_longitude > 180:
                normalized_est_longitude -= 360  # Convert to [-180, 180] range

            m.scatter(
                normalized_est_longitude,
                0,
                marker="s",
                s=10,
                label=f"Predicted crash: ({normalized_est_longitude:.2f}, 0)",
            )

        if uncertainties is not None:
            uncertainty = uncertainties[-150]
            variance = np.array([uncertainty[0][0], uncertainty[1][1]])
            long_variance = latlong_from_cart(variance) / (np.pi / 180)
            long_std = long_variance**0.5

            plt.errorbar(normalized_est_longitude, 0, xerr=long_std)

    if dim == 3:
        latlong_crash_coords = earth_rotation(crash_coords, true_crash_time)
        latlong_crash_coords /= np.pi / 180

        # normalize latitude coordinate
        normalized_latitude = latlong_crash_coords[0] % 180
        if normalized_latitude > 90:
            normalized_latitude -= 180  # Convert to [-90, 90] range

        # normalize longitude coordinate
        normalized_longitude = latlong_crash_coords[1] % 360
        if normalized_longitude > 180:
            normalized_longitude -= 360  # Convert to [-180, 180] range

        # plot crash point
        m.scatter(
            normalized_longitude,
            normalized_latitude,
            marker="x",
            color="r",
            s=10,
            label=f"True Crash: ({normalized_longitude:.2f}, {normalized_latitude:.2f})",
        )
        if estimated_traj is not None:
            estimated_crash_coords = estimated_traj[-1, :3]
            estimated_crash_latlong = earth_rotation(
                estimated_crash_coords, estimated_crash_time
            )
            estimated_crash_latlong /= np.pi / 180

            # normalize latitude coordinate
            est_norm_lat = estimated_crash_latlong[0] % 180
            if est_norm_lat > 90:
                est_norm_lat -= 180  # Convert to [-90, 90] range

            # normalize longitude coordinate
            est_norm_long = estimated_crash_latlong[1] % 360
            if est_norm_long > 180:
                est_norm_long -= 360  # Convert to [-180, 180] range

            m.scatter(
                est_norm_long,
                est_norm_lat,
                marker="s",
                s=10,
                label=f"Predicted crash: ({est_norm_long:.2f}, {est_norm_lat:.2f})",
            )
        if uncertainties is not None:
            uncertainty = uncertainties[-130]
            variance = np.array(
                [uncertainty[0][0], uncertainty[1][1], uncertainty[2][2]]
            )
            std = variance**0.5

            # puts it through the transform which is wrong
            latlong_std = latlong_from_cart(std) / (np.pi / 180)
            long_std = latlong_std[0]
            lat_std = latlong_std[1]
            # print(long_std)
            # print(lat_std)

            # baso percentage uncertainties, also wrong
            estimated_crash_longlat_std = estimated_crash_latlong * np.linalg.norm(
                std / estimated_crash_coords
            )
            # print(estimated_crash_longlat_std)

            # cant get this too work
            # major_axis = 2 *long_std
            # minor_axis = 2*lat_std
            # ellipse = Ellipse(xy=(est_norm_long, est_norm_lat), width=major_axis, height=minor_axis, angle=0)
            # m.ax.add_patch(ellipse)

            plt.errorbar(est_norm_long, est_norm_lat, xerr=long_std, yerr=lat_std)

    plt.legend()
    plt.title(title)
    if show:
        plt.show()
        plt.close()


def plot_error(
    true_traj, estimated_traj, title="Error in Trajectories", ax=None, show=True
) -> None:
    """
    Plots the norm of the error between the true and estimated trajectories over time.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray): The estimated trajectory data points.
        title (str): The title of the plot.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
    """
    true_traj = np.array(true_traj)
    estimated_traj = np.array(estimated_traj)
    error = np.linalg.norm(true_traj - estimated_traj, axis=1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(error)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    if show:
        plt.show()
        plt.close()


def Three_Dim_Slice_Trajectory(
    true_traj,
    estimated_traj,
    observation_states,
    observation_times,
    sim_times,
    dt,
    Three_Dim_crash_coords,
):
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
    ax = fig.add_subplot(111, projection="3d")

    plot_duration = 1000

    start_index = np.argmax(sim_times > observation_times[0])
    start_time = sim_times[start_index]
    end_time = start_time + plot_duration

    true_traj_slice = slice(start_index, int(end_time / dt))
    est_traj_slice = slice(0, int(plot_duration / dt))
    measure_slice = slice(0, np.argmax(observation_times > end_time))

    ax.plot(
        true_traj[:, 0][true_traj_slice],
        true_traj[:, 1][true_traj_slice],
        true_traj[:, 2][true_traj_slice],
        label="True Trajectory",
    )
    ax.scatter(
        observation_states[:, 0][measure_slice],
        observation_states[:, 1][measure_slice],
        observation_states[:, 2][measure_slice],
        marker="x",
        color="r",
        label="Noisy Measurements",
    )
    ax.plot(
        estimated_traj[:, 0][est_traj_slice],
        estimated_traj[:, 1][est_traj_slice],
        estimated_traj[:, 2][est_traj_slice],
        label="EKF Estimated Trajectory",
        linestyle="--",
    )
    ax.scatter(
        Three_Dim_crash_coords[0],
        Three_Dim_crash_coords[1],
        Three_Dim_crash_coords[2],
        color="red",
        marker="X",
        label="Crash Site",
    )
    ax.set_title("Extended Kalman Filter for Satellite Motion")
    ax.set_xlabel("Position X")
    ax.set_ylabel("Position Y")
    ax.set_zlabel("Position Z")
    ax.legend()
    plt.show()


def plot_position_error(
    true_traj,
    estimated_traj,
    times,
    observation_states=None,
    observation_times=None,
    ax=None,
    show=True,
) -> None:
    """
    Plots the position error between the true and estimated trajectories in 2D or 3D.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray): The estimated trajectory data points.
        times (np.ndarray): Timestamps for each data point.
        observation_states (np.ndarray, optional): Observed states during the trajectory.
        observation_times (np.ndarray, optional): Timestamps for each observation.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
    """
    # Calculate absolute error between true and estimated trajectories
    dim = len(true_traj[0])
    position_diff = true_traj[:dim] - estimated_traj[:dim]
    error = np.linalg.norm(position_diff, axis=1)
    fig, ax = plt.subplots()
    ax.plot(times, error, label="Error in EKF Position")
    if observation_states is not None:
        if observation_times is None:
            raise ValueError(
                "Observation times must be provided with observation states."
            )
        observation_error = np.linalg.norm(
            observation_states[:dim] - true_traj[:dim], axis=1
        )
        ax.scatter(observation_times, observation_error, label="Error in Observations")
    ax.set_title("Error in Position")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [m]")
    ax.legend()
    if show:
        plt.show()
        plt.close()


def plot_velocity_error(
    true_traj,
    estimated_traj,
    times,
    observation_states=None,
    observation_times=None,
    ax=None,
    show=True,
) -> None:
    """
    Plots the velocity error between the true and estimated trajectories in 2D or 3D.

    Args:
        true_traj (np.ndarray): The true trajectory data points.
        estimated_traj (np.ndarray): The estimated trajectory data points.
        times (np.ndarray): Timestamps for each data point.
        observation_states (np.ndarray, optional): Observed states during the trajectory.
        observation_times (np.ndarray, optional): Timestamps for each observation.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        show (bool): Whether to display the plot. Defaults to True.
    """
    # Calculate absolute error between true and estimated trajectories
    dim = len(true_traj[0])
    diff = true_traj[dim:] - estimated_traj[dim:]
    error = np.linalg.norm(diff, axis=1)
    fig, ax = plt.subplots()
    ax.plot(times, error, label="Error in EKF Velocity")
    if observation_states is not None:
        if observation_times is None:
            raise ValueError(
                "Observation times must be provided with observation states."
            )
        observation_error = np.linalg.norm(
            observation_states[dim:] - true_traj[dim:], axis=1
        )
        ax.scatter(observation_times, observation_error, label="Error in Observations")
    ax.set_title("Error in Velocity")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [m]")
    ax.legend()
    if show:
        plt.show()
        plt.close()


def plot_theoretical_empirical_observation_error(
    sim_states,
    sim_times,
    observation_states,
    observation_times,
    observed_covariances,
) -> None:
    """
    Plots both theoretical and empirical observation errors for position and velocity.

    Args:
        sim_states (np.ndarray): Simulated states over time.
        sim_times (np.ndarray): Timestamps for simulated states.
        observation_states (np.ndarray): Observed states.
        observation_times (np.ndarray): Timestamps for observed states.
        observed_covariances (np.ndarray): Covariance matrices for the observations.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    sim_times_observed = np.array(
        [i for i, t in enumerate(sim_times) if t in observation_times]
    )
    vel_observation_error = np.linalg.norm(
        (observation_states - sim_states[sim_times_observed])[:, 3:], axis=1
    )
    vel_std = np.sqrt(np.trace(observed_covariances[:, 3:, 3:], axis1=1, axis2=2))
    ax1.scatter(
        observation_times, vel_observation_error, label="Empirical", marker="x", s=20
    )
    ax1.scatter(observation_times, vel_std, label="Theoretical", marker="+", s=30)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Velocity error [m/s]")
    ax1.set_title("Velocity measurement error")
    ax1.legend()

    pos_observation_error = np.linalg.norm(
        (observation_states - sim_states[sim_times_observed])[:, :3], axis=1
    )
    pos_std = np.sqrt(np.trace(observed_covariances[:, :3, :3], axis1=1, axis2=2))
    ax2.scatter(
        observation_times, pos_observation_error, label="Empirical", marker="x", s=20
    )
    ax2.scatter(observation_times, pos_std, label="Theoretical", marker="+", s=30)
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
    for i in range(len(lats - 1)):
        for j in range(len(lons - 1)):
            # dist between mean trajectory and grid point
            dist = np.linalg.norm(mean_trajectory[:, :dim] - [lons[j], lats[i]], axis=1)
            dist = dist[:, np.newaxis]  # Column vector

            # Compute PDF based on uncertainty matrix
            try:
                inv_uncertainty_matrix = np.linalg.inv(
                    uncertainties
                )  # Invert uncertainty matrix
                pdf = np.exp(
                    -0.5 * np.sum(dist @ inv_uncertainty_matrix * dist, axis=1)
                )
            except np.linalg.LinAlgError:
                pdf = np.zeros_like(
                    dist.flatten()
                )  # If matrix is singular, set PDF to zero I kept getting LinAlgError: Singular matrix so had to add this condition

            # Update heatmap
            heatmap[i, j] = np.mean(pdf)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    m = Basemap(projection="mill", lon_0=0)
    m.drawcoastlines()
    m.drawparallels(np.arange(-90.0, 91.0, 30.0), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180.0, 181.0, 60.0), labels=[0, 0, 0, 1])
    x, y = m(grid_lons, grid_lats)
    m.pcolormesh(x, y, heatmap, cmap="hot_r")
    plt.colorbar(label="Probability Density")
    plt.title("Impact Location Heatmap")
    plt.show()

    return (
        print("Heatmap shape:", heatmap.shape),
        print("Length of lats:", len(lats)),
        print("Length of lons:", len(lons)),
    )  # Debugging


def slice_by_time(
    arr, times, start_time=None, end_time=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slices an array by time.

    Args:
        arr (np.ndarray): The array to slice.
        times (np.ndarray): The times corresponding to each row of the array.
        start_time (float): The start time of the slice.
        end_time (float): The end time of the slice.

    :return: The sliced array and corresponding times.
    """
    if arr is None:
        return None

    arr = np.array(arr)
    times = np.array(times)
    start_index = None
    end_index = None
    if start_time is not None:
        start_index = np.argmax(times >= start_time)
    if end_time is not None:
        end_index = np.argmax(times > end_time)
    return arr[start_index:end_index], times[start_index:end_index]
