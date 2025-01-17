import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from deorbit.utils.constants import EARTH_RADIUS, EARTH_ROTATIONAL_SPEED
from deorbit.utils.coords import cart_from_latlong


class Observer:
    """Class used to observe the state of a satellite during a simulation by modelling radar stations and their uncertainties.
    
    :keyword dim: Dimension of the simulation, 2 or 3. Default 2.
    :keyword radius: Sets the distance of the Earth's radius. Default EARTH_RADIUS.
    :keyword rotation: Sets the speed in rad/s of the Earth's rotation. Default EARTH_ROTATIONAL_SPEED.
    :keyword number_of_radars: Determines how many radar stations are used in the simulation. Default 1.
    :keyword positions_of_radars: Optional numpy array containing [latitude, longitude] for each satellite, otherwise default equally spaced positions set.
    :keyword radar_position_std_per_distance: Sets the standard deviation of the radar position noise per distance. Default 0.005.
    :keyword radar_velocity_std_per_distance: Sets the standard deviation of the radar velocity noise per distance. Default 0.000001.
    :keyword radar_velocity_std_per_speed: Sets the standard deviation of the radar velocity noise per speed. Default 0.0005.
    :ivar observed_states: After the Observer has been run, observed states are stored here to be used in the Predictor.
    :ivar observed_times: After the Observer has been run, observed times are stored here to be used in the Predictor.
    :ivar observed_covariances: The noise covariance associated with each state measurement made by the radar stations.
    """
    def __init__(self, **kwargs):
        self.dim: float = kwargs.get("dim", 2)
        self.radius: float = kwargs.get("radius", EARTH_RADIUS)
        self.rotation: float = kwargs.get("rotation", EARTH_ROTATIONAL_SPEED)
        self.number_of_radars: int = kwargs.get("number_of_radars", 1)
        self.positions_of_radars: np.ndarray = kwargs.get(
            "positions_of_radars",
            self._default_radar_positions(self.number_of_radars, self.dim),
        )
        # Observation distances are about 50-500km.
        self.radar_position_std_per_distance: float = kwargs.get("radar_position_std_per_distance", 0.005)
        # Want distance to only play a small role, in order to make velocity error more relative
        self.radar_velocity_std_per_distance: float = kwargs.get("radar_velocity_std_per_distance", 0.000001)
        self.radar_velocity_std_per_speed: float = kwargs.get("radar_velocity_std_per_speed", 0.0005)
        self.observed_states: list[list[float]] | None = None
        self.observed_times: list[float] | None = None
        self.observed_covariances: np.ndarray = None

        if self.dim not in [2, 3]:
            raise ValueError("dim must be 2 or 3")
        self._radar_position_validator()

    @staticmethod
    def _default_radar_positions(number_of_radars: int, dim: int) -> np.ndarray:
        """Sets default radar positions, returns latitude and longitude of radars equally spaced around the equator.
        
        :param number_of_radars: Number of radar stations
        :param dim: Dimension of the simulation
        :return: Array of latitude and longitude of radar positions. If dim=2, returns 1D array of longitudes.
        """
        if dim == 3:
            rad_default_positions = np.zeros(shape=(number_of_radars, 2))

            #transform to get uniform coverage of earth
            indices = np.arange(0, number_of_radars, dtype = float) + 0.5

            random_theta_sample = np.arcsin(1-2*indices/number_of_radars)
            random_phi_sample = np.pi*(1 + 5**0.5)*indices

            rad_default_positions = np.stack((random_theta_sample, random_phi_sample), axis=1)

        elif dim == 2:
            rad_default_positions = np.linspace(
                0, 2 * np.pi, number_of_radars, endpoint=False
            )
        return rad_default_positions

    def _radar_position_validator(self):
        """Validates shape of the numpy array containing radar positions
        """
        if self.dim == 2 and self.positions_of_radars.shape != (self.number_of_radars,):
            raise ValueError(
                f"With dim=2, positions_of_radars must be a numpy array with shape ({self.number_of_radars},)"
            )
        elif self.dim == 3 and self.positions_of_radars.shape != (
            self.number_of_radars,
            2,
        ):
            raise ValueError(
                f"With dim=3, positions_of_radars must be a numpy array with shape ({self.number_of_radars}, 2)"
            )

    def _measurement_noise(self, rad_latlong, sat_state) -> tuple[np.ndarray, np.ndarray]:
        """Method which returns the observed state with additional measurement noise sampled from a multivariate Gaussian.
        The noise increases linearly as distance between the radar and the satellite increases.
        
        :param rad_latlong: Latitude and longitude of the radar station
        :type rad_latlong: Sequence[float, float]
        :param sat_state: The true state of the satellite
        :type sat_state: np.ndarray
        :return: The noisy state and the covariance matrix of the noise.
        """
        distance = np.linalg.norm(
            sat_state[0 : self.dim] - cart_from_latlong(rad_latlong)
        )
        speed = np.linalg.norm(sat_state[self.dim :])
        pos_std = distance * self.radar_position_std_per_distance
        vel_std = distance * self.radar_velocity_std_per_distance + speed * self.radar_velocity_std_per_speed

        cov = np.eye(self.dim * 2)
        cov[:self.dim, :self.dim] *= pos_std ** 2
        cov[self.dim:, self.dim:] *= vel_std ** 2
        noisy_state = np.random.multivariate_normal(sat_state, cov)
        return noisy_state, cov

    def _check_los(self, latlong, state):
        """Checking line of sight using radar latlong and satellite state
        
        :param latlong: Latitude and longitude of the radar station
        :type latlong: Sequence[float, float]
        :param state: The true state of the satellite
        :type state: np.ndarray
        """
        if (
            np.dot(
                cart_from_latlong(latlong),
                (state[0 : self.dim] - cart_from_latlong(latlong)),
            )
            >= 0
        ):
            return True
        else:
            return False

    def plot_config(self):
        """Method which gives a visual representation of radar station layout
        """
        if self.dim == 2:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            ax.set_xlim(-self.radius * 1.1, self.radius * 1.1)
            ax.set_ylim(-self.radius * 1.1, self.radius * 1.1)
            circle = plt.Circle((0, 0), self.radius, color="g", alpha=0.5)
            ax.add_artist(circle)

            for i, xi in enumerate(self.positions_of_radars):
                x_radar, y_radar = cart_from_latlong(xi)
                ax.scatter(
                    x_radar, y_radar, color="r", marker="o", s=50, edgecolors="black"
                )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Earth Radar Station Config")
            plt.show()
        elif self.dim == 3:
            r = self.radius

            theta = np.linspace(0, 2 * np.pi, 100)
            phi = np.linspace(0, np.pi, 100)
            theta, phi = np.meshgrid(theta, phi)

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            ax.plot_surface(x, y, z, color="g", alpha=0.5)

            # Calculate distance of each radar station from the viewer
            distances = np.linalg.norm(self.positions_of_radars, axis=1)

            # Normalize distances to range [0, 1]
            normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

            for i, xi in enumerate(self.positions_of_radars):
                x_radar, y_radar, z_radar = cart_from_latlong(xi)
                # alpha = 1 - normalized_distances[i]  # Adjust alpha based on normalized distance
                ax.scatter(
                    x_radar,
                    y_radar,
                    z_radar,
                    color="r",
                    marker="o",
                    s=50,
                    edgecolors="black",
                    # alpha = alpha
                )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Earth Radar Station Config")
            plt.show()

    def run(self, sim_states: np.ndarray, sim_times: np.ndarray, checking_interval: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Method which runs the Observer. The Observer checks for line of sight at regular time intervals equal to (checking_interval * simulator interval) seconds.
        
        :param sim_states: The states of the satellite given by the Simulator
        :param sim_times: The times of each state given by the Simulator
        :param checking_interval: Radars check for line of sight at regular time intervals equal to (checking_interval * simulator time step) seconds.
        :return: The observed states, times and covariances
        """
        if sim_states.shape[1] != 2 * self.dim:
            raise ValueError(
                f"Provided data is for a {int(sim_states.shape[1] / 2)}D simulation, but Observer is {self.dim}D."
            )
        times_observed = []
        states_observed = []
        covariances = []

        # Only checking the states at a regular interval specified by time_step
        radar_initial_positions = self.positions_of_radars
        times_checked = sim_times[::checking_interval]
        states_checked = sim_states[::checking_interval]

        if self.dim == 3:
            for i, time in enumerate(times_checked):
                for latlong in radar_initial_positions:  # For each radar:
                    latlong[1] = (
                        latlong[1] + self.rotation * time
                    )  # Move longitude in line with Earth's rotation
                    in_sight = self._check_los(
                        latlong, states_checked[i]
                    )  # Check if the satellite is in los
                    if (
                        in_sight == True
                    ):  # If the satellite is in los, append the time and state to list of observed states
                        times_observed.append(time)
                        noisy_state, covariance = self._measurement_noise(
                            latlong, states_checked[i]
                        )
                        states_observed.append(noisy_state)
                        covariances.append(covariance)
                        break  # If it is in los, avoid checking other radars and move to next checking interval
        if self.dim == 2:
            for i, time in enumerate(times_checked):
                for long in radar_initial_positions:  # For each radar:
                    long = (
                        long + self.rotation * time
                    )  # Move longitude in line with Earth's rotation
                    in_sight = self._check_los(
                        long, states_checked[i]
                    )  # Check if the satellite is in los
                    if (
                        in_sight == True
                    ):  # If the satellite is in los, append the time and state to list of observed states
                        times_observed.append(time)
                        noisy_state, covariance = self._measurement_noise(
                            long, states_checked[i]
                        )
                        states_observed.append(noisy_state)
                        covariances.append(covariance)
                        break  # If it is in los, avoid checking other radars and move to next checking interval

        self.observed_times = np.array(times_observed)
        self.observed_states = np.array(states_observed)
        self.observed_covariances = np.array(covariances)
        
        return self.observed_states, self.observed_times, self.observed_covariances
