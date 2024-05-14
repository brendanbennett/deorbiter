import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from deorbit.utils.constants import EARTH_RADIUS, EARTH_ROTATIONAL_SPEED
from deorbit.utils.coords import cart_from_latlong


class Observer:
    def __init__(self, **kwargs):
        """
        ATTRIBUTES:
        :::Earth Model Configuration:::
        self.radius: Sets the distance of the Earth's radius
        self.rotation: Sets the speed in rad/s of the Earth's rotation

        :::Radar Stations Configuration:::
        self.number_of_radars: Determines how many radar stations are used in the simulation, default 1
        self.positions_of_radars: Optional numpy array containing [latitude, longitude] for each satellite, otherwise default positions set
        self.radar_variance_per_m: Simulates the uncertainty in the measurement. The variance increases linearly as distance to satellite increases. Default 0.1

        :::Observer Output:::
        self.observed_states: After the Observer has been run, observed states are stored here to be used in the Predictor
        self.observed_times: After the Observer has been run, observed times are stored here to be used in the Predictor
        self.observed_covariances: The noise covariance associated with each state measurement made by the radar stations

        METHODS:
        self.plot_config(): Shows a 3D plot of the radar station configuration

        self.run(sim_times, sim_states, checking_interval):
        ->sim_states: the states of the satellite given by the Simulator
        ->sim_times: the time steps of each state given by the Simulator
        ->checking_interval: Radars check for line of sight at regular time intervals equal to (checking_interval * simulator interval) seconds.
        """
        self.dim: float = kwargs.get("dim", 2)
        self.radius: float = kwargs.get("radius", EARTH_RADIUS)
        self.rotation: float = kwargs.get("rotation", EARTH_ROTATIONAL_SPEED)
        self.number_of_radars: int = kwargs.get("number_of_radars", 1)
        self.positions_of_radars: npt.NDArray = kwargs.get(
            "positions_of_radars",
            self._default_radar_positions(self.number_of_radars, self.dim),
        )
        self.radar_variance_per_m: float = kwargs.get("radar_noise_factor", 0.1)
        self.observed_states: list[list[float]] | None = None
        self.observed_times: list[float] | None = None
        self.observed_covariances: npt.NDArray = None

        if self.dim not in [2, 3]:
            raise ValueError("dim must be 2 or 3")
        self._radar_position_validator()

    @staticmethod
    def _default_radar_positions(number_of_radars: int, dim: int) -> npt.NDArray:
        """
        Sets default radar positions, returns longitude and latitude of radars equally spaced around the equator.
        """
        if dim == 3:
            rad_default_positions = np.zeros(shape=(number_of_radars, 2))

            indices = np.arange(0, number_of_radars, dtype = float) + 0.5

            random_theta_sample = np.arcsin(1-2*indices/number_of_radars)
            random_phi_sample = np.pi*(1 + 5**0.5)*indices

    
            # positions in spherical coordinates
            rad_default_positions = np.stack((random_theta_sample, random_phi_sample), axis=1)
    
            # Convert spherical coordinates to Cartesian coordinates
           # x = radii * np.sin(inclination_angles) * np.cos(azimuthal_angles)
           # y = radii * np.sin(inclination_angles) * np.sin(azimuthal_angles)
            #z = radii * np.cos(inclination_angles)

        elif dim == 2:
            rad_default_positions = np.linspace(
                0, 2 * np.pi, number_of_radars, endpoint=False
            )
        return rad_default_positions

    def _radar_position_validator(self):
        """
        Validates shape of the numpy array containing radar positions
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

    def _measurement_noise(self, rad_latlong, sat_state):
        """
        Method which returns the observed state with additional measurement noise sampled from a multivariate Gaussian.
        The noise increases linearly as distance between the radar and the satellite increases.
        """
        distance = np.linalg.norm(
            sat_state[0 : self.dim] - cart_from_latlong(rad_latlong)
        )
        variance = distance * self.radar_variance_per_m

        cov = np.eye(self.dim * 2) * variance
        noisy_state = np.random.multivariate_normal(sat_state, cov)

        return noisy_state, cov

    def _check_los(self, latlong, state):
        """
        Checking line of sight using radar latlong and satellite state
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
        """
        Method which gives a visual representation of radar station layout
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
                ax.scatter(x_radar, y_radar, color="r", marker="o", s=50, edgecolors="black")

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

            for i, xi in enumerate(self.positions_of_radars):
                x_radar, y_radar, z_radar = cart_from_latlong(xi)
                ax.scatter(
                    x_radar,
                    y_radar,
                    z_radar,
                    color="r",
                    marker="o",
                    s=50,
                    edgecolors="black",
                )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Earth Radar Station Config")
            plt.show()

    def run(self, sim_states, sim_times, checking_interval):
        """
        Runs the simulation of the radar measurements using an instance of the deorbit simulation.
        Radars check for line of sight at regular time intervals equal to (checking_interval * simulator interval) seconds.
        Returns self.observed_times and self.observed_states class attributes containing the times and states
        when the satellite has been observed by a radar station.
        """
        if sim_states.shape[1] != 2 * self.dim:
            raise ValueError(f"Provided data is for a {int(sim_states.shape[1] / 2)}D simulation, but Observer is {self.dim}D.")
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
