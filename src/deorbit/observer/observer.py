import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from deorbit.utils.constants import EARTH_RADIUS, EARTH_ROTATIONAL_SPEED


class Observer:
    def __init__(self, **kwargs):
        """
        ATTRIBUTES:
        :::Earth Model Configuration:::
        self.radius: Sets the distance of the Earth's radius
        self.rotation: Sets the speed in rad/s of the Earth's rotation

        :::Radar Stations Configuration:::
        self.number_of_radars: Determines how many radar stations are used in the simulation, default 1
        self.positions_of_radars: Optional numpy array containing [longitude, latitude] for each satellite, otherwise default positions set
        self.radar_variance_per_m: Simulates the uncertainty in the measurement. The variance increases linearly as distance to satellite increases. Default 0.1

        :::Observer Output:::
        self.observed_states: After the Observer has been run, observed states are stored here to be used in the Predictor
        self.observed_times: After the Observer has been run, observed times are stored here to be used in the Predictor

        METHODS:
        self.plot_config(): Shows a 3D plot of the radar station configuration

        self.run(sim_times, sim_states, checking_interval):
        ->sim_states: the states of the satellite given by the Simulator
        ->sim_times: the time steps of each state given by the Simulator
        ->checking_interval: the size of the jump between times in sim_times in which the radar stations are checking for line of sight
        """
        self.radius: float = kwargs.get("radius", EARTH_RADIUS)
        self.rotation: float = kwargs.get("rotation", EARTH_ROTATIONAL_SPEED)
        self.number_of_radars: int = kwargs.get("number_of_radars", 1)
        self.positions_of_radars: npt.NDArray = kwargs.get(
            "positions_of_radars", self._default_radar_positions(self.number_of_radars)
        )
        self.radar_variance_per_m: float = kwargs.get("radar_noise_factor", 0.1)
        self.observed_states: list[list[float]] | None = None
        self.observed_times: list[float] | None = None

        self._radar_position_validator()

    @classmethod
    def _default_radar_positions(cls, number_of_radars: int) -> npt.NDArray:
        """
        Sets default radar positions, returns longitude and latitude of radars equally spaced around the equator.
        """
        delta_lon = (np.pi * 2) / number_of_radars
        rad_default_positions = np.zeros(shape=(number_of_radars, 2))
        for i in range(number_of_radars):
            radar_lon = 0 + i * delta_lon
            rad_default_positions[i, 0] = radar_lon
        return rad_default_positions

    def _radar_position_validator(self):
        """
        Validates shape of the numpy array containing radar positions
        """
        if self.positions_of_radars.shape != (self.number_of_radars, 2):
            raise ValueError(
                f"positions_of_radars must be a numpy array with shape ({self.number_of_radars}, 2)"
            )

    def _rad_xyz(self, longlat):
        """
        Gives x and y on the Earth surface at a specified longitude and latitude
        """
        x_earth = self.radius * np.cos(longlat[1]) * np.sin(longlat[0])
        z_earth = self.radius * np.sin(longlat[1]) * np.sin(longlat[0])
        y_earth = self.radius * np.cos(longlat[0])

        return x_earth, y_earth, z_earth

    def _measurement_noise(self, rad_longlat, sat_state):
        """
        Method which returns the observed state with additional measurement noise sampled from a multivariate Gaussian.
        The noise increases linearly as distance between the radar and the satellite increases.
        """
        distance = np.linalg.norm(sat_state[0:3] - self._rad_xyz(rad_longlat))
        variance = distance * self.radar_variance_per_m

        cov = np.diag([variance, variance, variance, variance])
        noisy_state = np.random.multivariate_normal(sat_state, cov)

        return noisy_state

    def _check_los(self, longlat, state):
        """
        Checking line of sight using radar longlat and satellite state
        """
        if np.dot(self._rad_xyz(longlat), (state[0:3] - self._rad_xyz(longlat))) >= 0:
            return True
        else:
            return False

    def plot_config(self):
        """
        Method which gives a visual representation of radar station layout
        """
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
            x_radar, y_radar, z_radar = self._rad_xyz(xi)
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

    def run(self, sim_times, sim_states, checking_interval):
        """
        Runs the simulation of the radar measurements taken at regular intervals specified by 'checking_interval'.
        Returns self.observed_times and self.observed_states containing the times and states from the simulation
        where the satellite has been observed by a radar station.
        """

        times_observed = []
        states_observed = []

        # Only checking the states at a regular interval specified by time_step
        radar_initial_positions = self.positions_of_radars
        times_checked = sim_times[::checking_interval]
        states_checked = sim_states[::checking_interval]

        for i, xi in enumerate(times_checked):
            for radar, longlat in enumerate(radar_initial_positions):  # For each radar:
                longlat[0] = (
                    longlat[0] + self.rotation * times_checked[i]
                )  # Move longitude in line with Earth's rotation
                in_sight = self._check_los(
                    longlat, states_checked[i]
                )  # Check if the satellite is in los
                if (
                    in_sight == True
                ):  # If the satellite is in los, append the time and state to list of observed states
                    times_observed.append(times_checked[i])
                    states_observed.append(
                        self._measurement_noise(longlat, states_checked[i])
                    )
                    break  # If it is in los, avoid checking other radars and move to next checking interval

        self.observed_times = np.array(times_observed)
        self.observed_states = np.array(states_observed)
