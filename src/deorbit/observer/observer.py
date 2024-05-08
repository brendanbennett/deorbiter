import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from deorbit.utils.constants import EARTH_RADIUS, EARTH_ROTATIONAL_SPEED


class Observer:
    """
    HOW TO (current, subject to change hopefully):
    -self.states and self.times must be initially set to the states and times from the simulator.
    -self.number_of_radars can be specified (by default is 1) and by default they are give equally spaced positions (longitude-latitude)
    around the equator. self.positions_of_radars can be set in list[long, lat] format for custom positions.

    METHODS:
    -self.plot_config() method shows a plot of the radar station positions
    -self.run(checking_interval) runs the radar simulation but self.states and self.times needs setting first.


    THINGS TO DO STILL:
    - Test if this even works...
    - Make it more user friendly and just layed out better
    - self.run() method should generate an instance of ObsData which will be used going forward in the Predictor.
    - add explicit type checking
    - noise covariance to radar observations
    """

    def __init__(self, **kwargs):
        self.radius: float = kwargs.get("radius", EARTH_RADIUS)
        self.rotation: float = kwargs.get("rotation", EARTH_ROTATIONAL_SPEED)
        self.number_of_radars: int = kwargs.get("number_of_radars", 1)
        self.positions_of_radars: npt.NDArray = kwargs.get(
            "positions_of_radars", self._default_radar_positions(self.number_of_radars)
        )
        self.states: npt.NDArray | None = kwargs.get("states", None)
        self.times: npt.NDArray | None = kwargs.get("times", None)
        self.observed_states: list[list[float]] | None = None
        self.observed_times: list[float] | None = None

        self._radar_position_validator()

    @classmethod
    def _default_radar_positions(cls, number_of_radars: int) -> npt.NDArray:
        """
        Validator to check whether 'positions_of_radars' is in the correct format,
        if it is None then default positions are given as equally spaced radars around the equator
        """
        delta_lon = (np.pi * 2) / number_of_radars
        rad_default_positions = np.zeros(shape=(number_of_radars, 2))
        for i in range(number_of_radars):
            radar_lon = 0 + i * delta_lon
            rad_default_positions[i, 0] = radar_lon
        return rad_default_positions

    def _radar_position_validator(self):
        # Convert list to np.array if it's not None and validate shape
        if self.positions_of_radars.shape != (self.number_of_radars, 2):
            raise ValueError(
                f"positions_of_radars must be a numpy array with shape ({self.number_of_radars}, 2)"
            )

    def _rad_xyz(self, longlat):
        """Gives x and y on the Earth surface at a specified longitude and latitude"""
        x_earth = self.radius * np.cos(longlat[1]) * np.sin(longlat[0])
        z_earth = self.radius * np.sin(longlat[1]) * np.sin(longlat[0])
        y_earth = self.radius * np.cos(longlat[0])

        return x_earth, y_earth, z_earth

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

    def run(self, checking_interval):
        """
        Runs the simulation of the radar measurements taken at regular intervals specified by 'checking_interval'.
        Returns self.observed_times and self.observed_states containing the times and states from the simulation
        where the satellite has been observed by a radar station.
        """
        times_observed = []
        states_observed = []

        # Only checking the states at a regular interval specified by time_step
        radar_initial_positions = self.positions_of_radars
        times_checked = self.times[::checking_interval]
        states_checked = self.states[::checking_interval]

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
                    states_observed.append(states_checked[i])
                    break  # If it is in los, avoid checking other radars and move to next checking interval

        self.observed_times = np.array(times_observed)
        self.observed_states = np.array(states_observed)

    def _check_los(
        self, longlat, state
    ):  # Checking line of sight using radar longlat and satellite state
        if np.dot(self._rad_xyz(longlat), state[0:3]) >= 0:
            return True
        else:
            return False


'''
    def _earth_rad(self, latitude): 
        """Gives radius of earth at a specified latitude (rad) (longitude does not affect radius) for oblate spheroid"""
        if self.config.max_radius == self.config.min_radius: #if the earth is modelled as a sphere
            return EARTH_RADIUS
        else: #else the earth is an oblate spheroid
            a = self.config.max_radius
            b = self.config.min_radius
            radius = np.sqrt(((a**2 * np.cos(latitude))**2 + (b**2 * np.sin(latitude))**2) / ((a * np.cos(latitude))**2 + (b * np.sin(latitude))**2))
            return radius
'''
