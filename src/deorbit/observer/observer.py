#Pydantic has dropped @validator for @field_validator and I can't work out how to use it to set default radar positions :(

import sys
import deorbit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pydantic import BaseModel, field_validator
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D

from deorbit.utils.constants import (
    EARTH_RADIUS,
    EARTH_ROTATIONAL_SPEED
)

class ObsData(BaseModel):
    """
    Output of the Observer which is a sparser copy of SimData from the Simulator. 
    Only includes the states at times where the satellite is in view of a ground radar station
    """
    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None
    times: list[float]    

class Observer:
    """ 
    THINGS TO DO:
    - self.run() method should generate an instance of ObsData which will be used going forward in the Predictor. 
    - **ADD EXPLICIT TYPE CHECKS LATER**
    - noise covariance to radar observations
    - include velocity measurements
    """
    def __init__(self, **kwargs):
        self.radius: Optional[float] = kwargs.get('radius', EARTH_RADIUS)
        self.rotation: Optional[float] = kwargs.get('rotation', EARTH_ROTATIONAL_SPEED)
        self.number_of_radars: Optional[int] = kwargs.get('number_of_radars', 1)
        self.positions_of_radars: Optional[list[list[float]]] = kwargs.get('positions_of_radars', self._default_radar_positions())
        self.states: Optional[list[list[float]]] = kwargs.get('states', None)
        self.times: Optional[list[float]] = kwargs.get('times', None)
        self.observed_states: Optional[list[list[float]]] = None
        self.observed_times: Optional[list[float]] = None

        self._radar_position_validator()

    def _default_radar_positions(self):
        """
        Validator to check whether 'positions_of_radars' is in the correct format, 
        if it is None then default positions are given as equally spaced radars around the equator
        """
        delta_lon = (np.pi*2)/self.number_of_radars
        rad_default_positions = np.zeros(shape=(self.number_of_radars, 2))
        for i in range(self.number_of_radars):
            radar_lon = 0 + i*delta_lon
            rad_default_positions[i, 0] = radar_lon
        return rad_default_positions

    def _radar_position_validator(self):
        # Convert list to np.array if it's not None and validate shape
        self.number_of_radars = np.array(self.number_of_radars)
        if self.positions_of_radars.shape != (self.number_of_radars, 2):
            raise ValueError(f"positions_of_radars must be a numpy array with shape ({self.number_of_radars}, 2)")

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
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, z, color='g', alpha=0.5)
        
        for i, xi in enumerate(self.positions_of_radars):
            x_radar, y_radar, z_radar = self._rad_xyz(xi)
            ax.scatter(x_radar, y_radar, z_radar, color='r', marker='o', s=50, edgecolors='black')
                
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Earth Radar Station Config')
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
            for radar, longlat in enumerate(radar_initial_positions): # For each radar:
                longlat[0] = longlat[0] + self.rotation*times_checked[i] # Move longitude in line with Earth's rotation
                in_sight = self._check_los(longlat, states_checked[i]) # Check if the satellite is in los
                if in_sight == True: # If the satellite is in los, append the time and state to list of observed states
                    times_observed.append(times_checked[i])
                    states_observed.append(states_checked[i])
                    break # If it is in los, avoid checking other radars and move to next checking interval
                
        self.observed_times = np.array(times_observed)
        self.observed_states = np.array(states_observed)

    def _check_los(self, longlat, state): # Checking line of sight using radar longlat and satellite state
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