#Pydantic has dropped @validator for @field_validator and I can't work out how to use it to set default radar positions :(

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pydantic import BaseModel, field_validator
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(r'C:\Users\User\Dropbox\My PC (LAPTOP-L7DADSJH)\Documents\PMSC\MIR\mir-orbiter\src')
from utils.constants import (
    EARTH_RADIUS,
    EARTH_ROTATIONAL_SPEED,
    EARTH_EQUATORIAL_RADIUS,
    EARTH_POLAR_RADIUS
)


class SimData(BaseModel):
    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None  # Defaults to dim = 2
    times: list[float]

class ObsData(BaseModel):
    """
    Output of the Observer which is a sparser copy of SimData from the Simulator. 
    Only includes the states at times where the satellite is in view of a ground radar station
    """
    x1: list[float]
    x2: list[float]
    x3: Optional[list[float]] = None
    times: list[float]

class ObserverConfig(BaseModel):
    """
    Observer configuration class to specify number of radar stations, positions and Earth parameters
    """
    max_radius: Optional[float] = EARTH_RADIUS
    min_radius: Optional[float] = EARTH_RADIUS
    rotation: Optional[float] = EARTH_ROTATIONAL_SPEED
    number_of_radars: Optional[int] = 1
    positions_of_radars: Optional[list[list[float]]] = None

    
    @field_validator('positions_of_radars', mode='before')
    @classmethod
    def _check_positions_of_radars(cls, value):
        """
        Validator to check whether 'positions_of_radars' is in the correct format, 
        if it is None then default positions are given as equally spaced radars around the equator
        """
        if number_of_radars is None:
            delta_lon = (np.pi*2)/number_of_radars
            rad_default_positions = np.zeros(shape=(number_of_radars, 2))
            for i in range(number_of_radars):
                radar_lon = 0 + i*delta_lon
                rad_default_positions[i, 0] = radar_lon
            return rad_default_positions

        # Convert list to np.array if it's not None
        elif number_of_radars is not None:
            number_of_radars = np.array(number_of_radars)
            if number_of_radars.shape != (number_of_radars, 2):
                raise ValueError(f"positions_of_radars must be a numpy array with shape ({number_of_radars}, 2)")
        return number_of_radars
    
    def plot_config(self):
        """
        Method which gives a visual representation of radar station layout in 3D (later will be adapted to 2D also)
        """
        a = EARTH_EQUATORIAL_RADIUS  
        c = EARTH_POLAR_RADIUS  

        theta = np.linspace(0, 2 * np.pi, 100) 
        phi = np.linspace(0, np.pi, 100) 
        theta, phi = np.meshgrid(theta, phi)

        x = a * np.sin(phi) * np.cos(theta)
        y = a * np.sin(phi) * np.sin(theta)
        z = c * np.cos(phi)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, z, color='g', alpha=0.8)

        '''
        #(trying to plot radars)
        for i, xi in enumerate(self.config.positions_of_radars):
            x_radar, y_radar, z_radar = self._rad_xyz(xi)
            ax.scatter(x_radar, y_radar, z_radar)
        '''

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Earth Radar Station Config')
        plt.show()

class Observer:
    """
    Observer class is initialised with ObserverConfig and SimData instances. 
    Run method will generate an instance of ObsData which will be used going forward in the Predictor. 
    """
    def __init__(self, config: ObserverConfig, simdata: SimData) -> None:
        self.states_pre: SimData = simdata
        self.config: ObserverConfig = config

    def _earth_rad(self, latitude): 
        """Gives radius of earth at a specified latitude (rad) (longitude does not affect radius)"""
        if self.config.max_radius == self.config.min_radius: #if the earth is modelled as a sphere
            return EARTH_RADIUS
        else: #else the earth is an oblate spheroid
            a = self.config.max_radius
            b = self.config.min_radius
            radius = np.sqrt(((a**2 * np.cos(latitude))**2 + (b**2 * np.sin(latitude))**2) / ((a * np.cos(latitude))**2 + (b * np.sin(latitude))**2))
            return radius

    def _rad_xyz(self, longitude, latitude): 
        """Gives x and y on the Earth surface at a specified longitude and latitude"""
        x_earth = self._earth_rad(longitude) * np.cos(longitude)
        y_earth = self._earth_rad(longitude) * np.sin(longitude)
        z_earth = self._earth_rad(latitude) * np.sin(latitude)
        return x_earth, y_earth, z_earth
    
    def _run_3d(self):
        for i, interval in enumerate(simdata.times):
            #new_radar_pos()
        return
    def _run_2d(self):
        return
    
    def plot_config(self):
        self.config.plot_config()

    def run(self):
        """At each time step in SimData, will move each radar station to a new position based on EARTH_ROTATIONAL_SPEED.
        Will iterate through each radar station to check line of sight with the satellite xyz at that time interval. 

        Upon line of sight returning true it will save the time interval and satellite xyz with Gaussian measurement noise 
        to ObsData and move to the next time step to avoid unecessary calculations with other radars. 
        If there is no line of sight to any radar it will proceed to the next time step.
        """
        if simdata.x3 == None:
            self._run_2d()
        else:
            self._run_3d()



#testing some stuff...
number_of_radars = 2
radar_pos = np.array([[np.pi/2, 1.2*np.pi], [(3/2)*np.pi, 2*np.pi]])

obs_config = ObserverConfig(number_of_radars=number_of_radars, positions_of_radars=radar_pos)
simdata = SimData(x1=[1.0, 2.0, 3.0], x2=[1.0, 2.0, 3.0], times=[1.0, 2.0, 3.0]) #just a placeholder for the moment
obs_config.plot_config()

observer_object = Observer(config=obs_config, simdata=simdata)


'''
#EARTH
theta = np.linspace(0, 2*np.pi, 100)
r_earth = EARTH_RADIUS
x_earth = r_earth*np.cos(theta)
y_earth = r_earth*np.sin(theta)

#POSITIONS
sat_xy = np.array([r_earth+1000, -r_earth+1000], dtype=float)
rad1_xy = np.array([0, -r_earth], dtype=float)

#LINE OF SIGHT - Returns True or False for a given radar and satellite position vector (2-dimensional)
def check_lineofsight(satxy, radxy): #True if in sight
    d_radar = r_earth
    d_satellite = np.linalg.norm(satxy)
    theta = np.arccos(np.dot((satxy/np.linalg.norm(satxy)),(radxy/np.linalg.norm(radxy))))
    a = d_satellite*np.cos(theta)
    if a >= d_radar:
        return True
    else:
        return False
    
def new_radar_position(rad_latlon, dt): #gives the new radar position with a change in time dt

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111)

ax1.plot(x_earth, y_earth)
ax1.scatter(sat_xy[0], sat_xy[1], marker='x', color='r', label='Satellite')
ax1.scatter(rad1_xy[0], rad1_xy[1], marker='x', color='g', label='Radar 1')

ax1.set_ylim(-2*r_earth, 2*r_earth)
ax1.set_xlim(-2*r_earth, 2*r_earth)

plt.legend()
plt.show()
'''