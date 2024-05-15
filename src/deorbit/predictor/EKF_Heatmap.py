import deorbit
import matplotlib.pyplot as plt
import deorbit.data_models
from deorbit.predictor import EKF
from deorbit.utils.dataio import load_sim_data, load_sim_config
import numpy as np
from deorbit.observer import Observer
from mpl_toolkits.basemap import Basemap

class EKF_heatmap:
    def __init__(self, observations, dt, Q, R, P, H, true_traj):
        self.observations = observations
        self.dt = dt
        self.Q = Q
        self.R = R
        self.P = P
        self.H = H
        self.true_traj = true_traj

    def generate_heatmap(self):
        ekf = EKF()
        estimated_traj, uncertainties, estimated_times = ekf.run(
            self.observations, dt=self.dt, Q=self.Q, R=self.R, P=self.P, H=self.H)
        crash_coords = self.true_traj[-1, :]
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

# Usage example
# Instantiate EKF_heatmap class with parameters
## ekf_heatmap = EKF_heatmap(observations, dt, Q, R, P, H, true_traj)
## Call the generate_heatmap method
#ekf_heatmap.generate_heatmap()