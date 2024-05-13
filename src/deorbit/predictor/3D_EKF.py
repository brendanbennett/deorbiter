from itertools import count

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from deorbit.simulator import Simulator, generate_sim_config
from deorbit.simulator.atmos import AtmosphereModel
from deorbit.utils.constants import (
    EARTH_RADIUS,
    GM_EARTH,
    MEAN_DRAG_COEFF,
    MEAN_XSECTIONAL_AREA,
    SATELLITE_MASS,
)

class Three_D_EKF:
    def __init__(self, **kwargs):
        sim_method = kwargs.get("sim_method", "RK4")
        atmos_model = kwargs.get("atmos_model", "coesa_atmos_fast")
        sim_method_kwargs = kwargs.get("sim_method_kwargs", None)
        atmos_kwargs = kwargs.get("atmos_kwargs", None)

        integration_sim_config = generate_sim_config(
            sim_method,
            atmos_model,
            initial_state=[0, 0, 0, 0],
            time_step=0.1,
            sim_method_kwargs=sim_method_kwargs,
            atmos_kwargs=atmos_kwargs,
        )
        self.integration_sim = Simulator(integration_sim_config)
        self.atmos = AtmosphereModel(integration_sim_config.atmosphere_model_kwargs)

        try:
            self.atmos.derivative(
                np.array((EARTH_RADIUS + 150000, 0, 0, 7820)),
                0,
            )
        except NotImplementedError:
            raise ValueError("Atmosphere model must have a derivative method")

        self.estimated_trajectory = []
        self.times = []

    @property
    def dt(self):
        return self.integration_sim.time_step

    @dt.setter
    def dt(self, value):
        self.integration_sim.time_step = value 

    @staticmethod
    def compute_jacobian(state, time, atmos: AtmosphereModel):#MATHS mayb be a bit iffy
        x, y, z, vx, vy, vz = state
        r = np.linalg.norm([x, y, z])
        v = np.linalg.norm([vx, vy, vz])
        rho = atmos.density([x, y, z], time)
        drho_dx, drho_dy, drho_dz = atmos.derivative([x, y, z], time)  # Assume this correctly calculates partial derivatives

        drag_consts = (1 / (2 * SATELLITE_MASS)) * MEAN_DRAG_COEFF * MEAN_XSECTIONAL_AREA
        GM_over_r3 = GM_EARTH / r**3

        jacobian = np.zeros((6, 6))

        # Position derivatives affect velocity updates
        jacobian[3, 0] = GM_over_r3 * (-1 + 3 * (x/r)**2) - drag_consts * (drho_dx * vx**2 / v + rho * 2 * vx / v - rho * vx**3 / v**3)
        jacobian[4, 1] = GM_over_r3 * (-1 + 3 * (y/r)**2) - drag_consts * (drho_dy * vy**2 / v + rho * 2 * vy / v - rho * vy**3 / v**3)
        jacobian[5, 2] = GM_over_r3 * (-1 + 3 * (z/r)**2) - drag_consts * (drho_dz * vz**2 / v + rho * 2 * vz / v - rho * vz**3 / v**3)

        # Velocity derivatives (simplified)
        jacobian[3, 3] = -drag_consts * rho * (2 * vx / v - vx**3 / v**3)
        jacobian[4, 4] = -drag_consts * rho * (2 * vy / v - vy**3 / v**3)
        jacobian[5, 5] = -drag_consts * rho * (2 * vz / v - vz**3 / v**3)

        # State transition
        jacobian[0, 3] = jacobian[1, 4] = jacobian[2, 5] = 1

        return jacobian

    def run(self, observations, dt, Q, R, P, H):
        """Runs the Extended Kalman Filter on the given observations.

        Args:
            observations (NDArray): A tuple of (observations, measurement_times)
            dt (float): Time step for the Kalman Filter simulation
            Q (NDArray): Process noise matrix with shape (4, 4)
            R (NDArray): Measurement noise matrix with shape (4, 4) or (N, 4, 4) where N is the number of measurements
            P (NDArray): Initial state covariance matrix
            H (NDArray): Measurement matrix
            integration_sim_config (SimConfig | None, optional): Simulator config for the internal simulation engine. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.dt = dt

        # We set up a simulator for stepping the states. The simulator state that persists
        # is the atmosphere model and time step.
        if R.ndim == 2:
            R_mat = R

        # Estimated trajectories
        measurements, measurement_times = observations

        # Generator for the EKF time steps
        EKF_times = count(measurement_times[0], self.dt)

        estimated_trajectory = [measurements[0]]
        uncertainties = []

        j = 0  # to filter through measurement array at different rate

        # Extended Kalman Filter
        pbar = tqdm(total=len(measurement_times))
        for t in EKF_times:
            if j < len(measurements) and np.abs(measurement_times[j] - t) < self.dt / 2:
                if R.ndim == 3:
                    R_mat = R[j]
                x_hat, P = self.next_state(
                    estimated_trajectory[-1],
                    t,
                    Q,
                    P,
                    H,
                    observation=measurements[j],
                    R=R_mat,
                )
                # Count a measurement and move to the next measurement that is in the future.
                # NB: This may skip measurements.
                j = np.argmax(measurement_times > t + self.dt / 2)
                pbar.update(1)  # Progress bar
            else:
                x_hat, P = self.next_state(estimated_trajectory[-1], t, Q, P, H)

            estimated_trajectory.append(x_hat)
            uncertainties.append(P)

            if self.integration_sim.is_terminal(estimated_trajectory[-1]):
                break
        pbar.close()

        times = np.array(
            [EKF_times.__next__() for _ in range(len(estimated_trajectory))]
        )
        uncertainties = np.array(uncertainties)
        return np.array(estimated_trajectory), uncertainties, times