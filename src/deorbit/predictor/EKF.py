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


class EKF:
    def __init__(self, **kwargs):
        dim = kwargs.get("dim", 2)
        sim_method = kwargs.get("sim_method", "RK4")
        atmos_model = kwargs.get("atmos_model", "coesa_atmos_fast")
        sim_method_kwargs = kwargs.get("sim_method_kwargs", None)
        atmos_kwargs = kwargs.get("atmos_kwargs", None)

        if dim not in [2, 3]:
            raise NotImplementedError("Simulations have to be 2D or 3D")

        integration_sim_config = generate_sim_config(
            sim_method,
            atmos_model,
            initial_state=np.zeros(dim * 2),
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

    @property
    def dim(self):
        return self.integration_sim.dim

    @staticmethod
    def compute_jacobian(state, time, accel, atmos: AtmosphereModel):
        #calculate dimension for jacobian
        dim = len(state)/2

        jacobian = np.zeros((len(state), len(state)))
        rho = atmos.density(state, time)
        
        drag_consts = (
            (1 / (2 * SATELLITE_MASS)) * MEAN_DRAG_COEFF * MEAN_XSECTIONAL_AREA
        )

        # func to compute the Jacobian matrix dynamically
        if dim == 2:
            x_dot_dot, y_dot_dot = accel
            x, y, x_dot, y_dot = state
        
            r = np.linalg.norm((x, y))

            drho_dx = atmos.derivative(state, 0) * (x / r)
            drho_dy = atmos.derivative(state, 0) * (y / r)

            # State transition Jacobian part
            
            # PROBLEM: x_dot and y_dot can be zero, which will cause a division by zero error,
            # leading to P being NaN
            jacobian[0, 0] = x_dot_dot / x_dot
            jacobian[0, 1] = x_dot_dot / y_dot
            jacobian[1, 0] = y_dot_dot / x_dot
            jacobian[1, 1] = y_dot_dot / y_dot

            jacobian[0, 2] = 1
            jacobian[1, 3] = 1

            jacobian[2, 0] = GM_EARTH * r ** (-5) * (3 * x**2 - r ** (2)) - (
                drag_consts * (x_dot**2 * drho_dx + 2 * rho * x_dot_dot)
            )   

            jacobian[3, 1] = GM_EARTH * r ** (-5) * (3 * y**2 - r ** (2)) - (
                drag_consts * (y_dot**2 * drho_dy + 2 * rho * y_dot_dot)
            )

            jacobian[3, 0] = GM_EARTH * 3 * x * y * r ** (-5) - drag_consts * (
                drho_dx * y_dot**2 + 2 * rho * y_dot * y_dot_dot / x_dot
            )
            jacobian[2, 1] = GM_EARTH * 3 * x * y * r ** (-5) - drag_consts * (
                drho_dy * x_dot**2 + 2 * rho * x_dot * x_dot_dot / y_dot
            )

            jacobian[2, 2] = -drag_consts * rho * x_dot
            jacobian[3, 3] = -drag_consts * rho * y_dot

        if dim ==3:
            x_dot_dot, y_dot_dot, z_dot_dot = accel
            x, y, z, x_dot, y_dot, z_dot= state
        
            r = np.linalg.norm((x, y, z))

            drho_dx = atmos.derivative(state, 0) * (x / r)
            drho_dy = atmos.derivative(state, 0) * (y / r)
            drho_dz = atmos.derivative(state, 0) * (z / r)

            # State transition Jacobian part
            jacobian[0, 0] = x_dot_dot / x_dot
            jacobian[0, 1] = x_dot_dot / y_dot
            jacobian[0, 2] = x_dot_dot / z_dot
            jacobian[1, 0] = y_dot_dot / x_dot
            jacobian[1, 1] = y_dot_dot / y_dot
            jacobian[1, 2] = y_dot_dot / z_dot
            jacobian[2, 0] = z_dot_dot / x_dot
            jacobian[2, 1] = z_dot_dot / y_dot
            jacobian[2, 2] = z_dot_dot / z_dot

            jacobian[0, 3] = 1
            jacobian[1, 4] = 1
            jacobian[2, 5] = 1

            jacobian[3, 0] = GM_EARTH * r ** (-5) * (3 * x**2 - r ** (2)) - (
                drag_consts * (x_dot**2 * drho_dx + 2 * rho * x_dot_dot)
            )   

            jacobian[4, 1] = GM_EARTH * r ** (-5) * (3 * y**2 - r ** (2)) - (
                drag_consts * (y_dot**2 * drho_dy + 2 * rho * y_dot_dot)
            )

            jacobian[5, 2] = GM_EARTH * r ** (-5) * (3 * z**2 - r ** (2)) - (
                drag_consts * (z_dot**2 * drho_dz + 2 * rho * z_dot_dot)
            )

            jacobian[4, 0] = GM_EARTH * 3 * x * y * r ** (-5) - drag_consts * (
                drho_dx * y_dot**2 + 2 * rho * y_dot * y_dot_dot / x_dot
            )

            jacobian[5, 0] = GM_EARTH * 3 * x * z * r ** (-5) - drag_consts * (
                drho_dx * z_dot**2 + 2 * rho * z_dot * z_dot_dot / x_dot
            )

            jacobian[3, 1] = GM_EARTH * 3 * x * y * r ** (-5) - drag_consts * (
                drho_dy * x_dot**2 + 2 * rho * x_dot * x_dot_dot / y_dot
            )

            jacobian[5, 1] = GM_EARTH * 3 * z * y * r ** (-5) - drag_consts * (
                drho_dy * z_dot**2 + 2 * rho * z_dot * z_dot_dot / y_dot
            )

            jacobian[3, 2] = GM_EARTH * 3 * x * z * r ** (-5) - drag_consts * (
                drho_dz * x_dot**2 + 2 * rho * x_dot * x_dot_dot / z_dot
            )

            jacobian[4, 2] = GM_EARTH * 3 * z * y * r ** (-5) - drag_consts * (
                drho_dz * y_dot**2 + 2 * rho * y_dot * y_dot_dot / z_dot
            )

            jacobian[3, 3] = -drag_consts * rho * x_dot
            jacobian[4, 4] = -drag_consts * rho * y_dot
            jacobian[5, 5] = -drag_consts * rho * z_dot

        return jacobian

    def next_state(self, state, time, Q, P, H=None, dt=None, observation=None, R=None):
        if dt is not None:
            self.dt = dt
        if observation is not None and np.any((R is None, H is None)):
            raise ValueError("If observation is not None, R and H must be provided")
        
        if np.any([i is not None and i.shape != (2 * self.dim, 2 * self.dim) for i in [Q, P, H, R]]):
            print(Q.shape)
            print(P.shape)
            print(H.shape)
            print(R.shape)
            raise ValueError(f"Kalman matrices not same dimensions, should be {2 * self.dim} by {2 * self.dim}")
            
        accel = self.integration_sim._calculate_accel(state, time)
        # EKF Prediction
        F_t = self.compute_jacobian(state, time, accel, self.atmos)
        Phi_t: npt.NDArray = np.eye(2 * self.dim) + F_t * self.dt

        x_hat_minus = self.integration_sim._next_state(state, time)
        P_minus = Phi_t @ P @ Phi_t.T + Q
        if np.any(np.isnan(P_minus)):
            print(f"{P=}, {Phi_t=}, {P_minus=}")

        if observation is not None:
            if R is None:
                raise ValueError(
                    "Measurement noise matrix R must be provided if observation is provided"
                )
            # EKF Update with measurement
            K = P_minus @ H.T @ np.linalg.inv(H @ P_minus @ H.T + R)
            x_hat = x_hat_minus + K @ (observation - H @ x_hat_minus)
            P = (np.eye(2 * self.dim) - K @ H) @ P_minus

        else:
            # EKF Update without measurement
            x_hat = x_hat_minus
            P = P_minus

        return x_hat, P

    def run(self, observations, dt, Q, R, P, H):
        """Runs the Extended Kalman Filter on the given observations.

        Args:
            observations (NDArray): A tuple of (observations, measurement_times)
            dt (float): Time step for the Kalman Filter simulation
            Q (NDArray): Process noise matrix with shape (4, 4) or (6, 6) etc
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


class EKFOnline:
    def __init__(
        self, ekf: EKF, initial_state, initial_time, initial_uncertainty
    ) -> None:
        self.ekf = ekf
        self.estimated_trajectory = [initial_state]
        self.estimated_times = [initial_time]
        self.uncertainties = [initial_uncertainty]

    def step(self, time, Q, observation=None, R=None, H=None):
        dt = time - self.estimated_times[-1]
        self.estimated_times.append(time)
        x_hat, P = self.ekf.next_state(
            self.estimated_trajectory[-1],
            time,
            Q,
            self.uncertainties[-1],
            H,
            dt=dt,
            observation=observation,
            R=R,
        )
        self.estimated_trajectory.append(x_hat)
        self.uncertainties.append(P)
        if np.any(np.isnan(P)):
            print(f"{dt=}")
