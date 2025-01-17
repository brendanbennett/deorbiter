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
    """Extended Kalman Filter implementation for the prediction of a satellite's trajectory.

    :keyword dim: The dimension of the simulation. Defaults to 2.
    :keyword sim_method: The numerical method to use for the simulation. Defaults to "RK4".
    :keyword atmos_model: The atmosphere model to use for the simulation. Defaults to "coesa_atmos_fast".
    :keyword sim_method_kwargs: Additional keyword arguments for the numerical method. Defaults to None.
    :keyword atmos_kwargs: Additional keyword arguments for the atmosphere model. Defaults to None.
    """

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
    def dt(self) -> float:
        return self.integration_sim.time_step

    @dt.setter
    def dt(self, value: float):
        self.integration_sim.time_step = value

    @property
    def dim(self) -> int:
        return self.integration_sim.dim

    @staticmethod
    def compute_jacobian(state, time, accel, atmos: AtmosphereModel):
        """
        Compute the Jacobian matrix for the Extended Kalman Filter (EKF) state transition.

        :param state: The current state vector of the system, including positions and velocities.
        :type state: np.ndarray
        :param time: The current time of the system.
        :type time: float
        :param accel: The current acceleration vector of the system.
        :type accel: np.ndarray
        :param atmos: The atmosphere model used to compute density and its derivative.
        :type atmos: AtmosphereModel
        :return: The Jacobian matrix of the state transition.
        :rtype: np.ndarray

        The Jacobian matrix is calculated based on the dimension of the state vector. The function supports
        both 2D and 3D simulations, where the state vector has 4 or 6 elements, respectively.

        - For 2D, the state vector is [x, y, x_dot, y_dot].
        - For 3D, the state vector is [x, y, z, x_dot, y_dot, z_dot].

        The function accounts for the gravitational force and atmospheric drag force in the Jacobian matrix.
        The atmospheric density and its derivative are computed using the provided atmosphere model.
        """
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
            # jacobian[0, 0] = x_dot_dot / x_dot
            # jacobian[0, 1] = x_dot_dot / y_dot
            # jacobian[1, 0] = y_dot_dot / x_dot
            # jacobian[1, 1] = y_dot_dot / y_dot

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
            # jacobian[0, 0] = x_dot_dot / x_dot
            # jacobian[0, 1] = x_dot_dot / y_dot
            # jacobian[0, 2] = x_dot_dot / z_dot
            # jacobian[1, 0] = y_dot_dot / x_dot
            # jacobian[1, 1] = y_dot_dot / y_dot
            # jacobian[1, 2] = y_dot_dot / z_dot
            # jacobian[2, 0] = z_dot_dot / x_dot
            # jacobian[2, 1] = z_dot_dot / y_dot
            # jacobian[2, 2] = z_dot_dot / z_dot

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

    def next_state(
        self, state, time, Q, P, observation=None, H=None, R=None, dt=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the next state of the system using the Extended Kalman Filter.

        .. note:: The measurement matrix and measurement noise matrix need to be the same dimensions in domain and range as the state vector.
            This is a shortcoming of the current implementation.
        
        :param state: The current state of the system
        :param time: The current time of the system
        :param Q: Process noise matrix with shape (2*dim, 2*dim)
        :param P: Initial state covariance matrix with shape (2*dim, 2*dim)
        :param observation: The observation at the current time. Optional.
        :param H: Measurement matrix. Required if observation is not None.
        :param R: Measurement noise matrix. Required if observation is not None.
        :param dt: Time step for the Kalman Filter simulation. If None, the simulator's default is used.
        :return: Tuple of the next state of the system and the updated state covariance matrix.
        """
        if dt is not None:
            self.dt = dt
        if observation is not None and np.any((R is None, H is None)):
            raise ValueError("If observation is not None, R and H must be provided")
        
        if np.any([i is not None and i.shape != (2 * self.dim, 2 * self.dim) for i in [Q, P, H, R]]):
            raise ValueError(f"Kalman matrices not same dimensions, should be {2 * self.dim} by {2 * self.dim}")
            
        accel = self.integration_sim._calculate_accel(state, time)
        # EKF Prediction
        F_t = self.compute_jacobian(state, time, accel, self.atmos)
        Phi_t: np.ndarray = np.eye(2 * self.dim) + F_t * self.dt

        x_hat_minus = self.integration_sim._next_state(state, time)
        P_minus = Phi_t @ P @ Phi_t.T + Q
        if np.any(np.isnan(P_minus)):
            print(f"{P=}, {Phi_t=}, {P_minus=}")
            raise ValueError("P_minus is NaN")

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

    def run(
        self, observations, dt, Q, R, P, H, steps=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs the Extended Kalman Filter on the given observations.

        Args:
            observations (NDArray): A tuple of (observations, measurement_times)
            dt (float): Time step for the Kalman Filter simulation
            Q (NDArray): Process noise matrix with shape (4, 4) or (6, 6) etc
            R (NDArray): Measurement noise matrix with shape (4, 4) or (N, 4, 4) where N is the number of measurements
            P (NDArray): Initial state covariance matrix. Unused
            H (NDArray): Measurement matrix
            steps (int, optional): Number of steps to run the EKF. By default, it runs until the satellite crashes.
            integration_sim_config (SimConfig | None, optional): Simulator config for the internal simulation engine. Defaults to None.

        :return: The estimated trajectory, uncertainties, and times
        """
        self.dt = dt

        # We set up a simulator for stepping the states. The simulator state that persists
        # is the atmosphere model and time step.
        if R.ndim == 2:
            R_mat = R

        # Estimated trajectories
        measurements, measurement_times = observations
        
        # Generator for the EKF time steps. 
        # When the EKF runs, this starts at the second time step, since the first is at the initial state.
        EKF_times = count(measurement_times[0], self.dt)
        EKF_times_run = [next(EKF_times)]
        if steps is not None:
            EKF_times = [next(EKF_times) for _ in range(steps)]

        # Initial state and uncertainty from the first measurement
        estimated_trajectory = [measurements[0]]
        uncertainties = [R_mat] if R.ndim == 2 else [R[0]]
        P = uncertainties[0]
        
        # to filter through measurement array at different rate
        # We start at the second measurement so the first is used as the initial state
        j = 1  

        # Extended Kalman Filter
        pbar = tqdm(total=len(measurement_times))
        pbar.update(1)
        for t in EKF_times:
            EKF_times_run.append(t)
            if j < len(measurements) and np.abs(measurement_times[j] - t) < self.dt / 2:
                if R.ndim == 3:
                    R_mat = R[j]
                x_hat, P = self.next_state(
                    estimated_trajectory[-1],
                    t,
                    Q,
                    P,
                    observation=measurements[j],
                    H=H,
                    R=R_mat,
                )
                # Count a measurement and move to the next measurement that is in the future.
                # NB: This may skip measurements.
                j = np.argmax(measurement_times > t + self.dt / 2)
                pbar.update(1)  # Progress bar
            else:
                x_hat, P = self.next_state(estimated_trajectory[-1], t, Q, P)

            estimated_trajectory.append(x_hat)
            uncertainties.append(P)

            if self.integration_sim.is_terminal(estimated_trajectory[-1]):
                break
        pbar.close()

        times = np.array(EKF_times_run)
        uncertainties = np.array(uncertainties)
        return np.array(estimated_trajectory), uncertainties, times


class EKFOnline:
    """Extended Kalman Filter implementation for the real time prediction of a satellite's trajectory.
    This uses the logic of the :class:`EKF` class but is allows one to step through the prediction process one step at a time.

    :param ekf: The :class:`EKF` object to use for the prediction
    :param initial_state: The initial state of the system
    :param initial_time: The initial time of the system
    :param initial_uncertainty: The initial state covariance matrix

    .. warning:: The initial state must not have a zero velocity component. This is due a shortcoming of the EKF Jacobian calculation.
    """

    def __init__(
        self, ekf: EKF, initial_state, initial_time, initial_uncertainty
    ) -> None:
        self.ekf = ekf
        self.estimated_trajectory = [initial_state]
        self.estimated_times = [initial_time]
        self.uncertainties = [initial_uncertainty]

    def step(self, time, Q, observation=None, R=None, H=None) -> None:
        dt = time - self.estimated_times[-1]
        self.estimated_times.append(time)
        x_hat, P = self.ekf.next_state(
            self.estimated_trajectory[-1],
            time,
            Q,
            self.uncertainties[-1],
            observation=observation,
            H=H,
            R=R,
            dt=dt,
        )
        self.estimated_trajectory.append(x_hat)
        self.uncertainties.append(P)
