import numpy as np
from itertools import count
from deorbit.utils.constants import (
    GM_EARTH,
    MEAN_DRAG_COEFF,
    MEAN_XSECTIONAL_AREA,
    SATELLITE_MASS,
)
from deorbit.simulator import Simulator, generate_sim_config
from deorbit.simulator.atmos import AtmosphereModel
from deorbit.simulator.simulator import RK4Simulator
from deorbit.data_models.sim import SimConfig
from tqdm import tqdm


# func to compute the Jacobian matrix dynamically
def compute_jacobian(state, time, accel, atmos):
    x_dot_dot, y_dot_dot = accel
    x, y, x_dot, y_dot = state
    jacobian = np.zeros((len(state), len(state)))

    r = np.linalg.norm((x, y))

    rho = atmos.density(state, time)
    drho_dx = atmos.derivative(state, 0) * (x / r)
    drho_dy = atmos.derivative(state, 0) * (y / r)

    drag_consts = (
        (1 / (2 * SATELLITE_MASS)) * MEAN_DRAG_COEFF * MEAN_XSECTIONAL_AREA
    )

    # State transition Jacobian part
    jacobian[0, 0] = x_dot_dot / x_dot
    jacobian[0, 1] = x_dot_dot / y_dot
    jacobian[1, 0] = y_dot_dot / x_dot
    jacobian[1, 1] = y_dot_dot / y_dot

    jacobian[0, 2] = 1
    jacobian[1, 3] = 1

    jacobian[2, 0] = (
        GM_EARTH * r**(-5) * (3 * x**2 - r**(2))
        - (drag_consts * (x_dot**2 * drho_dx + 2 * rho * x_dot_dot))
    )
    
    jacobian[3, 1] = (
        GM_EARTH * r**(-5) * (3 * y**2 - r**(2))
        - (drag_consts * (y_dot**2 * drho_dy + 2 * rho * y_dot_dot))
    )
    
    jacobian[3, 0] = GM_EARTH * 3 * x * y * r ** (-5) - drag_consts * (
        drho_dx * y_dot**2 + 2 * rho * y_dot * y_dot_dot / x_dot
    )
    jacobian[2, 1] = GM_EARTH * 3 * x * y * r ** (-5) - drag_consts * (
        drho_dy * x_dot**2 + 2 * rho * x_dot * x_dot_dot / y_dot
    )

    jacobian[2, 2] = -drag_consts * rho * x_dot
    jacobian[3, 3] = -drag_consts * rho * y_dot

    # print(f"{time=}")
    # print(jacobian)
    # print(
    #     f"{x=}, {y=}, {x_dot=}, {y_dot=}, {r=}, {rho=}, {drho_dx=}, {drho_dy=}, {x_dot_dot=}, {y_dot_dot=}"
    # )

    return jacobian


def EKF(observations, dt, Q, R, P, H, integration_sim_config: SimConfig | None = None):
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
    # Define simulation parameters from the config
    # dt = config.simulation_method_kwargs.time_step
    
    # We set up a simulator for stepping the states. The simulator state that persists 
    # is the atmosphere model and time step.
    if R.ndim == 2:
        R_mat = R
    
    if integration_sim_config is None:
        integration_config = generate_sim_config(
            "RK4",
            "coesa_atmos_fast",
            initial_state=[0, 0, 0, 0],
            time_step=dt,
        )
    else:
        integration_config = integration_sim_config
    integration_sim: RK4Simulator = Simulator(integration_config)
    atmos = AtmosphereModel(integration_config.atmosphere_model_kwargs)

    # Estimated trajectories
    measurements, measurement_times = observations
    
    EKF_times = count(measurement_times[0], dt)

    estimated_trajectory = [measurements[0]]

    j = 0 #too filter through measurement array at different rate

    # Extended Kalman Filter
    pbar = tqdm(total=len(measurement_times))
    for t in EKF_times:

        # print(f"x_i = {estimated_trajectory[-1]}")
        accel = integration_sim._calculate_accel(
            estimated_trajectory[-1], t
        )
        # EKF Prediction
        F_t = compute_jacobian(
            estimated_trajectory[-1], t, accel, atmos
        )

        x_hat_minus = integration_sim._next_state_RK4(estimated_trajectory[-1], t)
        P_minus = F_t @ P @ F_t.T + Q

        if j < len(measurements) and np.abs(measurement_times[j] - t) < dt/2:
            if R.ndim == 3:
                R_mat = R[j]
                
            # Noisy measurement
            measurement = measurements[j]

            # EKF Update with measurement
            K = P_minus @ H.T @ np.linalg.inv(H @ P_minus @ H.T + R_mat)
            x_hat = x_hat_minus + K @ (measurement - H @ x_hat_minus)
            P = (np.eye(4) - K @ H) @ P_minus

            j += 1
            pbar.update(1)

        else:
            # EKF Update without measurement
            x_hat = x_hat_minus
            P = P_minus


        estimated_trajectory.append(x_hat)

        if integration_sim.is_terminal(estimated_trajectory[-1]):
            break
    pbar.close()

    times = np.array([EKF_times.__next__() for _ in range(len(estimated_trajectory))])
    return np.array(estimated_trajectory), times
