import numpy as np
from deorbit.utils.constants import (
    GM_EARTH,
    MEAN_DRAG_COEFF,
    MEAN_XSECTIONAL_AREA,
    SATELLITE_MASS,
)
from deorbit.simulator import Simulator, generate_sim_config
from deorbit.simulator.simulator import RK4Simulator
from tqdm import tqdm


# func to compute the Jacobian matrix dynamically
def compute_jacobian(state, atmos, time, args):
    x_dot_dot, y_dot_dot = args
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
        3 * GM_EARTH * r ** (-5) * x**2
        - GM_EARTH * r ** (-3)
        - (drag_consts * (x_dot**2 * drho_dx + 2 * rho * x_dot_dot))
    )
    jacobian[3, 1] = (
        3 * GM_EARTH * r ** (-5) * y**2
        - GM_EARTH * r ** (-3)
        - (drag_consts * (y_dot**2 * drho_dy + 2 * rho * y_dot_dot))
    )
    # TODO: Still to fix
    jacobian[3, 0] = GM_EARTH * 3 * x * y * r ** (-5) - drag_consts * (
        drho_dx * y_dot**2 + 2 * y_dot * y_dot_dot / x_dot
    )
    jacobian[2, 1] = GM_EARTH * 3 * x * y * r ** (-5) - drag_consts * (
        drho_dy * x_dot**2 + 2 * x_dot * x_dot_dot / y_dot
    )

    jacobian[2, 2] = -drag_consts * rho * x_dot
    jacobian[3, 3] = -drag_consts * rho * y_dot

    # print(f"{time=}")
    # print(jacobian)
    # print(
    #     f"{x=}, {y=}, {x_dot=}, {y_dot=}, {r=}, {rho=}, {drho_dx=}, {drho_dy=}, {x_dot_dot=}, {y_dot_dot=}"
    # )

    return jacobian


def EKF(simulation_data, config, atmos, dt, Q, R, P, H):
    # Define simulation parameters from the config
    # dt = sim_config.simulation_method_kwargs.time_step

    sim = Simulator(config)

    num_steps = len(
        simulation_data.times
    )  # Number of steps based on simulation data

    # Initialize state estimate
    initial_state = np.array(
        [
            simulation_data.x1[0],
            simulation_data.x2[0],
            simulation_data.v1[0],
            simulation_data.v2[0],
        ]
    )
    x_hat = initial_state + np.random.multivariate_normal([0, 0, 0, 0], R)

    # We set up a simulator for stepping the states. The simulator state that persists 
    # is the atmosphere model and time step.
    integration_config = generate_sim_config(
        "RK4",
        "coesa_atmos_fast",
        initial_state=initial_state,
        time_step=dt,
    )
    integration_sim: RK4Simulator = Simulator(integration_config)

    # Estimated trajectories
    # true_trajectory = [initial_state] #dont think need this
    estimated_trajectory = [x_hat]
    measurements = [x_hat]

    # Extended Kalman Filter
    for i in tqdm(range(1, num_steps)):
        # True state (from simulation, for plotting/comparison purposes only)
        true_state = np.array(
            [
                simulation_data.x1[i],
                simulation_data.x2[i],
                simulation_data.v1[i],
                simulation_data.v2[i],
            ]
        )
        # true_trajectory.append(true_state) #dont think need this

        # Noisy measurement
        measurement_noise = np.random.multivariate_normal([0, 0, 0, 0], R)

        measurement = true_state + measurement_noise

        # reducing number of measurements for experimentation if want
        # if i % 100 ==0:
        measurements.append(measurement)
        # print(f"x_i = {estimated_trajectory[-1]}")
        args = sim._calculate_accel(
            estimated_trajectory[-1], simulation_data.times[i]
        )
        # EKF Prediction
        F_t = compute_jacobian(
            estimated_trajectory[-1], atmos, simulation_data.times[i], args
        )
        x_hat_minus = integration_sim._next_state_RK4(estimated_trajectory[-1], simulation_data.times[i])
        P_minus = np.dot(F_t, np.dot(P, F_t.T)) + Q

        # EKF Update
        K = np.dot(
            P_minus,
            np.dot(H.T, (np.linalg.inv(np.dot(H, np.dot(P_minus, H.T)) + R))),
        )
        x_hat = x_hat_minus + np.dot(K, (measurement - np.dot(H, x_hat_minus)))
        P = np.dot((np.eye(4) - np.dot(K, H)), P_minus)

        estimated_trajectory.append(x_hat)

    return np.array(estimated_trajectory), np.array(measurements)
