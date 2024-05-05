import numpy as np
from deorbit.utils.constants import GM_EARTH

# func to compute the Jacobian matrix dynamically
def compute_jacobian(state, dt):
    x, y, vx, vy = state
    jacobian = np.diag([1, 1, 1, 1])
    
    # State transition Jacobian part
    jacobian[0, 2] = dt
    jacobian[1, 3] = dt
    
    # Gravity model simplified
    r = np.sqrt(x**2 + y**2)
    grav_accel = -4*GM_EARTH / r**3
    jacobian[2, 0] = grav_accel * x * dt / r 
    jacobian[3, 1] = grav_accel * y * dt / r
    
    
    return jacobian


def EKF(simulation_data, dt, Q, R, P, H):
    # Define simulation parameters from the config
    #dt = sim_config.simulation_method_kwargs.time_step

    num_steps = len(simulation_data.times)  # Number of steps based on simulation data

    # Initialize state estimate
    initial_state = np.array([simulation_data.x1[0], simulation_data.x2[0], simulation_data.v1[0], simulation_data.v2[0]])
    x_hat = initial_state + np.random.multivariate_normal([0, 0, 0, 0], R)

    # Estimated trajectories
    #true_trajectory = [initial_state] #dont think need this
    estimated_trajectory = [x_hat]
    measurements = [x_hat]

    # Extended Kalman Filter 
    for i in range(1, num_steps):
        # True state (from simulation, for plotting/comparison purposes only)
        true_state = np.array([simulation_data.x1[i], simulation_data.x2[i], simulation_data.v1[i], simulation_data.v2[i]])
        #true_trajectory.append(true_state) #dont think need this
    
        # Noisy measurement
        measurement_noise = np.random.multivariate_normal([0, 0, 0, 0], R)

        measurement = true_state + measurement_noise

        #reducing number of measurements for experimentation if want
        #if i % 100 ==0: 
        measurements.append(measurement)
    
        # EKF Prediction
        F_t = compute_jacobian(estimated_trajectory[-1], dt)
        x_hat_minus = np.dot(F_t, estimated_trajectory[-1])
        P_minus = np.dot(F_t, np.dot(P, F_t.T)) + Q
    
        # EKF Update
        K = np.dot(P_minus, np.dot(H.T, (np.linalg.inv(np.dot(H, np.dot(P_minus, H.T)) + R))))
        x_hat = x_hat_minus + np.dot(K, (measurement - np.dot(H, x_hat_minus)))
        P = np.dot((np.eye(4) - np.dot(K, H)), P_minus)
    
        estimated_trajectory.append(x_hat)

    return np.array(estimated_trajectory), np.array(measurements)
    

