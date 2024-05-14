# deorbit.predictor package

## Submodules

## deorbit.predictor.EKF module

### *class* deorbit.predictor.EKF.EKF(\*\*kwargs)

Bases: `object`

#### *static* compute_jacobian(state, time, accel, atmos: [AtmosphereModel](deorbit.simulator.md#deorbit.simulator.atmos.AtmosphereModel))

#### *property* dim

#### *property* dt

#### next_state(state, time, Q, P, H=None, dt=None, observation=None, R=None)

#### run(observations, dt, Q, R, P, H)

Runs the Extended Kalman Filter on the given observations.

* **Parameters:**
  * **observations** (*NDArray*) – A tuple of (observations, measurement_times)
  * **dt** (*float*) – Time step for the Kalman Filter simulation
  * **Q** (*NDArray*) – Process noise matrix with shape (4, 4) or (6, 6) etc
  * **R** (*NDArray*) – Measurement noise matrix with shape (4, 4) or (N, 4, 4) where N is the number of measurements
  * **P** (*NDArray*) – Initial state covariance matrix
  * **H** (*NDArray*) – Measurement matrix
  * **integration_sim_config** ([*SimConfig*](deorbit.data_models.md#deorbit.data_models.sim.SimConfig) *|* *None* *,* *optional*) – Simulator config for the internal simulation engine. Defaults to None.
* **Returns:**
  \_description_
* **Return type:**
  \_type_

### *class* deorbit.predictor.EKF.EKFOnline(ekf: [EKF](#deorbit.predictor.EKF.EKF), initial_state, initial_time, initial_uncertainty)

Bases: `object`

#### step(time, Q, observation=None, R=None, H=None)

## deorbit.predictor.EKF_class module

### *class* deorbit.predictor.EKF_class.EKF(H, P, Q, R)

Bases: `object`

Extended Kalman Filter (EKF) class for state estimation.

This class implements the Extended Kalman Filter (EKF) algorithm for
state estimation in a non-linear system. It takes simulation data,
configuration, atmosphere model, time step, process noise covariance,
measurement noise covariance, initial error covariance, and measurement
matrix as input and returns the estimated trajectory and measurements.

#### H

Measurement matrix.

* **Type:**
  np.ndarray

#### P

Error covariance matrix.

* **Type:**
  np.ndarray

#### Q

Process noise covariance matrix.

* **Type:**
  np.ndarray

#### R

Measurement noise covariance matrix.

* **Type:**
  np.ndarray

#### sim

Simulator object for state propagation (initialized later).

* **Type:**
  [Simulator](deorbit.simulator.md#deorbit.simulator.simulator.Simulator)

#### *static* compute_jacobian(state, time, accel, atmos)

Computes the Jacobian matrix of the system dynamics.

This static method calculates the Jacobian matrix of the system
dynamics with respect to the state vector.

* **Parameters:**
  * **state** (*list*) – Current state vector.
  * **time** (*float*) – Current time.
  * **accel** (*list*) – Acceleration vector.
  * **atmos** ([*AtmosphereModel*](deorbit.simulator.md#deorbit.simulator.atmos.AtmosphereModel)) – Atmosphere model object.
* **Returns:**
  Jacobian matrix.
* **Return type:**
  np.ndarray

#### run_ekf(simulation_data: [SimData](deorbit.data_models.md#deorbit.data_models.sim.SimData), config: [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig), atmos, dt, observations)

Runs the Extended Kalman Filter algorithm.

This method performs the EKF estimation for the given simulation
data, configuration, atmosphere model, and time step.

* **Parameters:**
  * **simulation_data** ([*SimData*](deorbit.data_models.md#deorbit.data_models.sim.SimData)) – Simulation data object.
  * **config** ([*SimConfig*](deorbit.data_models.md#deorbit.data_models.sim.SimConfig)) – Simulation configuration object.
  * **atmos** ([*AtmosphereModel*](deorbit.simulator.md#deorbit.simulator.atmos.AtmosphereModel)) – Atmosphere model object.
  * **dt** (*float*) – Time step.
  * **observations** (*tuple*) – A tuple containing measurements and their corresponding times.
* **Returns:**
  A tuple containing the estimated trajectory and measurements.
* **Return type:**
  tuple

## Module contents
