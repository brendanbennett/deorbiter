# deorbit.simulator package

## Submodules

## deorbit.simulator.atmos module

### *class* deorbit.simulator.atmos.AtmosphereModel(kwargs: [AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs))

Bases: `ABC`

Abstract base class for Atmosphere model implementations.
.. attribute:: kwargs

> A pydantic data model of model parameters

> * **type:**
>   AtmosKwargs

#### density(state, time)

abstract; must be implemented in any subclass

#### *abstract* density(state: ndarray, time: float)

#### derivative(state: ndarray, time: float)

#### plot(height_bounds_meters: tuple[float, float], num_points: int = 100, ax: Axes = None, label: str = None, derivative: bool = False)

#### velocity(state: ndarray, time: float)

Calculate the velocity of the atmosphere as a result of the Earth’s rotation at a given state (and time)

* **Parameters:**
  * **state** (*np.ndarray*) – The state of the object in the atmosphere
  * **time** (*float*) – The time at which the velocity is calculated
* **Returns:**
  The velocity of the atmosphere at the given state and time
* **Return type:**
  np.ndarray

### *class* deorbit.simulator.atmos.CoesaAtmos(kwargs: [AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs))

Bases: [`AtmosphereModel`](#deorbit.simulator.atmos.AtmosphereModel)

#### density(state: ndarray, time: float)

### *class* deorbit.simulator.atmos.CoesaAtmosFast(kwargs: [AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs))

Bases: [`AtmosphereModel`](#deorbit.simulator.atmos.AtmosphereModel)

Uses a lookup table of atmosphere densities

#### density(state: ndarray, time: float)

#### derivative(state: ndarray, time: float)

### *class* deorbit.simulator.atmos.IcaoAtmos(kwargs: [AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs))

Bases: [`AtmosphereModel`](#deorbit.simulator.atmos.AtmosphereModel)

#### density(state: ndarray, time: float)

### *class* deorbit.simulator.atmos.SimpleAtmos(kwargs: [AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs))

Bases: [`AtmosphereModel`](#deorbit.simulator.atmos.AtmosphereModel)

Generate simple atmospheric model

### density(state

np.ndarray, time: float) -> float: Density function taking state and time as input

#### model_kwargs()

Returns model parameters

#### density(state: ndarray, time: float)

### *class* deorbit.simulator.atmos.ZeroAtmos(kwargs: [AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs))

Bases: [`AtmosphereModel`](#deorbit.simulator.atmos.AtmosphereModel)

Generate zero atmospheric model

### density(state

np.ndarray, time: float) -> float: Density function taking state and time as input

#### model_kwargs()

Returns model parameters

#### density(state: ndarray, time: float)

### deorbit.simulator.atmos.get_available_atmos_models()

Find available atmosphere models in atmos.py

* **Returns:**
  Dictionary of {model name: subclass of AtmosphereModel}
* **Return type:**
  dict[str, type[[AtmosphereModel](#deorbit.simulator.atmos.AtmosphereModel)]]

### deorbit.simulator.atmos.raise_for_invalid_atmos_model(atmos_model: str)

Raises ValueError if the given simulation method name is not defined

## deorbit.simulator.simulator module

### *class* deorbit.simulator.simulator.AdamsBashforthSimulator(config: [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig))

Bases: [`Simulator`](#deorbit.simulator.simulator.Simulator)

### *class* deorbit.simulator.simulator.EulerSimulator(config: [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig))

Bases: [`Simulator`](#deorbit.simulator.simulator.Simulator)

### *class* deorbit.simulator.simulator.RK4Simulator(config: [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig))

Bases: [`Simulator`](#deorbit.simulator.simulator.Simulator)

### *class* deorbit.simulator.simulator.Simulator(config: [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig))

Bases: `ABC`

Base class for simulators.

#### states

List of state vectors.

* **Type:**
  list

#### times

List of simulation times.

* **Type:**
  list

#### \_atmosphere_model

Atmosphere model used for the simulation.

* **Type:**
  Callable

#### sim_method_kwargs

Configuration for the simulation method.

* **Type:**
  [MethodKwargs](deorbit.data_models.md#deorbit.data_models.methods.MethodKwargs)

#### export_config(self)

Returns a configuration object for the simulation.

### set_initial_conditions(self, state

np.ndarray, time: float): Sets the initial conditions for the simulation.

### set_atmosphere_model(self, model_kwargs

AtmosKwargs): Sets the atmosphere model for the simulation.

### atmosphere(self, state

np.ndarray, time: float) -> float: Calculates the atmosphere density at a given state and time.

### run(self, steps

int = None): Runs the simulation for a specified number of steps.

#### gather_data(self)

Generates a data object containing the simulation data and configuration.

### save_data(self, save_dir_path

str) -> Path: Saves the simulation data to a specified directory.

Examples:

```
``
```

\`
config = deorbit.data_models.sim.SimConfig(

> initial_state=(deorbit.constants.EARTH_RADIUS + 100000, 0, 0, 8000),
> simulation_method_kwargs=deorbit.data_models.methods.RK4Kwargs(time_step=0.1),
> atmosphere_model_kwargs=deorbit.data_models.atmos.CoesaFastKwargs()

)
sim = Simulator(config)
sim.run(150000)

```
``
```

```
`
```

#### atmosphere(state: ndarray, time: float)

#### atmosphere_velocity(state: ndarray, time: float)

#### *property* dim

#### export_config()

* **Returns:**
  Config object which can be used to recreate this simulation
* **Return type:**
  [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig)

#### gather_data()

Generates a portable data object containing all the simulation data reqiured to save.

* **Returns:**
  pydantic data model containing both simulated data and config.
* **Return type:**
  [SimData](deorbit.data_models.md#deorbit.data_models.sim.SimData)

#### is_terminal(state: ndarray)

#### *property* noise_types *: dict[str, [NoiseKwargs](deorbit.data_models.md#deorbit.data_models.noise.NoiseKwargs)]*

Returns a dictionary of noise types and their parameters. Empty dict if no noise is present.

* **Returns:**
  Dictionary of noise types and their kwargs models.
* **Return type:**
  dict[str, [NoiseKwargs](deorbit.data_models.md#deorbit.data_models.noise.NoiseKwargs)]

#### run(steps: int = None)

#### save_data(save_dir_path: str, overwrite: bool = True, format: str = 'json')

Saves simulation data to [save_dir_path] directory as defined in the SimData data model.

File name format: sim_data_[unix time in ms].json

* **Parameters:**
  **save_dir_path** (*Path like*) – Data directory to save json file.

#### set_atmosphere_model(model_kwargs: [AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs))

#### set_initial_conditions(state: ndarray, time: float)

Resets the simulation and initialises values with the given state vector and time

#### *property* time_step

### deorbit.simulator.simulator.generate_sim_config(sim_method: str, atmos_model: str, initial_state: \_SupportsArray[dtype[Any]] | \_NestedSequence[\_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | \_NestedSequence[bool | int | float | complex | str | bytes], initial_time: float = 0.0, time_step: float = 0.1, noise_types: dict[str, dict | [NoiseKwargs](deorbit.data_models.md#deorbit.data_models.noise.NoiseKwargs)] | None = None, sim_method_kwargs: dict | type[[MethodKwargs](deorbit.data_models.md#deorbit.data_models.methods.MethodKwargs)] | None = None, atmos_kwargs: dict | type[[AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs)] | None = None)

### deorbit.simulator.simulator.get_available_sim_methods()

Python magic to find the names of implemented simulation methods.

* **Returns:**
  a dictionary of {name: method class}
* **Return type:**
  dict[str, subclass([Simulator](#deorbit.simulator.simulator.Simulator))]

### deorbit.simulator.simulator.raise_for_invalid_noise_type(noise_types: dict[str, dict | [NoiseKwargs](deorbit.data_models.md#deorbit.data_models.noise.NoiseKwargs)] | None)

Raises ValueError if any of the given list of noise types is not defined

### deorbit.simulator.simulator.raise_for_invalid_sim_method(sim_method: str)

Raises ValueError if the given simulation method name is not defined

### deorbit.simulator.simulator.run(sim_method: str, atmos_model: str, initial_state: \_SupportsArray[dtype[Any]] | \_NestedSequence[\_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | \_NestedSequence[bool | int | float | complex | str | bytes], initial_time: float = 0.0, time_step: float = 0.1, noise_types: dict[str, dict | [NoiseKwargs](deorbit.data_models.md#deorbit.data_models.noise.NoiseKwargs)] | None = None, sim_method_kwargs: dict | type[[MethodKwargs](deorbit.data_models.md#deorbit.data_models.methods.MethodKwargs)] | None = None, atmos_kwargs: dict | type[[AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs)] | None = None, steps: int | None = None)

### deorbit.simulator.simulator.run_with_config(config: [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig), steps: int | None = None)

## Module contents

### *class* deorbit.simulator.Simulator(config: [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig))

Bases: `ABC`

Base class for simulators.

#### states

List of state vectors.

* **Type:**
  list

#### times

List of simulation times.

* **Type:**
  list

#### \_atmosphere_model

Atmosphere model used for the simulation.

* **Type:**
  Callable

#### sim_method_kwargs

Configuration for the simulation method.

* **Type:**
  [MethodKwargs](deorbit.data_models.md#deorbit.data_models.methods.MethodKwargs)

#### export_config(self)

Returns a configuration object for the simulation.

### set_initial_conditions(self, state

np.ndarray, time: float): Sets the initial conditions for the simulation.

### set_atmosphere_model(self, model_kwargs

AtmosKwargs): Sets the atmosphere model for the simulation.

### atmosphere(self, state

np.ndarray, time: float) -> float: Calculates the atmosphere density at a given state and time.

### run(self, steps

int = None): Runs the simulation for a specified number of steps.

#### gather_data(self)

Generates a data object containing the simulation data and configuration.

### save_data(self, save_dir_path

str) -> Path: Saves the simulation data to a specified directory.

Examples:

```
``
```

\`
config = deorbit.data_models.sim.SimConfig(

> initial_state=(deorbit.constants.EARTH_RADIUS + 100000, 0, 0, 8000),
> simulation_method_kwargs=deorbit.data_models.methods.RK4Kwargs(time_step=0.1),
> atmosphere_model_kwargs=deorbit.data_models.atmos.CoesaFastKwargs()

)
sim = Simulator(config)
sim.run(150000)

```
``
```

```
`
```

#### atmosphere(state: ndarray, time: float)

#### atmosphere_velocity(state: ndarray, time: float)

#### *property* dim

#### export_config()

* **Returns:**
  Config object which can be used to recreate this simulation
* **Return type:**
  [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig)

#### gather_data()

Generates a portable data object containing all the simulation data reqiured to save.

* **Returns:**
  pydantic data model containing both simulated data and config.
* **Return type:**
  [SimData](deorbit.data_models.md#deorbit.data_models.sim.SimData)

#### is_terminal(state: ndarray)

#### *property* noise_types *: dict[str, [NoiseKwargs](deorbit.data_models.md#deorbit.data_models.noise.NoiseKwargs)]*

Returns a dictionary of noise types and their parameters. Empty dict if no noise is present.

* **Returns:**
  Dictionary of noise types and their kwargs models.
* **Return type:**
  dict[str, [NoiseKwargs](deorbit.data_models.md#deorbit.data_models.noise.NoiseKwargs)]

#### run(steps: int = None)

#### save_data(save_dir_path: str, overwrite: bool = True, format: str = 'json')

Saves simulation data to [save_dir_path] directory as defined in the SimData data model.

File name format: sim_data_[unix time in ms].json

* **Parameters:**
  **save_dir_path** (*Path like*) – Data directory to save json file.

#### set_atmosphere_model(model_kwargs: [AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs))

#### set_initial_conditions(state: ndarray, time: float)

Resets the simulation and initialises values with the given state vector and time

#### *property* time_step

### deorbit.simulator.generate_sim_config(sim_method: str, atmos_model: str, initial_state: \_SupportsArray[dtype[Any]] | \_NestedSequence[\_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | \_NestedSequence[bool | int | float | complex | str | bytes], initial_time: float = 0.0, time_step: float = 0.1, noise_types: dict[str, dict | [NoiseKwargs](deorbit.data_models.md#deorbit.data_models.noise.NoiseKwargs)] | None = None, sim_method_kwargs: dict | type[[MethodKwargs](deorbit.data_models.md#deorbit.data_models.methods.MethodKwargs)] | None = None, atmos_kwargs: dict | type[[AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs)] | None = None)

### deorbit.simulator.get_available_atmos_models()

Find available atmosphere models in atmos.py

* **Returns:**
  Dictionary of {model name: subclass of AtmosphereModel}
* **Return type:**
  dict[str, type[[AtmosphereModel](#deorbit.simulator.atmos.AtmosphereModel)]]

### deorbit.simulator.get_available_sim_methods()

Python magic to find the names of implemented simulation methods.

* **Returns:**
  a dictionary of {name: method class}
* **Return type:**
  dict[str, subclass([Simulator](#deorbit.simulator.Simulator))]

### deorbit.simulator.run(sim_method: str, atmos_model: str, initial_state: \_SupportsArray[dtype[Any]] | \_NestedSequence[\_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | \_NestedSequence[bool | int | float | complex | str | bytes], initial_time: float = 0.0, time_step: float = 0.1, noise_types: dict[str, dict | [NoiseKwargs](deorbit.data_models.md#deorbit.data_models.noise.NoiseKwargs)] | None = None, sim_method_kwargs: dict | type[[MethodKwargs](deorbit.data_models.md#deorbit.data_models.methods.MethodKwargs)] | None = None, atmos_kwargs: dict | type[[AtmosKwargs](deorbit.data_models.md#deorbit.data_models.atmos.AtmosKwargs)] | None = None, steps: int | None = None)
