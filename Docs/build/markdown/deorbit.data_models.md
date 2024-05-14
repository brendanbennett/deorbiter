# deorbit.data_models package

## Submodules

## deorbit.data_models.atmos module

### *class* deorbit.data_models.atmos.AtmosKwargs(\*, earth_radius: float = 6371000, earth_angular_velocity: float = 7.2921159e-05)

Bases: `BaseModel`

Point of truth for atmosphere model parameters

#### atmos_name *: ClassVar[str]*

#### earth_angular_velocity *: float*

#### earth_radius *: float*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'earth_angular_velocity': FieldInfo(annotation=float, required=False, default=7.2921159e-05), 'earth_radius': FieldInfo(annotation=float, required=False, default=6371000)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

### *class* deorbit.data_models.atmos.CoesaFastKwargs(\*, earth_radius: float = 6371000, earth_angular_velocity: float = 7.2921159e-05, precision: int = 2)

Bases: [`AtmosKwargs`](#deorbit.data_models.atmos.AtmosKwargs)

#### atmos_name *: ClassVar[str]* *= 'coesa_atmos_fast'*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'earth_angular_velocity': FieldInfo(annotation=float, required=False, default=7.2921159e-05), 'earth_radius': FieldInfo(annotation=float, required=False, default=6371000), 'precision': FieldInfo(annotation=int, required=False, default=2)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

#### precision *: int*

### *class* deorbit.data_models.atmos.CoesaKwargs(\*, earth_radius: float = 6371000, earth_angular_velocity: float = 7.2921159e-05)

Bases: [`AtmosKwargs`](#deorbit.data_models.atmos.AtmosKwargs)

#### atmos_name *: ClassVar[str]* *= 'coesa_atmos'*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'earth_angular_velocity': FieldInfo(annotation=float, required=False, default=7.2921159e-05), 'earth_radius': FieldInfo(annotation=float, required=False, default=6371000)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

### *class* deorbit.data_models.atmos.IcaoKwargs(\*, earth_radius: float = 6371000, earth_angular_velocity: float = 7.2921159e-05)

Bases: [`AtmosKwargs`](#deorbit.data_models.atmos.AtmosKwargs)

#### atmos_name *: ClassVar[str]* *= 'icao_standard_atmos'*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'earth_angular_velocity': FieldInfo(annotation=float, required=False, default=7.2921159e-05), 'earth_radius': FieldInfo(annotation=float, required=False, default=6371000)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

### *class* deorbit.data_models.atmos.SimpleAtmosKwargs(\*, earth_radius: float = 6371000, earth_angular_velocity: float = 7.2921159e-05, surf_density: float = 1.225)

Bases: [`AtmosKwargs`](#deorbit.data_models.atmos.AtmosKwargs)

#### atmos_name *: ClassVar[str]* *= 'simple_atmos'*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'earth_angular_velocity': FieldInfo(annotation=float, required=False, default=7.2921159e-05), 'earth_radius': FieldInfo(annotation=float, required=False, default=6371000), 'surf_density': FieldInfo(annotation=float, required=False, default=1.225)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

#### surf_density *: float*

### *class* deorbit.data_models.atmos.ZeroAtmosKwargs(\*, earth_radius: float = 6371000, earth_angular_velocity: float = 7.2921159e-05)

Bases: [`AtmosKwargs`](#deorbit.data_models.atmos.AtmosKwargs)

#### atmos_name *: ClassVar[str]* *= 'zero_atmos'*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'earth_angular_velocity': FieldInfo(annotation=float, required=False, default=7.2921159e-05), 'earth_radius': FieldInfo(annotation=float, required=False, default=6371000)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

### deorbit.data_models.atmos.get_model_for_atmos(atmos_model_name: str)

Returns the correct kwargs model for the given atmosphere model.

* **Parameters:**
  **atmos_model_name** (*str*) – The name of the atmosphere model.
* **Returns:**
  The kwargs model corresponding to the given atmosphere model.
* **Return type:**
  type[[AtmosKwargs](#deorbit.data_models.atmos.AtmosKwargs)]
* **Raises:**
  **ValueError** – If the atmosphere model has no supporting kwargs model.

## deorbit.data_models.methods module

### *class* deorbit.data_models.methods.AdamsBashforthKwargs(\*, dimension: int = 2, time_step: float, noise_types: dict[str, dict | [NoiseKwargs](#deorbit.data_models.noise.NoiseKwargs)] = {})

Bases: [`MethodKwargs`](#deorbit.data_models.methods.MethodKwargs)

#### method_name *: ClassVar[str]* *= 'adams_bashforth'*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'dimension': FieldInfo(annotation=int, required=False, default=2), 'noise_types': FieldInfo(annotation=dict[str, Union[dict, NoiseKwargs]], required=False, default={}), 'time_step': FieldInfo(annotation=float, required=True)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

### *class* deorbit.data_models.methods.EulerKwargs(\*, dimension: int = 2, time_step: float, noise_types: dict[str, dict | [NoiseKwargs](#deorbit.data_models.noise.NoiseKwargs)] = {})

Bases: [`MethodKwargs`](#deorbit.data_models.methods.MethodKwargs)

#### method_name *: ClassVar[str]* *= 'euler'*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'dimension': FieldInfo(annotation=int, required=False, default=2), 'noise_types': FieldInfo(annotation=dict[str, Union[dict, NoiseKwargs]], required=False, default={}), 'time_step': FieldInfo(annotation=float, required=True)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

### *class* deorbit.data_models.methods.MethodKwargs(\*, dimension: int = 2, time_step: float, noise_types: dict[str, dict | [NoiseKwargs](#deorbit.data_models.noise.NoiseKwargs)] = {})

Bases: `BaseModel`

#### dimension *: int*

#### method_name *: ClassVar[str]*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'dimension': FieldInfo(annotation=int, required=False, default=2), 'noise_types': FieldInfo(annotation=dict[str, Union[dict, NoiseKwargs]], required=False, default={}), 'time_step': FieldInfo(annotation=float, required=True)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

#### noise_types *: dict[str, dict | [NoiseKwargs](#deorbit.data_models.noise.NoiseKwargs)]*

#### time_step *: float*

#### *classmethod* validate_noise_types(v: dict[str, dict | [NoiseKwargs](#deorbit.data_models.noise.NoiseKwargs)])

### *class* deorbit.data_models.methods.RK4Kwargs(\*, dimension: int = 2, time_step: float, noise_types: dict[str, dict | [NoiseKwargs](#deorbit.data_models.noise.NoiseKwargs)] = {})

Bases: [`MethodKwargs`](#deorbit.data_models.methods.MethodKwargs)

#### method_name *: ClassVar[str]* *= 'RK4'*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'dimension': FieldInfo(annotation=int, required=False, default=2), 'noise_types': FieldInfo(annotation=dict[str, Union[dict, NoiseKwargs]], required=False, default={}), 'time_step': FieldInfo(annotation=float, required=True)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

### deorbit.data_models.methods.get_model_for_sim(sim_method_name: str)

Returns the correct kwargs model for the given simulation method

* **Parameters:**
  **sim_method_name** (*str*) – The name of the simulation method
* **Returns:**
  The kwargs model for the given simulation method
* **Return type:**
  type[[MethodKwargs](#deorbit.data_models.methods.MethodKwargs)]
* **Raises:**
  **ValueError** – If the simulation method has no supporting kwargs model

## deorbit.data_models.noise module

### *class* deorbit.data_models.noise.GaussianNoiseKwargs(\*, noise_strength: float = 0.001)

Bases: [`NoiseKwargs`](#deorbit.data_models.noise.NoiseKwargs)

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'noise_strength': FieldInfo(annotation=float, required=False, default=0.001)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

#### noise_name *: ClassVar[str]* *= 'gaussian'*

#### noise_strength *: float*

### *class* deorbit.data_models.noise.ImpulseNoiseKwargs(\*, impulse_strength: float = 0.01, impulse_probability: float = 1e-05)

Bases: [`NoiseKwargs`](#deorbit.data_models.noise.NoiseKwargs)

#### impulse_probability *: float*

#### impulse_strength *: float*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'impulse_probability': FieldInfo(annotation=float, required=False, default=1e-05), 'impulse_strength': FieldInfo(annotation=float, required=False, default=0.01)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

#### noise_name *: ClassVar[str]* *= 'impulse'*

### *class* deorbit.data_models.noise.NoiseKwargs

Bases: `BaseModel`

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

#### noise_name *: ClassVar[str]*

### deorbit.data_models.noise.get_model_for_noise(noise_name: str)

Returns the correct kwargs model for the given simulation method

* **Parameters:**
  **sim_method_name** (*str*) – The name of the simulation method
* **Returns:**
  The kwargs model for the given simulation method
* **Return type:**
  type[[MethodKwargs](#deorbit.data_models.methods.MethodKwargs)]
* **Raises:**
  **ValueError** – If the simulation method has no supporting kwargs model

## deorbit.data_models.obs module

### *class* deorbit.data_models.obs.ObsData(\*, x1: list[float], x2: list[float], x3: list[float] | None = None, times: list[float])

Bases: `BaseModel`

Output of the Observer which is a sparser copy of SimData from the Simulator.
Only includes the states at times where the satellite is in view of a ground radar station

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'times': FieldInfo(annotation=list[float], required=True), 'x1': FieldInfo(annotation=list[float], required=True), 'x2': FieldInfo(annotation=list[float], required=True), 'x3': FieldInfo(annotation=Union[list[float], NoneType], required=False)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

#### times *: list[float]*

#### x1 *: list[float]*

#### x2 *: list[float]*

#### x3 *: list[float] | None*

## deorbit.data_models.sim module

### *class* deorbit.data_models.sim.SimConfig(\*, initial_state: list, initial_time: float = 0.0, simulation_method_kwargs: [MethodKwargs](#deorbit.data_models.methods.MethodKwargs), atmosphere_model_kwargs: [AtmosKwargs](#deorbit.data_models.atmos.AtmosKwargs))

Bases: `BaseModel`

#### atmosphere_model_kwargs *: [AtmosKwargs](#deorbit.data_models.atmos.AtmosKwargs)*

#### initial_state *: list*

#### initial_time *: float*

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'atmosphere_model_kwargs': FieldInfo(annotation=AtmosKwargs, required=True, metadata=[SerializeAsAny()]), 'initial_state': FieldInfo(annotation=list, required=True), 'initial_time': FieldInfo(annotation=float, required=False, default=0.0), 'simulation_method_kwargs': FieldInfo(annotation=MethodKwargs, required=True, metadata=[SerializeAsAny()])}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

#### simulation_method_kwargs *: [MethodKwargs](#deorbit.data_models.methods.MethodKwargs)*

### *class* deorbit.data_models.sim.SimData(\*, x1: list[float], x2: list[float], x3: list[float] | None = None, v1: list[float], v2: list[float], v3: list[float] | None = None, times: list[float])

Bases: `BaseModel`

#### model_config *: ClassVar[ConfigDict]* *= {}*

Configuration for the model, should be a dictionary conforming to [ConfigDict][pydantic.config.ConfigDict].

#### model_fields *: ClassVar[dict[str, FieldInfo]]* *= {'times': FieldInfo(annotation=list[float], required=True), 'v1': FieldInfo(annotation=list[float], required=True), 'v2': FieldInfo(annotation=list[float], required=True), 'v3': FieldInfo(annotation=Union[list[float], NoneType], required=False), 'x1': FieldInfo(annotation=list[float], required=True), 'x2': FieldInfo(annotation=list[float], required=True), 'x3': FieldInfo(annotation=Union[list[float], NoneType], required=False)}*

Metadata about the fields defined on the model,
mapping of field names to [FieldInfo][pydantic.fields.FieldInfo].

This replaces Model._\_fields_\_ from Pydantic V1.

#### state_array()

#### times *: list[float]*

#### v1 *: list[float]*

#### v2 *: list[float]*

#### v3 *: list[float] | None*

#### x1 *: list[float]*

#### x2 *: list[float]*

#### x3 *: list[float] | None*

## Module contents
