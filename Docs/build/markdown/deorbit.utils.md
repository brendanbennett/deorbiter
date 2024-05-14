# deorbit.utils package

## Submodules

## deorbit.utils.constants module

## deorbit.utils.coords module

### deorbit.utils.coords.cart_from_latlong(latlong, radius: float = 6371000)

Given a sequence of latitude and longitude, calculates the cartesian coordinates of a point on the Earth’s surface.
This does not account for the rotation of the Earth; we assume 0N, 0E is at (1, 0, 0).
If working in 2D, latlong should be a scalar representing longitude.

### deorbit.utils.coords.latlong_from_cart(cart)

Given a sequence of cartesian coordinates, calculates the latitude and longitude of a point on the Earth’s surface.
This point is the intersection of the line from the origin to the point and the Earth’s surface.
This does not account for the rotation of the Earth; we assume 0N, 0E is at (1, 0, 0).
If working in 2D, only returns longitude.

## deorbit.utils.dataio module

### *class* deorbit.utils.dataio.DataIO

Bases: `ABC`

#### *abstract* load(path, data_model: type[BaseModel])

#### name *: None | str* *= None*

#### *abstract* save(data: BaseModel, path)

### *class* deorbit.utils.dataio.JSONIO

Bases: [`DataIO`](#deorbit.utils.dataio.DataIO)

#### load(path, data_model: type[BaseModel])

#### name *: None | str* *= 'json'*

#### save(data: BaseModel, path)

### *class* deorbit.utils.dataio.PickleIO

Bases: [`DataIO`](#deorbit.utils.dataio.DataIO)

#### load(path)

#### name *: None | str* *= 'pkl'*

#### save(data: BaseModel, path)

### deorbit.utils.dataio.load_sim_config(save_path: str, silent: bool = True)

Load the simulation config from the provided directory path.
The config file is expected to be in the format config.pkl.

* **Parameters:**
  * **save_path** (*str*) – Directory path containing the simulation config. e.g. ./data/sim_data_1/
  * **silent** (*bool*) – If True, suppresses the FileNotFoundError exception if save_path
    is not found. Default: True
* **Raises:**
  * **NotADirectoryError** – save_path is not a directory
  * **FileNotFoundError** – No config file found in save_path
* **Returns:**
  Loaded simulation config
* **Return type:**
  [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig)

### deorbit.utils.dataio.load_sim_data(save_path: Path | str, silent: bool = True)

Load the simulation data from the provided directory path.
The simulation data file is expected to be in the format data.[format].

* **Parameters:**
  * **save_path** (*str*) – Directory path containing the simulation data. e.g. ./data/sim_data_1/
  * **silent** (*bool*) – If True, suppresses the FileNotFoundError exception if save_path
    is not found. Default: True
* **Raises:**
  * **NotADirectoryError** – save_path is not a directory
  * **FileNotFoundError** – No data file found in save_path
* **Returns:**
  Loaded simulation data
* **Return type:**
  [SimData](deorbit.data_models.md#deorbit.data_models.sim.SimData)

### deorbit.utils.dataio.save_sim_data_and_config(data: [SimData](deorbit.data_models.md#deorbit.data_models.sim.SimData), config: [SimConfig](deorbit.data_models.md#deorbit.data_models.sim.SimConfig), save_path: Path | str, overwrite: bool = True, format: str = 'pkl')

Saves the simulation data data and config config in the provided format.
The config and data are saved in separate files in a new directory.

* **Parameters:**
  * **data** ([*SimData*](deorbit.data_models.md#deorbit.data_models.sim.SimData)) – Data to be saved
  * **save_path** (*str*) – Directory where the data and config files will be saved.
  * **format** (*str*) – Data file format to use. Default: pkl
* **Raises:**
  **NotADirectoryError** – Raised if save_path exists and is not a valid directory.
* **Returns:**
  Path to the saved data file.
* **Return type:**
  Path

## deorbit.utils.plotting module

This module encapsulates various plotting methods for visualizing trajectories, errors, and other relevant data associated with the simulation and prediction of satellite trajectories.

### deorbit.utils.plotting.Three_Dim_Slice_Trajectory(true_traj, estimated_traj, observation_states, observation_times, sim_times, dt, Three_Dim_crash_coords)

### deorbit.utils.plotting.plot_absolute_error(true_traj, estimated_traj)

### deorbit.utils.plotting.plot_error(true_traj, estimated_traj, title='Error in Trajectories')

### deorbit.utils.plotting.plot_position_error(true_traj, estimated_traj, observation_states)

### deorbit.utils.plotting.plot_trajectories(true_traj, estimated_traj=None, observations=None, title='Trajectories')

### deorbit.utils.plotting.plot_velocity_error(true_traj, estimated_traj, title='Error in Velocity')

## Module contents
