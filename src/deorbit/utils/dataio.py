import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from deorbit.data_models.sim import SimConfig, SimData

DATA_STEM = "data"
CONFIG_STEM = "config"
CONFIG_FMT = "pkl"


class DataIO(ABC):
    name: None | str = None

    def __init__(self):
        if self.name is None:
            raise ValueError("DataIO subclasses must define a name attribute")

    @abstractmethod
    def save(self, data: BaseModel, path) -> None: ...

    @abstractmethod
    def load(self, path, data_model: type[BaseModel]) -> BaseModel: ...


class JSONIO(DataIO):
    name = "json"

    def save(self, data: BaseModel, path) -> None:
        with open(path, "w") as f:
            json.dump(data.model_dump_json(), f)

    def load(self, path, data_model: type[BaseModel]) -> BaseModel:
        with open(path) as f:
            model = data_model.model_validate_json(json.load(f))
            return model


class PickleIO(DataIO):
    name = "pkl"

    def save(self, data: BaseModel, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path) -> BaseModel:
        with open(path, "rb") as f:
            model = pickle.load(f)
            return model


formats = {io.name: io for io in DataIO.__subclasses__()}


def _get_filename(
    stem: str,
    dir_path: None | str = None,
    file_type: None | str = None,
) -> str:
    """Generates filenames in the format [dir_path]/[stem].[file_type]

    Args:
        stem (str): Stem for the filename.
        dir_path (optional, str): Directory path prefix
        file_type (optional, str): file type suffix

    Returns:
        Path: filename, with optional suffix and directory path
    """
    if dir_path is None:
        dir_path = ""

    if file_type is None:
        file_type = ""
    else:
        file_type = "." + file_type

    filename = stem + file_type

    return Path(dir_path, filename)


def _bump_file_name(path: Path) -> Path:
    """Given a path to a file, check if the file already exists.
    If so, append `_[number]` to the filename where `number` is the
    lowest number such that the file name is unique.

    Args:
        path (Path): File path to be checked

    Returns:
        Path: Potentially modified, unique file path
    """
    if path.exists():
        stem = path.stem
        i = 1
        while path.exists():
            path = path.with_stem(stem + "_" + str(i))
            i += 1
    return path


# TODO: Make saving easier
def save_sim_data_and_config(
    data: SimData,
    config: SimConfig,
    save_path: Path | str,
    overwrite: bool = True,
    format: str = "pkl",
) -> Path:
    """Saves the simulation data `data` and config `config` in the provided format.
    The config and data are saved in separate files in a new directory.

    Args:
        data (SimData): Data to be saved
        save_path (str): Directory where the `data` and `config` files will be saved.
        format (str): Data file format to use. Default: `pkl`

    Raises:
        NotADirectoryError: Raised if `save_path` exists and is not a valid directory.

    Returns:
        Path: Path to the saved data file.
    """
    # Config format should be pickle as it maintains information about the model/method used

    save_path = Path(save_path)
    
    os.makedirs(save_path, exist_ok=True)

    data_path = _get_filename(DATA_STEM, dir_path=save_path, file_type=format)
    if not overwrite:
        data_path = _bump_file_name(data_path)

    config_path = _get_filename(
        CONFIG_STEM, dir_path=save_path, file_type=CONFIG_FMT
    )
    if not overwrite:
        config_path = _bump_file_name(config_path)

    io = formats[format]()
    io.save(data, data_path)
    io = formats[CONFIG_FMT]()
    io.save(config, config_path)

    return save_path


def load_sim_data(save_path: Path | str, silent: bool = True) -> SimData | None:
    """Load the simulation data from the provided directory path.
    The simulation data file is expected to be in the format `data.[format]`.

    Args:
        save_path (str): Directory path containing the simulation data. e.g. `./data/sim_data_1/`
        silent (bool): If True, suppresses the FileNotFoundError exception if save_path 
            is not found. Default: True

    Raises:
        NotADirectoryError: `save_path` is not a directory
        FileNotFoundError: No data file found in `save_path`

    Returns:
        SimData: Loaded simulation data
    """
    save_path: Path = Path(save_path)
    
    if not save_path.exists():
        if silent:
            return None
        raise FileNotFoundError(f"The directory {save_path} does not exist")

    if not save_path.is_dir():
        raise NotADirectoryError(f"{save_path} is not a directory")

    for p in save_path.iterdir():
        if p.is_file() and p.stem.startswith(DATA_STEM):
            path = p
            break
    else:
        raise FileNotFoundError(f"No data file found in {save_path}")

    # format is defined by extension
    format = path.suffix[1:]
    io = formats[format]()
    # json loader needs the target datamodel
    if format == "json":
        sim_data = io.load(path, data_model=SimData)
    else:
        sim_data = io.load(path)
    return sim_data


def load_sim_config(save_path: str, silent: bool = True) -> SimConfig | None:
    """Load the simulation config from the provided directory path.
    The config file is expected to be in the format `config.pkl`.

    Args:
        save_path (str): Directory path containing the simulation config. e.g. `./data/sim_data_1/`
        silent (bool): If True, suppresses the FileNotFoundError exception if save_path
            is not found. Default: True

    Raises:
        NotADirectoryError: `save_path` is not a directory
        FileNotFoundError: No config file found in `save_path`

    Returns:
        SimConfig: Loaded simulation config
    """
    save_path: Path = Path(save_path)
    
    if not save_path.exists():
        if silent:
            return None
        raise FileNotFoundError(f"The directory {save_path} does not exist")

    if not save_path.is_dir():
        raise NotADirectoryError(f"{save_path} is not a directory")

    path: Path = Path(save_path) / (CONFIG_STEM + "." + CONFIG_FMT)

    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist")

    # format is defined by extension
    io = formats[path.suffix[1:]]()

    sim_config = io.load(path)
    return sim_config
