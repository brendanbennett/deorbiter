import json
import os
from datetime import datetime
from pathlib import Path

from deorbit.data_models.sim import SimData


def save_sim_data(data: SimData, dir_path_string: str) -> Path:
    """Saves the simulation data `data` as a json file.
    The json filename is auto-generated by `dataio._get_filename()`.

    Args:
        data (SimData): Data to be saved
        dir_path_string (str): Directory where the `data` json with be saved.

    Raises:
        NotADirectoryError: Raised if `dir_path_string` exists and is not a valid directory.

    Returns:
        Path: Path to the saved data file.
    """
    dir_path = Path(dir_path_string)

    if not dir_path.exists():
        os.makedirs(dir_path)
    elif not dir_path.is_dir():
        raise NotADirectoryError(
            f"dir_path_string must be a directory! {dir_path_string} is not a directory."
        )

    filename = _get_filename("sim_data")
    path = dir_path / (filename + ".json")

    path = _check_for_file(path)

    with open(path, "w") as f:
        json.dump(data.model_dump_json(), f)

    return path


def _get_filename(stem: str) -> str:
    """Generates filenames in the format [stem]_yyyymmdd_hhmmss

    Returns:
        str: filename
    """
    # time() give floating point unix time.
    t = datetime.utcnow().timetuple()
    timestamp = f"{t[0]:0>4d}{t[1]:0>2d}{t[2]:0>2d}_{t[3]:0>2d}{t[4]:0>2d}{t[5]:0>2d}"
    return stem + "_" + timestamp


def _check_for_file(path: Path) -> Path:
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


def load_sim_data(path: str) -> SimData:
    with open(path) as f:
        model = SimData.model_validate_json(json.load(f))
        return model
