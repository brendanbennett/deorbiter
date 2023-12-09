from src.data_models import SimData
import json


def save_sim_data(data: SimData, path: str) -> None:
    # TODO add safe file io (with path checks)
    with open(path, "w") as f:
        json.dump(data.model_dump_json(), f)


def load_sim_data(path: str) -> SimData:
    with open(path) as f:
        model = SimData.model_validate_json(json.load(f))
        return model
