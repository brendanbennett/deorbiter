from typing import ClassVar, Optional

from pydantic import BaseModel, field_validator

from deorbit.data_models.noise import NoiseKwargs, get_model_for_noise


class MethodKwargs(BaseModel):
    method_name: ClassVar[str]
    # By default, our simulation is 2 dimensional
    dimension: int = 2
    time_step: float
    noise_types: dict[str, dict | NoiseKwargs] = {}

    # We want to be able to take dictionary as input for noise types' parameters,
    # but we want it stored as NoiseKwargs objects.
    @field_validator("noise_types")
    @classmethod
    def validate_noise_types(cls, v: dict[str, dict | NoiseKwargs]):
        for noise_name, noise_kwargs in v.items():
            if isinstance(noise_kwargs, dict):
                v[noise_name] = get_model_for_noise(noise_name)(**noise_kwargs)
        return v


# Children of MethodKwargs should have usable defaults for every attribute
class RK4Kwargs(MethodKwargs):
    method_name = "RK4"


class AdamsBashforthKwargs(MethodKwargs):
    method_name = "adams_bashforth"


class EulerKwargs(MethodKwargs):
    method_name = "euler"


def get_model_for_sim(sim_method_name: str) -> type[MethodKwargs]:
    """Returns the correct kwargs model for the given simulation method

    Args:
        sim_method_name (str): The name of the simulation method

    Returns:
        type[MethodKwargs]: The kwargs model for the given simulation method

    Raises:
        ValueError: If the simulation method has no supporting kwargs model
    """

    for model in MethodKwargs.__subclasses__():
        if model.method_name == sim_method_name:
            return model
    else:
        raise ValueError(
            f"Sim method {sim_method_name} has no supporting kwargs model!"
        )
