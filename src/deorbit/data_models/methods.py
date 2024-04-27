from typing import ClassVar, Optional

from pydantic import BaseModel


class MethodKwargs(BaseModel):
    method_name: ClassVar[str]
    # By default, our simulation is 2 dimensional
    dimension: int = 2
    time_step: float
    noise_strength: float = 0.0
    noise_type: str | None = None


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
