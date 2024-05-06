from typing import ClassVar

from pydantic import BaseModel


class NoiseKwargs(BaseModel):
    noise_name: ClassVar[str]


class GaussianNoiseKwargs(NoiseKwargs):
    noise_name = "gaussian"
    noise_strength: float = 0.001
    

class ImpulseNoiseKwargs(NoiseKwargs):
    noise_name = "impulse"
    impulse_strength: float = 0.01
    impulse_probability: float = 1e-5


def get_model_for_noise(noise_name: str) -> type[NoiseKwargs]:
    """Returns the correct kwargs model for the given simulation method

    Args:
        sim_method_name (str): The name of the simulation method

    Returns:
        type[MethodKwargs]: The kwargs model for the given simulation method

    Raises:
        ValueError: If the simulation method has no supporting kwargs model
    """

    for model in NoiseKwargs.__subclasses__():
        if model.noise_name == noise_name:
            return model
    else:
        raise ValueError(
            f"Sim method {noise_name} has no supporting kwargs model!"
        )
