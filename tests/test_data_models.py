import pytest

from deorbit.data_models.atmos import get_model_for_atmos
from deorbit.data_models.methods import get_model_for_sim

def test_get_model_invalid_string():
    with pytest.raises(ValueError):
        get_model_for_sim("not_a_method")
    with pytest.raises(ValueError):
        get_model_for_atmos("not_an_atmos")