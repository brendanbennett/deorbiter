import pytest

from deorbit.data_models.atmos import SimpleAtmosKwargs, get_model_for_atmos
from deorbit.simulator.atmos import (
    AtmosphereModel,
    SimpleAtmos,
    get_available_atmos_models,
)
from deorbit.utils.constants import AIR_DENSITY_SEA_LEVEL, EARTH_RADIUS


def test_simple_atmos():
    state = (200, 0, -3, 20)
    time = 0.1
    model_kwargs = {"earth_radius": 200, "surf_density": 1}
    model_kwargs = SimpleAtmosKwargs(earth_radius=200, surf_density=1)
    simple_atmos_model = SimpleAtmos(model_kwargs)
    returned_model_kwargs = simple_atmos_model.kwargs
    density = simple_atmos_model.density(state=state, time=time)
    assert density == 1
    assert model_kwargs == returned_model_kwargs


@pytest.mark.parametrize("model", list(get_available_atmos_models().keys()))
def test_atmos_eval(model):
    kwargs_cls = get_model_for_atmos(model)
    kwargs = kwargs_cls()
    atmos_model = get_available_atmos_models()[model](kwargs)
    state = (EARTH_RADIUS + 8000, 10000, 0, 0)
    density = atmos_model.density(state, time=0.0)
    assert density >= 0
    if model == "zero_atmos":
        assert density == 0


def test_defining_atmos_class_no_name():
    """AtmosphereModel subclasses need a name defined"""
    with pytest.raises(SyntaxError):

        class MyAtmos(AtmosphereModel):
            def density(self):
                pass


def test_defining_atmos_class_no_run_method():
    """AtmosphereModel subclasses need a `_run_method` method"""

    class MyAtmos(AtmosphereModel, model_name="myatmos"):
        pass

    with pytest.raises(TypeError):
        MyAtmos()


def test_adding_sim_subclass():
    """Creating a subclass should update the name dictionary in the parent class"""
    model_name = "myatmosphere"

    class MyAtmos(AtmosphereModel, model_name=model_name):
        pass

    assert model_name in AtmosphereModel._models
    assert model_name in get_available_atmos_models()
