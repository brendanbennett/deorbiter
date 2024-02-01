from deorbit.simulator.atmos import SimpleAtmos


def test_simple_atmos():
    state = (200, 0, -3, 20)
    time = 0.1
    model_kwargs = {"earth_radius": 200, "surf_density": 1}
    simple_atmos_model = SimpleAtmos(earth_radius=200, surf_density=1)
    returned_model_kwargs = simple_atmos_model.kwargs.model_dump()
    density = simple_atmos_model.density(state=state, time=time)
    assert density == 1
    assert model_kwargs == returned_model_kwargs