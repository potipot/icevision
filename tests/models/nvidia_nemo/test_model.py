import pytest


@pytest.mark.parametrize("model", ("pretrained_model", "scratch_model"))
def test_initialize_model(model, request):
    model = request.getfixturevalue(model)
    assert True  # just testing if model init works
