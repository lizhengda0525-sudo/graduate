import numpy as np
import torch

from tc_cmlp.data.synthetic import generate_synthetic_data
from tc_cmlp.model import TemporallyCoupledCMLP
from tc_cmlp.official import REPOSITORY_ROOT, cMLP, module


def test_official_cmlp_source() -> None:
    assert module.__file__ is not None
    assert module.__file__.startswith(str(REPOSITORY_ROOT))


def test_temporal_model_uses_original_networks() -> None:
    dataset = generate_synthetic_data(5, 2, 20, 2, 0.15, 1, 0)
    torch.manual_seed(0)
    model = TemporallyCoupledCMLP(2, 5, 2, [4])
    assert all(isinstance(window_model, cMLP) for window_model in model.window_models)

    first_network = model.window_models[0].networks[0]
    second_network = model.window_models[1].networks[0]
    assert first_network.layers[0] is not second_network.layers[0]
    assert second_network.layers[1] is first_network.layers[1]
    assert torch.equal(first_network.layers[0].weight, second_network.layers[0].weight)

    predictors = torch.from_numpy(dataset.windows[:, None, :-1, :])
    predictions = model(predictors)
    assert predictions.shape == (2, 1, 18, 5)
    assert model.GC(threshold=False).shape == (2, 5, 5)
    assert np.isfinite(model.GC(threshold=False).detach().numpy()).all()
