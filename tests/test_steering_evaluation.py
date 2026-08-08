import numpy as np
import pytest
from safety_governor.evaluation import ControlTax, conceptual_hub, cosine_similarity, suppression
from safety_governor.steering import add_vector


def test_position_aware_steering_does_not_touch_padding_or_prior_tokens():
    activation = np.zeros((1, 3, 2))
    changed = add_vector(activation, np.array([1., 2.]), 3, "final_response_token", positions=[1])
    assert np.all(changed[0, 0] == 0)
    assert np.all(changed[0, 1] == np.array([3., 6.]))
    assert np.all(changed[0, 2] == 0)


def test_position_aware_mode_refuses_implicit_last_column():
    with pytest.raises(ValueError, match="explicit non-padding positions"):
        add_vector(np.zeros((1, 2, 2)), np.ones(2), 1, "final_response_token")


def test_control_tax_threshold_and_hub():
    assert ControlTax(.71, .1, -.02, .1).viable
    assert conceptual_hub({0: .3, 4: .9}) == 4
    assert suppression(1.0, .2) == .8
    assert np.allclose(cosine_similarity(np.eye(2), np.eye(2)), [1., 1.])

def test_hook_adapter_accepts_transformerlens_keyword():
    from safety_governor.domain import InterventionSpec
    from safety_governor.steering import make_hook

    activation = np.zeros((1, 2, 2))
    hook = make_hook(
        np.ones(2),
        InterventionSpec(layer=0, coefficient=1.0, token_mode="final_response_token"),
        positions=[1],
    )
    changed = hook(activation, hook=object())
    assert np.array_equal(changed[0, 1], np.ones(2))