import numpy as np
from safety_governor.evaluation import ControlTax, conceptual_hub, cosine_similarity, suppression
from safety_governor.steering import add_vector


def test_last_token_steering_does_not_touch_prior_tokens():
    activation = np.zeros((1, 2, 2))
    changed = add_vector(activation, np.array([1., 2.]), 3, "last_prompt_token")
    assert np.all(changed[0, 0] == 0)
    assert np.all(changed[0, 1] == np.array([3., 6.]))


def test_control_tax_threshold_and_hub():
    assert ControlTax(.71, .1, -.02, .1).viable
    assert conceptual_hub({0: .3, 4: .9}) == 4
    assert suppression(1.0, .2) == .8
    assert np.allclose(cosine_similarity(np.eye(2), np.eye(2)), [1., 1.])
