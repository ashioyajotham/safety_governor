import numpy as np
import pytest
from safety_governor.vectors import bootstrap_cosine, difference_in_means, pca_direction, probe_direction


@pytest.fixture
def activations():
    safe = np.array([[0., 0.], [0.2, 0.]])
    unsafe = np.array([[2., 0.], [2.2, 0.]])
    return safe, unsafe


@pytest.mark.parametrize("extractor", [difference_in_means, pca_direction, probe_direction])
def test_extractors_point_toward_unsafe(activations, extractor):
    safe, unsafe = activations
    assert extractor(safe, unsafe)[0] > 0.9


def test_bootstrap_resamples_pairs_together():
    safe = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    unsafe = safe + np.array([0.0, 1.0])
    scores = bootstrap_cosine(difference_in_means, safe, unsafe, samples=20, seed=7)
    assert np.allclose(scores, 1.0)


def test_bootstrap_rejects_unpaired_rows():
    with pytest.raises(ValueError, match="equal safe and unsafe"):
        bootstrap_cosine(difference_in_means, np.zeros((2, 2)), np.zeros((3, 2)))
