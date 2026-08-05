import numpy as np
import pytest
from safety_governor.vectors import difference_in_means, pca_direction, probe_direction


@pytest.fixture
def activations():
    safe = np.array([[0., 0.], [0.2, 0.]])
    unsafe = np.array([[2., 0.], [2.2, 0.]])
    return safe, unsafe


@pytest.mark.parametrize("extractor", [difference_in_means, pca_direction, probe_direction])
def test_extractors_point_toward_unsafe(activations, extractor):
    safe, unsafe = activations
    assert extractor(safe, unsafe)[0] > 0.9
