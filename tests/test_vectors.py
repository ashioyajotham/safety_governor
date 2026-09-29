import numpy as np
import pytest
from safety_governor.vectors import (
    archetype_balanced_weights,
    balanced_difference_in_means,
    balanced_ridge_direction,
    binary_auc,
    bootstrap_cosine,
    deterministic_group_folds,
    difference_in_means,
    paired_delta_pca,
    probe_direction,
    select_balanced_ridge_l2,
)


@pytest.fixture
def activations():
    safe = np.array([[0., 0.], [0.2, 0.]])
    unsafe = np.array([[2., 0.], [2.2, 0.]])
    return safe, unsafe


@pytest.mark.parametrize("extractor", [difference_in_means, paired_delta_pca, probe_direction])
def test_extractors_point_toward_unsafe(activations, extractor):
    safe, unsafe = activations
    assert extractor(safe, unsafe)[0] > 0.9


def test_paired_delta_pca_ignores_shared_prompt_variance():
    safe = np.array([[100., 0.], [-100., 0.], [50., 0.]])
    unsafe = safe + np.array([0., 2.])
    assert paired_delta_pca(safe, unsafe)[1] > 0.9


def test_bootstrap_resamples_pairs_together():
    safe = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    unsafe = safe + np.array([0.0, 1.0])
    scores = bootstrap_cosine(difference_in_means, safe, unsafe, samples=20, seed=7, group_ids=["a", "a", "b"])
    assert np.allclose(scores, 1.0)


def test_bootstrap_rejects_unpaired_rows():
    with pytest.raises(ValueError, match="equal safe and unsafe"):
        bootstrap_cosine(difference_in_means, np.zeros((2, 2)), np.zeros((3, 2)))


def test_archetype_weights_assign_equal_total_mass():
    weights = archetype_balanced_weights(["many", "many", "many", "few"])
    assert np.isclose(weights[:3].sum(), 0.5)
    assert np.isclose(weights[3], 0.5)


def test_balanced_estimators_do_not_let_repeated_archetype_dominate():
    safe = np.zeros((4, 2))
    unsafe = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 3.]])
    archetypes = ["many", "many", "many", "few"]
    dim = balanced_difference_in_means(safe, unsafe, archetypes)
    ridge = balanced_ridge_direction(safe, unsafe, archetypes, l2=1.0)
    assert dim[1] > dim[0]
    assert ridge.vector[1] > ridge.vector[0]
    assert np.isclose(np.linalg.norm(ridge.vector), 1.0)


def test_group_folds_are_deterministic_and_stratified():
    groups = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(6)]
    archetypes = ["a"] * 6 + ["b"] * 6
    first = deterministic_group_folds(groups, archetypes, folds=3, seed=4)
    second = deterministic_group_folds(groups, archetypes, folds=3, seed=4)
    assert np.array_equal(first, second)
    for archetype in ("a", "b"):
        local = first[np.asarray(archetypes) == archetype]
        assert sorted(np.bincount(local)) == [2, 2, 2]


def test_balanced_ridge_selection_returns_oof_diagnostics():
    safe = np.array([[0., 0.], [0.1, 0.], [0., 0.1], [0.1, 0.1], [0., -0.1], [-0.1, 0.]])
    unsafe = safe + np.array([2., 0.])
    groups = [f"g{i}" for i in range(6)]
    archetypes = ["a"] * 3 + ["b"] * 3
    fitted, diagnostics = select_balanced_ridge_l2(
        safe, unsafe, groups, archetypes, candidates=(0.1, 1.0), folds=3
    )
    assert fitted.vector[0] > 0.99
    assert diagnostics["selected_l2"] in {0.1, 1.0}
    assert all(row["macro_auc"] == 1.0 for row in diagnostics["candidates"])
    assert binary_auc(np.array([0, 1]), np.array([0.0, 1.0])) == 1.0
