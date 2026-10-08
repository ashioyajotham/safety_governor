"""Causal fixture audit, not evidence about Llama behavioral efficacy.

The fixture propagates earlier states with a causal prefix mean. This makes
history coverage observable without downloads or expensive model inference.
"""
import numpy as np
import pytest

from safety_governor.models import (
    SteeringSite, ProjectionGate, generate_with_governor,
    response_negative_log_likelihood_governed,
    response_negative_log_likelihood_frontier,
)


PREFIX = [1, 2]
RESPONSE = [3, 4, 5]
BUDGET = 0.2


def causal_forward(tokens, positions=(), budget=BUDGET):
    """Two blocks: injection at residual-pre, then causal state mixing."""
    hidden = np.array([[float(t), 1.0] for t in tokens])
    for _ in range(2):
        if positions:
            indices = list(positions)
            hidden[indices, 0] -= budget * 0.5 * np.linalg.norm(hidden[indices], axis=-1)
        hidden = hidden + np.cumsum(hidden, axis=0) / np.arange(1, len(tokens) + 1)[:, None]
    return hidden @ np.array([[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
                              [-0.1, 0.2, 0.3, -0.3, 0.1, 0.0]])


def scores(policy, budget=BUDGET):
    if policy == 'parallel':
        positions = range(len(PREFIX) - 1, len(PREFIX) + len(RESPONSE) - 1)
        return causal_forward(PREFIX + RESPONSE, positions, budget)[1:4]
    result = []
    for i in range(len(RESPONSE)):
        history = PREFIX + RESPONSE[:i]
        positions = (len(history) - 1,) if policy == 'frontier' else ()
        result.append(causal_forward(history, positions, budget)[-1])
    return np.array(result)


def test_zero_budget_parallel_and_incremental_logits_match():
    np.testing.assert_allclose(scores('parallel', 0), scores('frontier', 0), atol=1e-12)


def test_first_predictor_matches_but_later_frontier_predictors_differ():
    parallel, frontier = scores('parallel'), scores('frontier')
    np.testing.assert_allclose(parallel[0], frontier[0], atol=1e-12)
    assert np.max(np.abs(parallel[1:] - frontier[1:])) > 0.01


def test_future_response_tokens_do_not_leak_into_predictor():
    positions = range(1, 4)
    original = causal_forward(PREFIX + RESPONSE, positions)
    changed = causal_forward(PREFIX + [3, 4, 99], positions)
    np.testing.assert_allclose(original[:4], changed[:4], atol=1e-12)


def fixture_model():
    torch = pytest.importorskip('torch')

    class Tokenizer:
        chat_template = None

        def encode(self, text, add_special_tokens=False):
            return PREFIX if text.startswith('User:') else RESPONSE

    class Model:
        tokenizer = Tokenizer()

        def parameters(self):
            yield torch.zeros(1, dtype=torch.float64)

        def run_with_hooks(self, tokens, attention_mask, return_type, fwd_hooks):
            hidden = torch.stack((tokens.double(), torch.ones_like(tokens).double()), dim=-1)
            hooks = dict(fwd_hooks)
            for layer in range(2):
                name = f'blocks.{layer}.hook_resid_pre'
                if name in hooks:
                    hidden = hooks[name](hidden)
                denominator = torch.arange(1, tokens.shape[1] + 1, dtype=torch.float64)[None, :, None]
                hidden = hidden + hidden.cumsum(dim=1) / denominator
            readout = torch.tensor([[0.3, -0.2, 0.1, 0.4, -0.1, 0.2],
                                    [-0.1, 0.2, 0.3, -0.3, 0.1, 0.0]], dtype=torch.float64)
            return hidden @ readout

    return Model()


def test_actual_screen_scorer_matches_parallel_not_frontier_fixture():

    sites = [SteeringSite(i, np.array([1., 0.]), 0.5) for i in range(2)]
    loss, count = response_negative_log_likelihood_governed(
        fixture_model(), 'prompt', 'completion', sites, total_relative_l2=BUDGET,
    )

    def nll(logits):
        maximum = logits.max(axis=1)
        normalizer = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
        return float((normalizer - logits[np.arange(3), RESPONSE]).sum())

    assert count == 3
    assert loss == pytest.approx(nll(scores('parallel')), abs=1e-10)
    assert abs(loss - nll(scores('frontier'))) > 1e-3


@pytest.mark.parametrize('budget', [0., BUDGET])
def test_frontier_scorer_matches_incremental_reference(budget):
    sites = [SteeringSite(i, np.array([1., 0.]), .5) for i in range(2)]
    loss, count = response_negative_log_likelihood_frontier(
        fixture_model(), 'prompt', 'completion', sites, total_relative_l2=budget)
    logits = scores('frontier', budget)
    maximum = logits.max(axis=1)
    expected = (maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
                - logits[np.arange(3), RESPONSE]).sum()
    assert count == 3
    assert loss == pytest.approx(expected, abs=1e-6)


def test_frontier_empty_sites_baseline_equals_zero_budget_control():
    model = fixture_model()
    sites = [SteeringSite(0, np.array([1., 0.]), 1.)]
    assert response_negative_log_likelihood_frontier(model, 'p', 'c') == (
        response_negative_log_likelihood_frontier(model, 'p', 'c', sites))


@pytest.mark.parametrize('budget', [-1., float('nan'), float('inf'), .2])
def test_frontier_rejects_invalid_or_siteless_positive_budget(budget):
    with pytest.raises(ValueError):
        response_negative_log_likelihood_frontier(None, 'p', 'c', total_relative_l2=budget)


@pytest.mark.parametrize('gate', [None, ProjectionGate(1., 2.)])
def test_frontier_scorer_and_generation_share_histories_and_hook_effects(gate):
    torch = pytest.importorskip('torch')

    class Tokenizer:
        chat_template = None
        eos_token_id = None
        def encode(self, text, add_special_tokens=False):
            return PREFIX if text.startswith('User:') else RESPONSE
        def decode(self, tokens, skip_special_tokens=True):
            assert tokens == RESPONSE
            return 'completion'

    class Model:
        tokenizer = Tokenizer()
        def __init__(self):
            self.calls = []
        def parameters(self):
            yield torch.zeros(1)
        def run_with_hooks(self, tokens, attention_mask, return_type, fwd_hooks):
            assert not torch.is_grad_enabled()
            assert attention_mask.shape == tokens.shape
            hidden = torch.stack((tokens.float(), torch.ones_like(tokens).float()), -1)
            for _, hook in fwd_hooks:
                hidden = hook(hidden)
            self.calls.append((tokens.clone(), hidden.clone()))
            logits = torch.zeros(1, tokens.shape[1], 6)
            logits[0, -1, RESPONSE[tokens.shape[1] - len(PREFIX)]] = 4.
            return logits

    sites = [SteeringSite(0, np.array([1., 0.]), 1., gate)]
    generated = Model()
    with torch.inference_mode():
        text = generate_with_governor(generated, 'p', sites,
            total_relative_l2=BUDGET, max_new_tokens=3, token_policy='generation_frontier')
    scored = Model()
    loss, count = response_negative_log_likelihood_frontier(
        scored, 'p', text, sites, total_relative_l2=BUDGET)
    assert count == 3 and np.isfinite(loss)
    for index, ((gt, gh), (st, sh)) in enumerate(zip(generated.calls, scored.calls)):
        assert st.tolist() == [PREFIX + RESPONSE[:index]]
        torch.testing.assert_close(gt, st)
        torch.testing.assert_close(gh, sh)


def test_frontier_rejects_empty_response():
    model = fixture_model()
    original = model.tokenizer.encode
    model.tokenizer.encode = lambda text, **kw: original(text, **kw) if text.startswith('User:') else []
    with pytest.raises(ValueError, match='empty response'):
        response_negative_log_likelihood_frontier(model, 'p', '')


@pytest.mark.parametrize('exact_first', [True, False])
def test_screen_resume_cannot_mix_parallel_and_frontier_policies(tmp_path, exact_first):
    pytest.importorskip('torch')
    from scripts.screen_validation_v3 import _lock_screen_spec
    historical = {'schema_version': 1, 'phase': 'historical_parallel'}
    exact = {**historical, 'schema_version': 2, 'scoring_policy': 'frontier_exact'}
    first, second = (exact, historical) if exact_first else (historical, exact)
    _lock_screen_spec(tmp_path, first, False)
    original = (tmp_path / 'run_spec.json').read_bytes()
    _lock_screen_spec(tmp_path, first, True)
    with pytest.raises(ValueError, match='specification differs'):
        _lock_screen_spec(tmp_path, second, True)
    assert (tmp_path / 'run_spec.json').read_bytes() == original


def test_screen_rejects_orphaned_shards(tmp_path):
    pytest.importorskip('torch')
    from scripts.screen_validation_v3 import _lock_screen_spec
    (tmp_path / 'orphan.json').write_text('{}')
    with pytest.raises(FileExistsError, match='no specification'):
        _lock_screen_spec(tmp_path, {}, True)
