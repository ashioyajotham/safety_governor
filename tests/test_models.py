from types import SimpleNamespace

import numpy as np
import pytest

from safety_governor.models import (
    _governor_span_intervention,
    ProjectionGate,
    SteeringSite,
    generate_unsteered,
    generate_with_governor,
    generate_with_steering,
    residual_at_last_token,
    residual_at_response,
    residuals_at_response,
)


def test_generated_span_scales_each_original_position_and_preserves_prompt():
    torch = pytest.importorskip("torch")
    original = torch.tensor([[[7., 8.], [3., 4.], [0., 10.], [0., 0.]]])
    trace = []
    hook = _governor_span_intervention(
        SteeringSite(12, np.array([1., 0.]), 1.0), 1, 4, 0.2, trace,
    )
    result = hook(original)
    assert torch.allclose(result, torch.tensor([[[7., 8.], [2., 4.], [-2., 10.], [0., 0.]]]))
    assert torch.equal(original[0, 1], torch.tensor([3., 4.]))
    assert trace[0]["modified_positions"] == 3
    assert trace[0]["injected_l2_sum"] == pytest.approx(3.0)
    assert np.isfinite(trace[0]["relative_delta"])


def test_policy_first_step_equivalence_and_generated_history_coverage():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        chat_template = None
        eos_token_id = 9
        def encode(self, text, add_special_tokens):
            return [1, 2]
        def decode(self, tokens, skip_special_tokens):
            return "x" * len(tokens)

    class Model:
        tokenizer = Tokenizer()
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.activations = []
        def parameters(self):
            yield self.weight
        def run_with_hooks(self, tokens, attention_mask, return_type, fwd_hooks):
            active = torch.tensor([[[3., 4.]]]).repeat(1, tokens.shape[1], 1)
            self.activations.append(fwd_hooks[0][1](active))
            logits = torch.zeros((1, tokens.shape[1], 10))
            logits[0, -1, 9 if len(self.activations) == 3 else 3] = 1
            return logits

    models = [Model(), Model()]
    for model, policy in zip(models, ("generation_frontier", "generated_span")):
        meta = {}
        assert generate_with_governor(
            model, "prompt", [SteeringSite(12, np.array([1., 0.]), 1.)],
            total_relative_l2=0.2, max_new_tokens=4,
            token_policy=policy, generation_metadata=meta,
        ) == "xx"
        assert meta["stop_reason"] == "stop_token"
    assert torch.equal(models[0].activations[0], models[1].activations[0])
    assert torch.equal(models[1].activations[2][0, :2], torch.tensor([[3., 4.], [3., 4.]]))
    assert torch.equal(models[0].activations[2][0, 2], torch.tensor([3., 4.]))
    assert torch.equal(models[1].activations[2][0, 2:], torch.tensor([[2., 4.], [2., 4.]]))


def test_governor_rejects_unknown_token_policy_before_inference():
    with pytest.raises(ValueError, match="token policy"):
        generate_with_governor(None, "prompt", [], total_relative_l2=0.2, token_policy="unknown")


def test_residual_capture_uses_last_non_padding_token():
    torch = pytest.importorskip("torch")

    class FakeModel:
        tokenizer = SimpleNamespace(pad_token_id=0)
        def to_tokens(self, prompts):
            return torch.tensor([[1, 2, 3], [1, 4, 0]])
        def run_with_cache(self, tokens, return_type):
            values = torch.tensor([[[10.0], [11.0], [12.0]], [[20.0], [21.0], [99.0]]])
            return None, {"blocks.0.hook_resid_pre": values}

    captured = residual_at_last_token(FakeModel(), ["long", "short"], layer=0)
    assert np.array_equal(captured, np.array([[12.0], [21.0]], dtype=np.float32))


def test_response_mean_uses_explicit_response_mask():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        chat_template = None
        pad_token_id = 0
        eos_token_id = 0
        padding_side = "right"
        def encode(self, text, add_special_tokens):
            if text.startswith("User:"):
                return [1, 2]
            return [3] if text == "short" else [3, 4]

    class FakeModel:
        tokenizer = Tokenizer()
        def run_with_cache(self, tokens, attention_mask, return_type, names_filter=None):
            assert names_filter("blocks.0.hook_resid_pre")
            assert not names_filter("blocks.0.attn.hook_q")
            values = torch.tensor([
                [[90.0], [91.0], [10.0], [20.0]],
                [[80.0], [81.0], [30.0], [999.0]],
            ])
            return None, {"blocks.0.hook_resid_pre": values}

    captured = residual_at_response(FakeModel(), ["a", "b"], ["long", "short"], layer=0, site="response_mean")
    assert np.array_equal(captured, np.array([[15.0], [30.0]], dtype=np.float32))

def test_response_mean_supports_left_padding():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        chat_template = None
        pad_token_id = 0
        eos_token_id = 0
        padding_side = "left"

        def encode(self, text, add_special_tokens):
            if text.startswith("User:"):
                return [1, 2]
            return [3] if text == "short" else [3, 4]

    class FakeModel:
        tokenizer = Tokenizer()

        def run_with_cache(self, tokens, attention_mask, return_type, names_filter=None):
            values = torch.tensor([
                [[90.0], [91.0], [10.0], [20.0]],
                [[999.0], [80.0], [81.0], [30.0]],
            ])
            return None, {"blocks.0.hook_resid_pre": values}

    captured = residual_at_response(
        FakeModel(), ["a", "b"], ["long", "short"], layer=0, site="response_mean"
    )
    assert np.array_equal(captured, np.array([[15.0], [30.0]], dtype=np.float32))

def test_model_loader_uses_pinned_snapshot_and_bridge_factory(monkeypatch):
    import sys
    from types import ModuleType
    from safety_governor.models import load_transformerlens_model

    calls = {}
    hub = ModuleType("huggingface_hub")
    def snapshot_download(repo_id, revision):
        calls["snapshot"] = (repo_id, revision)
        return "immutable-snapshot"
    hub.snapshot_download = snapshot_download

    bridge_module = ModuleType("transformer_lens.model_bridge")
    class Bridge:
        @classmethod
        def boot_transformers(cls, snapshot, device, dtype):
            calls["boot"] = (snapshot, device, dtype)
            return cls()
        def enable_compatibility_mode(self, *, no_processing):
            calls["compatibility"] = {"no_processing": no_processing}
    bridge_module.TransformerBridge = Bridge
    torch_module = ModuleType("torch")
    torch_module.float32 = object()
    torch_module.float16 = object()
    torch_module.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "transformer_lens.model_bridge", bridge_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    load_transformerlens_model("model", "commit", "cpu", "bfloat16")
    assert calls == {
        "snapshot": ("model", "commit"),
        "boot": ("immutable-snapshot", "cpu", torch_module.bfloat16),
        "compatibility": {"no_processing": True},
    }


def test_model_loader_rejects_weight_processing_mode():
    from safety_governor.models import load_transformerlens_model

    with pytest.raises(ValueError, match="bridge weight mode"):
        load_transformerlens_model(
            "model", "commit", "cpu", "bfloat16", "processed_compatibility"
        )


def test_multi_layer_capture_returns_only_requested_layers():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        chat_template = None
        pad_token_id = 0
        eos_token_id = 0
        padding_side = "right"

        def encode(self, text, add_special_tokens):
            return [1, 2] if text.startswith("User:") else [3]

    class FakeModel:
        tokenizer = Tokenizer()

        def run_with_cache(self, tokens, attention_mask, return_type, names_filter):
            cache = {}
            for layer in (0, 4, 8):
                name = f"blocks.{layer}.hook_resid_pre"
                if names_filter(name):
                    cache[name] = torch.full((1, 3, 1), float(layer + 1))
            return None, cache

    captured = residuals_at_response(FakeModel(), ["question"], ["answer"], [0, 8])
    assert sorted(captured) == [0, 8]
    assert captured[0].item() == 1.0
    assert captured[8].item() == 9.0


def test_response_capture_moves_inputs_to_model_device():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        chat_template = None
        pad_token_id = 0
        eos_token_id = 0
        padding_side = "right"

        def encode(self, text, add_special_tokens):
            return [1, 2] if text.startswith("User:") else [3]

    class FakeModel:
        tokenizer = Tokenizer()

        def parameters(self):
            yield torch.nn.Parameter(torch.empty(1, device="meta"))

        def run_with_cache(self, tokens, attention_mask, return_type, names_filter):
            assert tokens.device.type == "meta"
            assert attention_mask.device.type == "meta"
            values = torch.ones((1, 3, 1))
            return None, {"blocks.0.hook_resid_pre": values}

    captured = residuals_at_response(FakeModel(), ["question"], ["answer"], [0])
    assert captured[0].shape == (1, 1)


def test_generation_steers_opposite_unsafe_direction_at_frontier():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        chat_template = None
        pad_token_id = 0
        eos_token_id = 9

        def encode(self, text, add_special_tokens):
            return [1, 2] if text.startswith("User:") else [3]

        def decode(self, tokens, skip_special_tokens):
            return "generated"

    class FakeModel:
        tokenizer = Tokenizer()

        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.activations = []

        def parameters(self):
            yield self.weight

        def run_with_hooks(self, tokens, attention_mask, return_type, fwd_hooks):
            activation = torch.zeros((1, tokens.shape[1], 2))
            self.activations.append(fwd_hooks[0][1](activation))
            logits = torch.zeros((1, tokens.shape[1], 10))
            logits[0, -1, 3 if len(self.activations) == 1 else 9] = 1
            return logits

    model = FakeModel()
    assert generate_with_steering(
        model, "prompt", np.array([1., 2.]), layer=0, magnitude=2,
        token_mode="generation_frontier", max_new_tokens=2,
    ) == "generated"
    assert torch.equal(model.activations[0][0, -1], torch.tensor([-2., -4.]))
    assert torch.equal(model.activations[1][0, -1], torch.tensor([-2., -4.]))
    assert torch.equal(model.activations[1][0, 0], torch.zeros(2))


def test_multi_layer_governor_spends_relative_l2_profile_at_frontier():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        chat_template = None
        pad_token_id = 0
        eos_token_id = 9

        def encode(self, text, add_special_tokens):
            return [1, 2] if text.startswith("User:") else [3]

        def decode(self, tokens, skip_special_tokens):
            return "generated" if tokens else ""

    class FakeModel:
        tokenizer = Tokenizer()

        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.by_layer = {}

        def parameters(self):
            yield self.weight

        def run_with_hooks(self, tokens, attention_mask, return_type, fwd_hooks):
            for name, hook in fwd_hooks:
                activation = torch.zeros((1, tokens.shape[1], 2))
                activation[0, -1] = torch.tensor([3., 4.])
                self.by_layer[name] = hook(activation)
            logits = torch.zeros((1, tokens.shape[1], 10))
            logits[0, -1, 9] = 1
            return logits

    trace = []
    metadata = {}
    model = FakeModel()
    sites = [
        SteeringSite(12, np.array([1., 0.]), 0.25),
        SteeringSite(24, np.array([0., 1.]), 0.75),
    ]
    assert generate_with_governor(
        model, "prompt", sites, total_relative_l2=0.2, max_new_tokens=1,
        trace=trace, generation_metadata=metadata,
    ) == ""
    assert metadata == {
        "generated_token_count": 0, "stop_reason": "stop_token", "stop_token_id": 9,
    }
    assert torch.allclose(
        model.by_layer["blocks.12.hook_resid_pre"][0, -1],
        torch.tensor([2.75, 4.0]),
    )
    assert torch.allclose(
        model.by_layer["blocks.24.hook_resid_pre"][0, -1],
        torch.tensor([3.0, 3.25]),
    )
    assert [round(row["relative_delta"], 3) for row in trace] == [0.05, 0.15]


def test_multi_layer_governor_records_token_limit():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        chat_template = None
        eos_token_id = 9

        def encode(self, text, add_special_tokens):
            return [1, 2]

        def decode(self, tokens, skip_special_tokens):
            return "x" * len(tokens)

    class FakeModel:
        tokenizer = Tokenizer()

        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            yield self.weight

        def run_with_hooks(self, tokens, attention_mask, return_type, fwd_hooks):
            activation = torch.tensor([[[3.0, 4.0]]]).expand(1, tokens.shape[1], 2)
            for _, hook in fwd_hooks:
                hook(activation.clone())
            logits = torch.zeros((1, tokens.shape[1], 10))
            logits[0, -1, 3] = 1
            return logits

    metadata = {}
    response = generate_with_governor(
        FakeModel(), "prompt", [SteeringSite(12, np.array([1., 0.]), 1.0)],
        total_relative_l2=0.2, max_new_tokens=2, generation_metadata=metadata,
    )
    assert response == "xx"
    assert metadata == {
        "generated_token_count": 2, "stop_reason": "token_limit", "stop_token_id": None,
    }


@pytest.mark.parametrize("stop_at_eos", [False, True])
def test_unsteered_generation_records_stop_reason(stop_at_eos):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        chat_template = None
        eos_token_id = 9

        def encode(self, text, add_special_tokens):
            return [1, 2]

        def decode(self, tokens, skip_special_tokens):
            return "x" * len(tokens)

    class FakeModel:
        tokenizer = Tokenizer()

        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def parameters(self):
            yield self.weight

        def __call__(self, tokens, attention_mask):
            self.calls += 1
            logits = torch.zeros((1, tokens.shape[1], 10))
            logits[0, -1, 9 if stop_at_eos and self.calls == 2 else 3] = 1
            return logits

    metadata = {}
    response = generate_unsteered(
        FakeModel(), "prompt", max_new_tokens=2, generation_metadata=metadata,
    )
    assert response == ("x" if stop_at_eos else "xx")
    assert metadata == {
        "generated_token_count": 1 if stop_at_eos else 2,
        "stop_reason": "stop_token" if stop_at_eos else "token_limit",
        "stop_token_id": 9 if stop_at_eos else None,
    }


def test_projection_gate_can_leave_activation_untouched():
    torch = pytest.importorskip("torch")
    from safety_governor.models import _governor_intervention

    site = SteeringSite(
        12,
        np.array([1., 0.]),
        1.0,
        gate=ProjectionGate(threshold=2.0, transition_width=1.0),
    )
    activation = torch.tensor([[[1., 2.]]])
    changed = _governor_intervention(site, 0, 0.5, None)(activation)
    assert torch.equal(changed, activation)
