from types import SimpleNamespace

import numpy as np
import pytest

from safety_governor.models import (
    generate_with_steering,
    residual_at_last_token,
    residual_at_response,
    residuals_at_response,
)


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
