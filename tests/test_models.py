from types import SimpleNamespace

import numpy as np
import pytest

from safety_governor.models import residual_at_last_token, residual_at_response


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
        def run_with_cache(self, tokens, attention_mask, return_type):
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

        def run_with_cache(self, tokens, attention_mask, return_type):
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
        def boot_transformers(cls, snapshot, device):
            calls["boot"] = (snapshot, device)
            return cls()
        def enable_compatibility_mode(self):
            calls["compatibility"] = True
    bridge_module.TransformerBridge = Bridge
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "transformer_lens.model_bridge", bridge_module)

    load_transformerlens_model("model", "commit", "cpu")
    assert calls == {
        "snapshot": ("model", "commit"),
        "boot": ("immutable-snapshot", "cpu"),
        "compatibility": True,
    }