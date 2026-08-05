from safety_governor.data import validate_records
from safety_governor.domain import Behavior, ContrastiveRecord, Polarity


def record(polarity, **kwargs):
    return ContrastiveRecord("p1", Behavior.HARMFUL_COMPLIANCE, polarity, "en", kwargs.pop("prompt", polarity.value), "refuse", "fixture", "approved", **kwargs)


def test_valid_contrastive_pair():
    assert validate_records([record(Polarity.SAFE), record(Polarity.UNSAFE)]) == []


def test_detects_split_leakage():
    assert any("split leakage" in e for e in validate_records([
        record(Polarity.SAFE), record(Polarity.UNSAFE, split="test")
    ]))
