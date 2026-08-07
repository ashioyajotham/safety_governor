from safety_governor.data import validate_records
from safety_governor.domain import Behavior, ContrastiveRecord, Polarity


def record(polarity, **kwargs):
    return ContrastiveRecord(
        "p1", Behavior.HARMFUL_COMPLIANCE, polarity, "en",
        kwargs.pop("prompt", polarity.value), "refuse", "fixture", "approved",
        split=kwargs.pop("split", "train"), source_group_id=kwargs.pop("source_group_id", "g1"),
        **kwargs,
    )


def test_valid_contrastive_pair():
    assert validate_records([record(Polarity.SAFE), record(Polarity.UNSAFE)]) == []


def test_detects_split_leakage():
    assert any("split leakage" in e for e in validate_records([
        record(Polarity.SAFE), record(Polarity.UNSAFE, split="test")
    ]))


def test_detects_source_group_leakage_across_pairs():
    records = [
        record(Polarity.SAFE),
        record(Polarity.UNSAFE),
        ContrastiveRecord("p2", Behavior.HARMFUL_COMPLIANCE, Polarity.SAFE, "en", "x", "refuse", "fixture", "approved", split="test", source_group_id="g1"),
        ContrastiveRecord("p2", Behavior.HARMFUL_COMPLIANCE, Polarity.UNSAFE, "en", "y", "refuse", "fixture", "approved", split="test", source_group_id="g1"),
    ]
    assert any("source-group split leakage" in error for error in validate_records(records))