import json

import pytest

from espnet2.utils.alignment import (
    Alignment,
    Interval,
    read_alignments,
    score_tier,
    summarize_tier,
    write_alignments,
)


def test_paper_macro_error_and_pooled_duration():
    first = score_tier([Interval("a", 0, 0.2)], [Interval("a", 0.1, 0.4)])
    second = score_tier(
        [Interval("a", 0, 0.1), Interval("b", 0.1, 0.2)],
        [Interval("a", 0, 0.1), Interval("b", 0.1, 0.2)],
    )
    result = summarize_tier([first, second], "P")
    assert result["PBE_ms"] == pytest.approx(75)
    assert result["PDUR_ref_ms"] == pytest.approx(400 / 3)
    assert result["PDUR_hyp_ms"] == pytest.approx(500 / 3)
    assert result["n_matched"] == 3


def test_shared_boundaries_are_two_unit_timestamps():
    result = score_tier(
        [Interval("a", 0, 0.2), Interval("b", 0.2, 0.4)],
        [Interval("a", 0, 0.1), Interval("b", 0.3, 0.4)],
    )
    assert result.boundary_error_ms == pytest.approx(50)


def test_label_matching_exclusions_and_missing_tiers():
    ref = [
        Interval("A", 0, 0.1),
        Interval("sil", 0.1, 0.2),
        Interval("<unk>", 0.2, 0.3),
        Interval("B", 0.3, 0.4),
    ]
    result = score_tier(ref, [Interval("a", 0, 0.1)], words=True)
    assert result.n_matched == 1
    assert result.n_ref == 2
    assert result.n_excluded_ref == 2
    assert result.boundary_error_ms == 0
    missing = summarize_tier([score_tier(ref, None)], "P")
    assert missing["PBE_ms"] is None
    assert missing["PDUR_hyp_ms"] is None
    assert missing["n_unmatched_ref"] == 2


def test_phone_mapping_is_explicit_and_matching_is_not_time_based():
    ref = [Interval("AH0", 0, 0.1), Interval("AH0", 0.8, 0.9)]
    hyp = [Interval("AH", 0.8, 0.9)]
    assert score_tier(ref, hyp).n_matched == 0
    result = score_tier(ref, hyp, phone_map={"AH0": "AH"})
    assert result.n_matched == 1
    assert result.n_ref == 2
    # The alignment's fixed traceback, not time proximity, selects the pair.
    shifted = score_tier(ref, [Interval("AH", 0, 0.1)], phone_map={"AH0": "AH"})
    assert result.boundary_error_ms + shifted.boundary_error_ms == pytest.approx(800)


@pytest.mark.parametrize(
    "intervals",
    [
        [{"label": "a", "start": 0.2, "end": 0.1}],
        [{"label": "a", "start": float("nan"), "end": 0.1}],
        [{"label": "a", "start": 0, "end": 2}],
        [
            {"label": "a", "start": 0, "end": 0.5},
            {"label": "b", "start": 0.4, "end": 0.6},
        ],
    ],
)
def test_invalid_interval_records_are_rejected(tmp_path, intervals):
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"utt_id": "u", "duration_s": 1, "phones": intervals}))
    with pytest.raises(ValueError, match="invalid.jsonl:1"):
        read_alignments(path)


def test_atomic_publication_and_duplicate_records(tmp_path):
    path = tmp_path / "output.jsonl"
    original = Alignment("u", 1, phones=[Interval("a", 0.1, 0.2)])
    write_alignments(path, [original])
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        write_alignments(path, [original, original])
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
    path.write_bytes(previous + previous)
    with pytest.raises(ValueError, match="duplicate"):
        read_alignments(path)
