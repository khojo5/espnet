"""Alignment records and Huang et al. (arXiv:2406.02560) boundary metrics."""

import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, TextIO

from splet.alignment import levenshtein_alignment

SILENCE_LABELS = frozenset({"", "sil", "sp", "spn", "<sil>", "<eps>"})
UNKNOWN_LABELS = frozenset({"<unk>", "<unknown>"})


def finite_number(value: Any, name: str) -> float:
    """Read a finite JSON number, not a string or boolean."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite: {value!r}")
    return result


@dataclass
class Interval:
    label: str
    start: float
    end: float
    score: Optional[float] = None
    word_index: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise ValueError("interval label must be a string")
        self.start = finite_number(self.start, "start")
        self.end = finite_number(self.end, "end")
        if not 0 <= self.start <= self.end:
            raise ValueError(f"invalid interval: {self.label} {self.start} {self.end}")
        if self.score is not None:
            self.score = finite_number(self.score, "score")
        if self.word_index is not None and (
            type(self.word_index) is not int or self.word_index < 0
        ):
            raise ValueError("word_index must be a non-negative integer")


@dataclass
class Alignment:
    utt_id: str
    duration_s: float
    words: Optional[list[Interval]] = None
    phones: Optional[list[Interval]] = None
    recording_id: Optional[str] = None
    crop_offset_s: float = 0.0
    pronunciations: Optional[list[list[str]]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.utt_id, str) or not self.utt_id:
            raise ValueError("utt_id must be a nonempty string")
        self.duration_s = finite_number(self.duration_s, "duration_s")
        self.crop_offset_s = finite_number(self.crop_offset_s, "crop_offset_s")
        if self.duration_s <= 0 or self.crop_offset_s < 0:
            raise ValueError(f"{self.utt_id}: invalid duration or crop offset")
        for tier in (self.words, self.phones):
            previous_end = 0.0
            for interval in tier or []:
                if interval.start < previous_end - 1e-6:
                    raise ValueError(f"{self.utt_id}: overlapping/unordered intervals")
                if interval.end > self.duration_s + 1e-6:
                    raise ValueError(f"{self.utt_id}: interval exceeds audio duration")
                previous_end = interval.end

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Alignment":
        def tier(name: str) -> Optional[list[Interval]]:
            values = data.get(name)
            if values is None:
                return None
            if not isinstance(values, list):
                raise ValueError(f"{name} must be an interval list")
            return [Interval(**value) for value in values]

        return cls(
            utt_id=data["utt_id"],
            duration_s=data["duration_s"],
            words=tier("words"),
            phones=tier("phones"),
            recording_id=data.get("recording_id"),
            crop_offset_s=data.get("crop_offset_s", 0.0),
            pronunciations=data.get("pronunciations"),
        )


def read_alignments(path: Path, allow_empty: bool = False) -> dict[str, Alignment]:
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("alignment record must be a JSON object")
                record = Alignment.from_dict(data)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if record.utt_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate ID {record.utt_id}")
            records[record.utt_id] = record
    if not records and not allow_empty:
        raise ValueError(f"{path}: no alignment records")
    return records


@contextmanager
def atomic_output(path: Path) -> Iterator[TextIO]:
    """Only publish a complete output; leave existing results intact on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.flush()
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    with atomic_output(path) as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def write_alignments(path: Path, records: Iterable[Alignment]) -> None:
    seen = set()
    with atomic_output(path) as stream:
        for record in records:
            if record.utt_id in seen:
                raise ValueError(f"duplicate output ID: {record.utt_id}")
            seen.add(record.utt_id)
            stream.write(
                json.dumps(asdict(record), ensure_ascii=False, allow_nan=False)
            )
            stream.write("\n")
        if not seen:
            raise ValueError("no alignment output records")


@dataclass
class TierScore:
    n_ref: int
    n_hyp: int
    n_matched: int
    n_excluded_ref: int
    n_excluded_hyp: int
    boundary_error_ms: Optional[float]
    duration_ref_sum_ms: float
    duration_hyp_sum_ms: float
    reason: Optional[str]


def score_tier(
    reference: Optional[list[Interval]],
    hypothesis: Optional[list[Interval]],
    *,
    words: bool = False,
    phone_map: Optional[Mapping[str, str]] = None,
    silence_labels: Iterable[str] = SILENCE_LABELS,
) -> TierScore:
    """Match by labels only, then average both timestamp errors per unit."""
    silence = {label.lower() for label in silence_labels}

    def prepare(tier: Optional[list[Interval]]) -> tuple[list[Interval], list[str]]:
        kept, labels = [], []
        for interval in tier or []:
            label = interval.label.lower() if words else interval.label
            if not words and phone_map is not None:
                label = phone_map.get(label, label)
            if (
                label.lower() in silence | UNKNOWN_LABELS
                or interval.start == interval.end
            ):
                continue
            kept.append(interval)
            labels.append(label)
        return kept, labels

    ref, ref_labels = prepare(reference)
    hyp, hyp_labels = prepare(hypothesis)
    alignment = levenshtein_alignment(ref_labels, hyp_labels, backend="python")
    pairs = []
    i = j = 0
    for operation, _, _ in alignment.operations:
        if operation == "hit":
            pairs.append((ref[i], hyp[j]))
        i += operation != "ins"
        j += operation != "del"
    error = sum((abs(r.start - h.start) + abs(r.end - h.end)) / 2 for r, h in pairs)
    reason = None
    if reference is None:
        reason = "reference_tier_unavailable"
    elif hypothesis is None:
        reason = "hypothesis_tier_unavailable"
    elif not pairs:
        reason = "no_label_matched_units"
    return TierScore(
        n_ref=len(ref),
        n_hyp=len(hyp),
        n_matched=len(pairs),
        n_excluded_ref=len(reference or []) - len(ref),
        n_excluded_hyp=len(hypothesis or []) - len(hyp),
        boundary_error_ms=1000 * error / len(pairs) if pairs else None,
        duration_ref_sum_ms=1000 * sum(r.end - r.start for r, _ in pairs),
        duration_hyp_sum_ms=1000 * sum(h.end - h.start for _, h in pairs),
        reason=reason,
    )


def summarize_tier(scores: Iterable[TierScore], prefix: str) -> dict[str, Any]:
    """Boundary error is utterance-macro; duration is matched-unit-micro."""
    scores = list(scores)
    eligible = [s.boundary_error_ms for s in scores if s.boundary_error_ms is not None]
    matched = sum(s.n_matched for s in scores)
    return {
        f"{prefix}BE_ms": sum(eligible) / len(eligible) if eligible else None,
        f"{prefix}DUR_ref_ms": (
            sum(s.duration_ref_sum_ms for s in scores) / matched if matched else None
        ),
        f"{prefix}DUR_hyp_ms": (
            sum(s.duration_hyp_sum_ms for s in scores) / matched if matched else None
        ),
        "n_eligible_utts": len(eligible),
        "n_excluded_utts": len(scores) - len(eligible),
        "n_ref": sum(s.n_ref for s in scores),
        "n_hyp": sum(s.n_hyp for s in scores),
        "n_matched": matched,
        "n_unmatched_ref": sum(s.n_ref for s in scores) - matched,
        "n_unmatched_hyp": sum(s.n_hyp for s in scores) - matched,
        "n_excluded_ref": sum(s.n_excluded_ref for s in scores),
        "n_excluded_hyp": sum(s.n_excluded_hyp for s in scores),
        "reason": None if eligible else "no_eligible_utterances",
    }
