"""Score saved word/phone alignments without importing a model or torch."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from espnet2.utils.alignment import (
    SILENCE_LABELS,
    Alignment,
    atomic_output,
    read_alignments,
    score_tier,
    summarize_tier,
    write_json,
)

TIERS = (("words", "W"), ("phones", "P"))


def read_hypotheses(paths: list[Path]) -> dict[str, Alignment]:
    """Merge per-job hypothesis files; an ID may appear in only one file."""
    hypotheses = {}
    for path in paths:
        records = read_alignments(path, allow_empty=True)
        overlap = hypotheses.keys() & records.keys()
        if overlap:
            raise ValueError(
                f"duplicate hypothesis IDs across files: {sorted(overlap)}"
            )
        hypotheses.update(records)
    return hypotheses


def read_phone_map(path: Optional[Path]) -> Optional[dict[str, str]]:
    if path is None:
        return None
    with path.open(encoding="utf-8") as stream:
        phone_map = json.load(stream)
    is_string_map = isinstance(phone_map, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in phone_map.items()
    )
    if not is_string_map:
        raise ValueError("--phone-map must contain a JSON string-to-string map")
    return phone_map


def check_same_time_axis(ref: Alignment, hyp: Alignment) -> None:
    """Reference and hypothesis must describe the same audio span."""
    same_duration = abs(ref.duration_s - hyp.duration_s) <= 0.001
    same_crop = ref.recording_id is None or (
        ref.recording_id == hyp.recording_id
        and abs(ref.crop_offset_s - hyp.crop_offset_s) <= 1e-6
    )
    if not (same_duration and same_crop):
        raise ValueError(f"{ref.utt_id}: reference/hypothesis time coordinates differ")


def write_summary_tsv(path: Path, summary: dict) -> None:
    with atomic_output(path) as stream:
        stream.write("tier\tBE_ms\tDUR_ref_ms\tDUR_hyp_ms\tn_matched\n")
        for tier, prefix in TIERS:
            row = summary[tier]
            values = [
                row[f"{prefix}BE_ms"],
                row[f"{prefix}DUR_ref_ms"],
                row[f"{prefix}DUR_hyp_ms"],
                row["n_matched"],
            ]
            cells = ["NA" if v is None else str(v) for v in values]
            stream.write("\t".join([tier, *cells]) + "\n")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--hypothesis", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phone-map", type=Path, help="JSON mapping for both phone label inventories"
    )
    parser.add_argument("--silence-labels", nargs="*", default=sorted(SILENCE_LABELS))
    return parser


def main(cmd=None) -> None:
    args = get_parser().parse_args(cmd)
    logging.basicConfig(level=logging.INFO)
    reference = read_alignments(args.reference)
    hypothesis = read_hypotheses(args.hypothesis)
    phone_map = read_phone_map(args.phone_map)
    missing = sorted(reference.keys() - hypothesis.keys())
    extra = sorted(hypothesis.keys() - reference.keys())
    if extra:
        logging.warning("No reference for %d hypothesis IDs: %s", len(extra), extra)

    scores = {tier: [] for tier, _ in TIERS}
    n_empty = 0
    with atomic_output(args.output_dir / "per_utterance.jsonl") as stream:
        for uid, ref in reference.items():
            hyp = hypothesis.get(uid)
            if hyp is not None:
                check_same_time_axis(ref, hyp)
                n_empty += not hyp.words and not hyp.phones
            result = {"utt_id": uid, "missing_hypothesis": hyp is None}
            for tier, _ in TIERS:
                score = score_tier(
                    getattr(ref, tier),
                    getattr(hyp, tier) if hyp is not None else None,
                    words=tier == "words",
                    phone_map=phone_map,
                    silence_labels=args.silence_labels,
                )
                scores[tier].append(score)
                result[tier] = asdict(score)
            stream.write(json.dumps(result, allow_nan=False) + "\n")

    summary = {
        "boundary_averaging": "unit_then_utterance_then_corpus",
        "duration_averaging": "matched_units_pooled",
        "n_reference_utts": len(reference),
        "missing_hypothesis_ids": missing,
        "hypothesis_ids_without_reference": extra,
        "n_empty_hypotheses": n_empty,
        **{tier: summarize_tier(scores[tier], prefix) for tier, prefix in TIERS},
        "phone_map": phone_map,
        "silence_labels": args.silence_labels,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_summary_tsv(args.output_dir / "summary.tsv", summary)
    logging.info(
        "Scored %d utterances; missing hypotheses: %d", len(reference), len(missing)
    )


if __name__ == "__main__":
    main()
