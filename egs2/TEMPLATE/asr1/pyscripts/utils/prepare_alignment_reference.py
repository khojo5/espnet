"""Convert utterance-relative TextGrid intervals to alignment reference JSONL."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from espnet2.fileio.read_text import read_2columns_text
from espnet2.utils.alignment import (
    Alignment,
    Interval,
    atomic_output,
    write_alignments,
    write_json,
)

# Same exclusions as the phoneme training targets (local/textgrid_to_phoneme.py).
NON_PHONE_LABELS = frozenset({"", "sil", "sp"})


def word_spans(uid: str, words: list[str], labels: list[str]) -> list[tuple]:
    """Map transcript words to consecutive TextGrid word intervals.

    A TextGrid word matches a transcript word case-insensitively, and "<unk>"
    matches any word. MFA may split a word at apostrophes (QUILTER'S ->
    quilter, s), so the apostrophe-separated parts are tried as well.
    """

    def candidates(word: str) -> list[list[str]]:
        parts = [part for part in word.split("'") if part]
        return [[word]] + ([parts] if parts and parts != [word] else [])

    def matches(label: str, part: str) -> bool:
        return label == "<unk>" or label.lower() == part.lower()

    # reachable[i][j]: words[i:] can consume labels[j:]
    reachable = [[False] * (len(labels) + 1) for _ in range(len(words) + 1)]
    reachable[len(words)][len(labels)] = True
    for i in range(len(words) - 1, -1, -1):
        for j in range(len(labels)):
            reachable[i][j] = any(
                j + len(parts) <= len(labels)
                and all(matches(lab, p) for lab, p in zip(labels[j:], parts))
                and reachable[i + 1][j + len(parts)]
                for parts in candidates(words[i])
            )
    if not reachable[0][0]:
        raise ValueError(f"{uid}: TextGrid words do not match the transcript")
    spans, j = [], 0
    for i, word in enumerate(words):
        for parts in candidates(word):
            end = j + len(parts)
            if (
                end <= len(labels)
                and all(matches(lab, p) for lab, p in zip(labels[j:], parts))
                and reachable[i + 1][end]
            ):
                spans.append((j, end))
                j = end
                break
    return spans


def group_pronunciations(
    uid: str, words: list[str], word_tier: list, phone_tier: list
) -> list[list[str]]:
    """Word-grouped TextGrid phones for transcript words (labels only)."""
    entries = [entry for entry in word_tier if entry.label.strip()]
    phones = [
        entry
        for entry in phone_tier
        if entry.label.strip().lower() not in NON_PHONE_LABELS
    ]
    spans = word_spans(uid, words, [entry.label.strip() for entry in entries])
    grouped, assigned = [], 0
    for word, (first, last) in zip(words, spans):
        start, end = entries[first].start, entries[last - 1].end
        group = [
            phone.label.strip()
            for phone in phones
            if phone.start >= start - 1e-6 and phone.end <= end + 1e-6
        ]
        if not group:
            raise ValueError(f"{uid}: no TextGrid phones within word {word!r}")
        grouped.append(group)
        assigned += len(group)
    if assigned != len(phones):
        raise ValueError(f"{uid}: TextGrid phones outside or across word intervals")
    return grouped


def find_textgrids(
    textgrid_root: Path, textgrid_scp: Optional[Path]
) -> dict[str, Path]:
    """Map utterance IDs to TextGrid paths (file stem = ID unless an scp is given)."""
    if textgrid_scp is not None:
        return {
            uid: Path(path) for uid, path in read_2columns_text(textgrid_scp).items()
        }
    paths = {}
    for path in sorted(textgrid_root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".textgrid":
            if path.stem in paths:
                raise ValueError(f"duplicate TextGrid ID {path.stem}: {path}")
            paths[path.stem] = path
    return paths


def read_textgrid(
    uid: str, path: Path, tier_names: tuple[str, str]
) -> tuple[float, list[Interval], list[Interval]]:
    """Return (duration, word intervals, phone intervals), keeping empty labels."""
    from praatio import textgrid

    grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
    if grid.minTimestamp != 0:
        raise ValueError(f"{uid}: TextGrid must be utterance-relative (start=0)")
    tiers = []
    for name in tier_names:
        if name not in grid.tierNames:
            raise ValueError(f"{uid}: TextGrid missing tier {name!r}")
        tiers.append(
            [
                Interval(entry.label, entry.start, entry.end)
                for entry in grid.getTier(name).entries
            ]
        )
    word_tier, phone_tier = tiers
    return grid.maxTimestamp, word_tier, phone_tier


def main(cmd=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrid-root", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True, help="IDs to include")
    parser.add_argument(
        "--textgrid-scp", type=Path, help="Explicit ID-to-TextGrid mapping"
    )
    parser.add_argument("--word-tier", default="words")
    parser.add_argument("--phone-tier", default="phones")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pronunciation-output",
        type=Path,
        help="Also write word-grouped TextGrid phones for alignment_inference",
    )
    parser.add_argument("--key-output", type=Path, help="Write the converted IDs")
    args = parser.parse_args(cmd)
    logging.basicConfig(level=logging.INFO)

    transcripts = read_2columns_text(args.text)
    if not transcripts:
        raise ValueError("reference ID selection is empty")
    paths = find_textgrids(args.textgrid_root, args.textgrid_scp)
    converted = [uid for uid in transcripts if uid in paths]
    missing = sorted(transcripts.keys() - paths.keys())
    extra = sorted(paths.keys() - transcripts.keys())
    if missing:
        logging.warning("No TextGrid for %d utterances: %s", len(missing), missing)

    references, pronunciations = [], []
    for uid in converted:
        duration, word_tier, phone_tier = read_textgrid(
            uid, paths[uid], (args.word_tier, args.phone_tier)
        )
        references.append(Alignment(uid, duration, word_tier, phone_tier))
        if args.pronunciation_output is not None:
            words = transcripts[uid].split()
            phones = group_pronunciations(uid, words, word_tier, phone_tier)
            pronunciations.append({"utt_id": uid, "words": words, "phones": phones})

    write_alignments(args.output, references)
    if args.pronunciation_output is not None:
        with atomic_output(args.pronunciation_output) as stream:
            for record in pronunciations:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.key_output is not None:
        with atomic_output(args.key_output) as stream:
            stream.writelines(f"{uid}\n" for uid in converted)
    write_json(
        args.output.with_suffix(".coverage.json"),
        {
            "n_requested": len(transcripts),
            "n_converted": len(converted),
            "missing_textgrid_ids": missing,
            "extra_textgrid_ids": extra,
            "time_origin": "utterance",
            "word_tier": args.word_tier,
            "phone_tier": args.phone_tier,
        },
    )


if __name__ == "__main__":
    main()
