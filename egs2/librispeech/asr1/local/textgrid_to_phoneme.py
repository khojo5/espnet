"""Extract phone sequences for utterance IDs selected by Kaldi text files."""

import argparse
import logging
from pathlib import Path

from praatio import textgrid

from espnet2.fileio.read_text import read_2columns_text


def extract_phones(path: Path, utterance_id: str, phone_tier: str) -> list[str]:
    grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
    if phone_tier not in grid.tierNames:
        raise ValueError(f"{utterance_id}: {path} is missing tier {phone_tier!r}")
    tier = grid.getTier(phone_tier)
    if not isinstance(tier, textgrid.IntervalTier):
        raise ValueError(
            f"{utterance_id}: {path}: {phone_tier!r} is not an interval tier"
        )

    phones = []
    for interval in sorted(tier.entries, key=lambda entry: entry.start):
        label = interval.label.strip()
        if label.lower() in {"", "sil", "sp"}:
            continue
        if any(character.isspace() for character in label):
            raise ValueError(f"{utterance_id}: {path}: invalid phone label {label!r}")
        phones.append(label)
    if not phones:
        raise ValueError(f"{utterance_id}: {path}: empty phoneme sequence")
    return phones


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrid-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, nargs="+", required=True)
    parser.add_argument("--phone-tier", default="phones")
    parser.add_argument("--missing-textgrid", choices=("error", "g2p"), default="error")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.textgrid_root.is_dir():
        parser.error(f"TextGrid directory not found: {args.textgrid_root}")
    if len(args.input) != len(args.output):
        parser.error("--input and --output must have the same number of paths")
    input_paths = {path.resolve() for path in args.input}
    report_paths = (
        [path.with_name(path.name + ".g2p_fallback_utt_ids") for path in args.output]
        if args.missing_textgrid == "g2p"
        else []
    )
    generated_paths = args.output + report_paths
    output_paths = [path.resolve() for path in generated_paths]
    if input_paths.intersection(output_paths) or len(set(output_paths)) != len(
        output_paths
    ):
        parser.error("Output paths must be distinct and different from all inputs")
    for path in generated_paths:
        if path.exists() or path.is_symlink():
            parser.error(f"Output already exists: {path}")

    ids_per_input = [list(read_2columns_text(path)) for path in args.input]
    for path, ids in zip(args.input, ids_per_input, strict=True):
        if not ids:
            raise ValueError(f"Empty utterance selection: {path}")
    selected_ids = {uid for ids in ids_per_input for uid in ids}
    paths: dict[str, Path] = {}
    logging.info("Searching for TextGrids in %s", args.textgrid_root)
    for path in args.textgrid_root.rglob("*"):
        if (
            path.suffix.lower() != ".textgrid"
            or path.stem not in selected_ids
            or not path.is_file()
        ):
            continue
        if path.stem in paths:
            raise ValueError(
                f"Duplicate TextGrid ID {path.stem}: {paths[path.stem]} and {path}"
            )
        paths[path.stem] = path
    missing = sorted(selected_ids - paths.keys())
    if missing and args.missing_textgrid == "error":
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing TextGrids for {len(missing)} utterances: {preview}")
    logging.info(
        "Found TextGrids for %d/%d selected utterances", len(paths), len(selected_ids)
    )
    g2p = None
    if missing:
        from espnet2.text.phoneme_tokenizer import G2p_en

        logging.warning("Using English G2P for %d missing TextGrids", len(missing))
        g2p = G2p_en(no_space=True)

    for input_path, ids, output in zip(
        args.input, ids_per_input, args.output, strict=True
    ):
        fallback_ids = [uid for uid in ids if uid not in paths]
        fallback_text = (
            read_2columns_text(input_path, keys_to_load=set(fallback_ids))
            if fallback_ids
            else {}
        )
        with output.open("x", encoding="utf-8") as fout:
            for uid in ids:
                if uid in paths:
                    phones = extract_phones(paths[uid], uid, args.phone_tier)
                else:
                    if g2p is None:
                        raise RuntimeError(f"{uid}: missing TextGrid without G2P")
                    if not fallback_text[uid].strip():
                        raise ValueError(f"{uid}: empty transcription for G2P")
                    logging.warning("G2P fallback for missing TextGrid: %s", uid)
                    phones = g2p(fallback_text[uid])
                    if not phones or any(
                        not phone.strip()
                        or any(character.isspace() for character in phone)
                        for phone in phones
                    ):
                        raise ValueError(
                            f"{uid}: empty or invalid G2P phoneme sequence"
                        )
                fout.write(f"{uid} {' '.join(phones)}\n")
        if args.missing_textgrid == "g2p":
            report = output.with_name(output.name + ".g2p_fallback_utt_ids")
            with report.open("x", encoding="utf-8") as fout:
                fout.writelines(f"{uid}\n" for uid in fallback_ids)
            logging.info(
                "Recorded %d G2P fallback IDs in %s", len(fallback_ids), report
            )
        logging.info("Wrote %d phoneme sequences to %s", len(ids), output)


if __name__ == "__main__":
    main()
