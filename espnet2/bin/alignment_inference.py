"""Phoneme CTC forced alignment of Kaldi-style audio and transcripts.

Each utterance is aligned with explicit word-grouped pronunciations
(``--pronunciations``), and the word and phone intervals are written as JSONL.
"""

import argparse
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import soundfile

from espnet2.bin.align import ForcedAligner
from espnet2.fileio.read_text import read_2columns_text
from espnet2.fileio.segments import seconds_to_sample
from espnet2.text.cleaner import TextCleaner
from espnet2.utils.alignment import Alignment, Interval, write_alignments, write_json

RESERVED_TOKENS = {"<unk>", "<sos/eos>", "<sos>", "<eos>"}


def read_segments(path: Path) -> dict[str, tuple[str, float, float]]:
    """Read a Kaldi segments file as {utt_id: (recording_id, start_s, end_s)}."""
    segments = {}
    for uid, value in read_2columns_text(path).items():
        fields = value.split()
        if len(fields) != 3:
            raise ValueError(f"{path}: {uid}: expected recording start end")
        recording = fields[0]
        start, end = float(fields[1]), float(fields[2])
        is_finite = np.isfinite(start) and np.isfinite(end)
        is_ordered = start >= 0 and (end > start or end == -1)
        if not (is_finite and is_ordered):
            raise ValueError(f"{path}: {uid}: invalid segment bounds")
        segments[uid] = (recording, start, end)
    return segments


class AudioReader:
    """Read waveforms from wav.scp (files, pipes, or Kaldi waveform archives)."""

    def __init__(
        self, path: Path, audio_type: str, input_fs: Optional[int] = None
    ) -> None:
        self.entries = read_2columns_text(path)
        self.input_fs = input_fs
        self.ark = None
        if audio_type == "kaldi_ark":
            from omniio import kaldi

            self.ark = kaldi.load_scp(str(path))

    def _load(self, recording: str) -> tuple[np.ndarray, int]:
        """Return the whole recording as (audio, rate)."""
        if self.ark is not None:
            value = self.ark[recording]
            if not isinstance(value, tuple):
                if self.input_fs is None:
                    raise ValueError("raw waveform ark requires --input-fs")
                return value, self.input_fs
            if isinstance(value[0], int):
                rate, audio = value
            else:
                audio, rate = value
            return audio, rate

        entry = self.entries[recording]
        if entry.endswith("|"):
            from omniio import kaldi

            with kaldi.open_like_kaldi(entry, "rb") as stream:
                return soundfile.read(BytesIO(stream.read()), dtype="float32")
        return soundfile.read(entry, dtype="float32")

    def read(
        self, recording: str, start: float = 0.0, end: float = -1.0
    ) -> tuple[np.ndarray, int, float]:
        """Return (mono segment audio, rate, segment start in seconds)."""
        # audio: (samples,) or (samples, channels)
        audio, rate = self._load(recording)
        is_waveform = isinstance(audio, np.ndarray) and (
            audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] in (1, 2))
        )
        if not isinstance(rate, (int, np.integer)) or rate <= 0 or not is_waveform:
            raise ValueError(f"{recording}: expected mono/stereo waveform and rate")
        rate = int(rate)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)  # (samples,)

        first = seconds_to_sample(start, rate)
        last = len(audio) if end == -1 else seconds_to_sample(end, rate)
        if not 0 <= first < last <= len(audio):
            raise ValueError(f"{recording}: segment outside recording")
        audio = audio[first:last].astype(np.float32)  # (segment_samples,)
        if not np.isfinite(audio).all():
            raise ValueError(f"{recording}: waveform contains non-finite samples")
        return audio, rate, first / rate


def check_phones(
    aligner: ForcedAligner, uid: str, word: str, phones: list[str]
) -> None:
    """Every phone must be a non-reserved model token kept intact by the tokenizer."""
    vocabulary = aligner.network.token_list
    reserved = RESERVED_TOKENS | {vocabulary[aligner.network.blank_id]}
    is_valid = phones and all(
        isinstance(p, str) and p in vocabulary and p not in reserved for p in phones
    )
    if not is_valid:
        raise ValueError(f"{uid}: {word!r}: empty, unknown or reserved phones {phones}")
    if aligner.model.tokenizer.text2tokens(" ".join(phones)) != phones:
        raise ValueError(f"{uid}: model tokenizer changes supplied phones {phones}")


def align_phones(
    aligner: ForcedAligner,
    uid: str,
    audio: np.ndarray,
    words: list[str],
    pronunciations: list[list[str]],
    duration: float,
    recording: str,
    offset: float,
) -> Alignment:
    """Align word-grouped phones once, retaining both interval tiers.

    Args:
        audio: (samples,) waveform at the model sample rate.
        pronunciations: One phone list per word.
    """
    if not words or len(words) != len(pronunciations):
        raise ValueError(f"{uid}: words and pronunciations must correspond")
    for word, phones in zip(words, pronunciations):
        check_phones(aligner, uid, word, phones)

    # One aligner "utterance" per word, so each segment is a word interval.
    segments = aligner(audio, [" ".join(phones) for phones in pronunciations])
    if len(segments) != len(words):
        raise RuntimeError(f"{uid}: aligner returned a different number of words")

    word_intervals, phone_intervals = [], []
    for index, (word, phones, segment) in enumerate(
        zip(words, pronunciations, segments)
    ):
        if [token.text for token in segment.tokens] != phones:
            raise RuntimeError(f"{uid}: aligner output does not match target phones")
        word_intervals.append(
            Interval(word, segment.start, segment.end, segment.score, index)
        )
        for token in segment.tokens:
            phone_intervals.append(
                Interval(token.text, token.start, token.end, token.score, index)
            )
    return Alignment(
        uid,
        duration,
        word_intervals,
        phone_intervals,
        recording,
        offset,
        pronunciations,
    )


def read_pronunciations(path: Path) -> dict[str, dict]:
    """Read {"utt_id", "words", "phones": [[phone, ...] per word]} JSONL."""
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            data = json.loads(line)
            if not isinstance(data, dict) or not isinstance(data.get("utt_id"), str):
                raise ValueError(f"{path}:{line_number}: expected utterance object")
            uid = data["utt_id"]
            if uid in records:
                raise ValueError(f"{path}:{line_number}: duplicate ID {uid}")
            words, phones = data.get("words"), data.get("phones")
            has_words = isinstance(words, list) and all(
                isinstance(w, str) for w in words
            )
            has_phones = isinstance(phones, list) and all(
                isinstance(group, list) and all(isinstance(p, str) for p in group)
                for group in phones
            )
            if not (has_words and has_phones):
                raise ValueError(
                    f"{path}:{line_number}: expected words and grouped phones"
                )
            records[uid] = data
    return records


def load_aligner(args: argparse.Namespace) -> ForcedAligner:
    """Load a phoneme CTC ASR model and set its input sample rate."""
    if args.model_tag:
        if args.asr_train_config or args.asr_model_file:
            raise ValueError("choose model tag OR config/checkpoint")
        aligner = ForcedAligner.from_pretrained(args.model_tag, device=args.device)
    else:
        if args.asr_train_config is None or args.asr_model_file is None:
            raise ValueError(
                "both --asr-train-config and --asr-model-file are required"
            )
        aligner = ForcedAligner(
            device=args.device,
            asr_train_config=str(args.asr_train_config),
            asr_model_file=str(args.asr_model_file),
            ctc_weight=1.0,
            beam_size=1,
        )
    if aligner.s2t:
        raise ValueError("this baseline requires an ASR phoneme CTC model")

    training = aligner.model.asr_train_args
    frontend_rate = (getattr(training, "frontend_conf", None) or {}).get("fs")
    if frontend_rate is not None:
        if isinstance(frontend_rate, str):
            import humanfriendly

            frontend_rate = humanfriendly.parse_size(frontend_rate)
        if frontend_rate <= 0:
            raise ValueError("the checkpoint frontend sample rate must be positive")
        # ASR Speech2Text does not expose this metadata to ForcedAligner.
        aligner.model.sample_rate = int(frontend_rate)

    # Phone sequences may also be trained as space-separated "word" tokens.
    if training.token_type not in ("phn", "word"):
        raise ValueError("expected token_type=phn or word (phone tokens)")
    return aligner


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-scp", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument(
        "--pronunciations",
        type=Path,
        required=True,
        help="Word-grouped phone JSONL",
    )
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--key-file", type=Path, help="Align only these IDs")
    parser.add_argument("--audio-type", choices=["sound", "kaldi_ark"], default="sound")
    parser.add_argument("--input-fs", type=int)
    parser.add_argument("--asr-train-config", type=Path)
    parser.add_argument("--asr-model-file", type=Path)
    parser.add_argument("--model-tag")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(cmd=None) -> None:
    args = get_parser().parse_args(cmd)
    logging.basicConfig(level=logging.INFO)
    if args.input_fs is not None and args.input_fs <= 0:
        raise ValueError("--input-fs must be positive")
    aligner = load_aligner(args)

    # Inputs: text and audio IDs must agree; keys select a subset of them.
    text = read_2columns_text(args.text)
    audio = AudioReader(args.wav_scp, args.audio_type, args.input_fs)
    segments = read_segments(args.segments) if args.segments is not None else None
    if segments is None:
        audio_ids = audio.entries.keys()
    else:
        audio_ids = segments.keys()
        if any(rec not in audio.entries for rec, _, _ in segments.values()):
            raise ValueError("segments refer to recordings absent from wav.scp")
    if text.keys() != audio_ids:
        raise ValueError("text IDs must equal audio/segment IDs")

    keys = list(read_2columns_text(args.key_file)) if args.key_file else list(text)
    if not keys or set(keys) - text.keys():
        raise ValueError("empty key selection or keys absent from text")
    pronunciations = read_pronunciations(args.pronunciations)
    if pronunciations.keys() - text.keys() or set(keys) - pronunciations.keys():
        raise ValueError(
            "pronunciation IDs must be within text IDs and cover the selected keys"
        )
    cleaner = TextCleaner(getattr(aligner.model.asr_train_args, "cleaner", None))

    def records() -> Iterator[Alignment]:
        for uid in keys:
            logging.info("Aligning %s", uid)
            words = cleaner(text[uid]).split()
            if pronunciations[uid]["words"] != words:
                raise ValueError(f"{uid}: pronunciation words differ from cleaned text")

            if segments is None:
                recording, start, end = uid, 0.0, -1.0
            else:
                recording, start, end = segments[uid]
            # waveform: (samples,)
            waveform, rate, offset = audio.read(recording, start, end)
            if rate != aligner.sample_rate:
                import librosa

                waveform = librosa.resample(
                    waveform, orig_sr=rate, target_sr=aligner.sample_rate
                )
            duration = len(waveform) / aligner.sample_rate
            yield align_phones(
                aligner,
                uid,
                waveform,
                words,
                pronunciations[uid]["phones"],
                duration,
                recording,
                offset,
            )

    write_alignments(args.output, records())
    configuration = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    configuration["model_sample_rate"] = aligner.sample_rate
    write_json(args.output.with_suffix(".config.json"), configuration)


if __name__ == "__main__":
    main()
