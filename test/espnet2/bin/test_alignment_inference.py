import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from espnet2.bin import alignment_inference as inference
from espnet2.bin.align import ForcedAligner
from espnet2.utils.alignment import read_alignments


def phone_aligner():
    vocabulary = ["<blank>", "AA1", "B", "<unk>", "<sos/eos>"]
    model = SimpleNamespace(
        asr_model=SimpleNamespace(
            ctc=SimpleNamespace(ctc_type="builtin"),
            blank_id=0,
            token_list=vocabulary,
        ),
        asr_train_args=SimpleNamespace(token_type="phn", cleaner=None),
        sample_rate=16000,
        tokenizer=SimpleNamespace(text2tokens=str.split),
        converter=SimpleNamespace(
            tokens2ids=lambda ps: [vocabulary.index(p) for p in ps]
        ),
        read_audio=lambda speech: speech,
    )
    aligner = ForcedAligner(model)
    path = [1, 0, 1, 2]
    probabilities = np.full((4, len(vocabulary)), 1e-6, dtype=np.float32)
    for i, label in enumerate(path):
        probabilities[i, label] = 1
    aligner.log_probs = lambda speech: np.log(probabilities)
    return aligner


def test_actual_ctc_phone_grouping_and_cli_output(tmp_path, monkeypatch):
    wav = tmp_path / "audio.wav"
    soundfile.write(wav, np.zeros(8000), 8000)
    scp = tmp_path / "wav.scp"
    text = tmp_path / "text"
    pronunciations = tmp_path / "pronunciations.jsonl"
    scp.write_text(f"u {wav}\n")
    text.write_text("u FIRST SECOND\n")
    pronunciations.write_text(
        json.dumps(
            {
                "utt_id": "u",
                "words": ["FIRST", "SECOND"],
                "phones": [["AA1", "AA1"], ["B"]],
            }
        )
        + "\n"
    )
    output = tmp_path / "hyp.jsonl"
    monkeypatch.setattr(inference, "ForcedAligner", lambda **kwargs: phone_aligner())
    inference.main(
        [
            "--wav-scp",
            str(scp),
            "--text",
            str(text),
            "--pronunciations",
            str(pronunciations),
            "--asr-train-config",
            "config.yaml",
            "--asr-model-file",
            "model.pth",
            "--output",
            str(output),
        ]
    )
    record = read_alignments(output)["u"]
    assert [p.label for p in record.phones] == ["AA1", "AA1", "B"]
    assert [p.word_index for p in record.phones] == [0, 0, 1]
    assert [w.label for w in record.words] == ["FIRST", "SECOND"]
    assert record.words[0].start == record.phones[0].start
    assert record.words[0].end == record.phones[1].end
    assert record.words[0].end == pytest.approx(0.75)
    assert record.duration_s == 1
    assert (
        json.loads(output.with_suffix(".config.json").read_text())["model_sample_rate"]
        == 16000
    )


def test_label_prior_ctc_aligns_with_priors():
    import torch

    log_prior = torch.log(torch.tensor([0.8, 0.1, 0.1]))  # (V,)
    ctc = SimpleNamespace(
        log_softmax=lambda enc: torch.log_softmax(enc, dim=-1),  # (1, T', V)
        apply_prior=lambda log_probs: log_probs - 0.3 * log_prior,
    )
    enc = torch.zeros(1, 2, 3)  # (1, T', V)
    network = SimpleNamespace(
        ctc=ctc, encode=lambda speech, speech_lengths: (enc, speech_lengths)
    )
    model = SimpleNamespace(asr_model=network, dtype="float32", device="cpu")
    log_probs = ForcedAligner(model).log_probs(np.zeros(320))  # (T', V)
    expected = torch.log_softmax(enc, dim=-1)[0] - 0.3 * log_prior
    np.testing.assert_allclose(log_probs, expected.numpy(), rtol=1e-6)


@pytest.mark.parametrize("use_key_file", [True, False])
def test_pronunciations_may_cover_only_selected_keys(
    tmp_path, monkeypatch, use_key_file
):
    wav = tmp_path / "audio.wav"
    soundfile.write(wav, np.zeros(8000), 8000)
    scp, text, keys = tmp_path / "wav.scp", tmp_path / "text", tmp_path / "keys"
    pronunciations = tmp_path / "pronunciations.jsonl"
    scp.write_text(f"u {wav}\nno_textgrid {wav}\n")
    text.write_text("u FIRST SECOND\nno_textgrid FIRST\n")
    keys.write_text("u\n")
    pronunciations.write_text(
        json.dumps(
            {"utt_id": "u", "words": ["FIRST", "SECOND"], "phones": [["AA1"], ["B"]]}
        )
        + "\n"
    )
    output = tmp_path / "hyp.jsonl"
    monkeypatch.setattr(inference, "ForcedAligner", lambda **kwargs: phone_aligner())
    arguments = [
        "--wav-scp",
        str(scp),
        "--text",
        str(text),
        "--pronunciations",
        str(pronunciations),
        "--asr-train-config",
        "config.yaml",
        "--asr-model-file",
        "model.pth",
        "--output",
        str(output),
    ]
    if not use_key_file:
        with pytest.raises(ValueError, match="cover the selected keys"):
            inference.main(arguments)
        return
    inference.main([*arguments, "--key-file", str(keys)])
    assert list(read_alignments(output)) == ["u"]


@pytest.mark.parametrize("audio_type", ["sound", "pipe", "kaldi_ark"])
def test_audio_formats_and_segment_time_origin(tmp_path, audio_type):
    wave = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    path = tmp_path / "source.wav"
    soundfile.write(path, wave, 16000, subtype="FLOAT")
    scp = tmp_path / "wav.scp"
    kind = "sound"
    if audio_type == "pipe":
        scp.write_text(f"rec cat '{path}' |\n")
    elif audio_type == "kaldi_ark":
        from omniio import kaldi

        kaldi.save_ark(
            str(tmp_path / "audio.ark"),
            {"rec": (wave, 16000)},
            scp=str(scp),
            write_function="soundfile",
            write_kwargs={"format": "WAV", "subtype": "FLOAT"},
        )
        kind = "kaldi_ark"
    else:
        scp.write_text(f"rec {path}\n")
    audio, rate, offset = inference.AudioReader(scp, kind).read("rec", 0.25, 0.5)
    assert rate == 16000
    assert offset == 0.25
    np.testing.assert_allclose(audio, wave[4000:8000])
    with pytest.raises(ValueError, match="outside recording"):
        inference.AudioReader(scp, kind).read("rec", 0, 2)


@pytest.mark.parametrize("phones", [[[]], [["<unk>"]], [["NOT_IN_VOCAB"]]])
def test_invalid_phone_targets_fail_before_alignment(phones):
    with pytest.raises(ValueError, match="empty, unknown or reserved"):
        inference.align_phones(
            phone_aligner(), "u", np.zeros(16000), ["word"], phones, 1, "u", 0
        )


def test_tokenizer_must_preserve_phone_units():
    aligner = phone_aligner()
    aligner.model.tokenizer.text2tokens = list
    with pytest.raises(ValueError, match="tokenizer changes"):
        inference.align_phones(
            aligner, "u", np.zeros(16000), ["word"], [["AA1"]], 1, "u", 0
        )


def test_segment_ids_and_bounds(tmp_path):
    path = tmp_path / "segments"
    path.write_text("u rec 0.25 -1\n")
    assert inference.read_segments(path) == {"u": ("rec", 0.25, -1)}
    path.write_text("u rec nan 1\n")
    with pytest.raises(ValueError, match="invalid segment"):
        inference.read_segments(path)


def test_checkpoint_guards(tmp_path, monkeypatch):
    aligner = phone_aligner()
    aligner.model.asr_train_args.token_type = "char"
    monkeypatch.setattr(inference, "ForcedAligner", lambda **kwargs: aligner)
    with pytest.raises(ValueError, match="expected token_type=phn or word"):
        inference.main(
            [
                "--wav-scp",
                str(tmp_path / "wav.scp"),
                "--text",
                str(tmp_path / "text"),
                "--pronunciations",
                str(tmp_path / "pronunciations.jsonl"),
                "--asr-train-config",
                "config.yaml",
                "--asr-model-file",
                "model.pth",
                "--output",
                str(tmp_path / "hyp.jsonl"),
            ]
        )


@pytest.mark.parametrize("sample_rate", [16000, 8000])
def test_model_loading_to_saved_scoring_smoke(tmp_path, sample_rate):
    import torch

    from espnet2.bin.alignment_scoring import main as score
    from espnet2.tasks.asr import ASRTask

    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blank>\nAA1\nB\n<unk>\n<sos/eos>\n")
    ASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "model"),
            "--token_list",
            str(tokens),
            "--token_type",
            "phn",
            "--frontend_conf",
            json.dumps({"fs": sample_rate if sample_rate == 16000 else "8k"}),
            "--encoder",
            "rnn",
            "--encoder_conf",
            "{num_layers: 1, hidden_size: 8, output_size: 8}",
            "--model_conf",
            "{ctc_weight: 1.0}",
        ]
    )
    config = tmp_path / "model/config.yaml"
    model, _ = ASRTask.build_model_from_file(config, None)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint)
    wav = tmp_path / "audio.wav"
    soundfile.write(wav, np.zeros(sample_rate // 2), sample_rate)
    data_dir = tmp_path / "data/eval"
    data_dir.mkdir(parents=True)
    (data_dir / "wav.scp").write_text(f"u {wav}\n")
    (data_dir / "text").write_text("u A B\n")
    reference_dir = tmp_path / "data/reference"
    reference_dir.mkdir()
    (reference_dir / "eval.keys").write_text("u\n")
    (reference_dir / "eval.pronunciations.jsonl").write_text(
        json.dumps({"utt_id": "u", "words": ["A", "B"], "phones": [["AA1"], ["B"]]})
        + "\n"
    )
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "utils").symlink_to(root / "egs2/TEMPLATE/asr1/utils")
    (tmp_path / "path.sh").write_text("")
    subprocess.run(
        [
            "bash",
            str(root / "egs2/TEMPLATE/align1/align.sh"),
            "--stage",
            "1",
            "--stop-stage",
            "1",
            "--nj",
            "2",
            "--test-sets",
            "eval",
            "--data-root",
            "data",
            "--asr-train-config",
            str(config),
            "--asr-model-file",
            str(checkpoint),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(root)),
        check=True,
        capture_output=True,
        text=True,
    )
    output_dir = tmp_path / "exp/alignment/eval"
    output = output_dir / "log/hyp.1.jsonl"
    assert (output_dir / "hypotheses.list").read_text().splitlines() == [
        "exp/alignment/eval/log/hyp.1.jsonl"
    ]
    assert len(read_alignments(output)["u"].phones) == 2
    assert (
        json.loads(output.with_suffix(".config.json").read_text())["model_sample_rate"]
        == sample_rate
    )
    score(
        [
            "--reference",
            str(output),
            "--hypothesis",
            str(output),
            "--output-dir",
            str(tmp_path / "score"),
        ]
    )
    summary = json.loads((tmp_path / "score/summary.json").read_text())
    assert summary["phones"]["PBE_ms"] == 0
    assert summary["words"]["WBE_ms"] == 0
