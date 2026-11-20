import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from espnet2.bin.alignment_scoring import main
from espnet2.utils.alignment import (
    Alignment,
    Interval,
    read_alignments,
    write_alignments,
)

ROOT = Path(__file__).resolve().parents[3]


def test_saved_scoring_missing_and_unavailable_tiers(tmp_path):
    reference = tmp_path / "ref.jsonl"
    hypothesis = tmp_path / "hyp.jsonl"
    write_alignments(
        reference,
        [
            Alignment("u1", 1, words=[Interval("A", 0, 0.2)]),
            Alignment("u2", 1, words=[Interval("B", 0, 0.2)]),
        ],
    )
    write_alignments(hypothesis, [Alignment("u1", 1, words=[Interval("a", 0.1, 0.4)])])
    main(
        [
            "--reference",
            str(reference),
            "--hypothesis",
            str(hypothesis),
            "--output-dir",
            str(tmp_path / "score"),
        ]
    )
    summary = json.loads((tmp_path / "score/summary.json").read_text())
    assert summary["words"]["WBE_ms"] == pytest.approx(150)
    assert summary["words"]["WDUR_hyp_ms"] == pytest.approx(300)
    assert summary["missing_hypothesis_ids"] == ["u2"]
    assert summary["phones"]["PBE_ms"] is None


@pytest.mark.parametrize(
    "recording_id,crop_offset_s", [("different-recording", 3), ("recording", 4)]
)
def test_segment_provenance_is_optional_but_checked_when_present(
    tmp_path, recording_id, crop_offset_s
):
    reference, hypothesis = tmp_path / "ref.jsonl", tmp_path / "hyp.jsonl"
    ref = Alignment("u", 1, phones=[Interval("A", 0, 0.2)])
    hyp = Alignment(
        "u",
        1,
        phones=[Interval("A", 0, 0.2)],
        recording_id="recording",
        crop_offset_s=3,
    )
    write_alignments(reference, [ref])
    write_alignments(hypothesis, [hyp])
    arguments = [
        "--reference",
        str(reference),
        "--hypothesis",
        str(hypothesis),
        "--output-dir",
        str(tmp_path / "score"),
    ]
    main(arguments)
    assert (
        json.loads((tmp_path / "score/summary.json").read_text())["phones"]["PBE_ms"]
        == 0
    )
    ref.recording_id, ref.crop_offset_s = recording_id, crop_offset_s
    write_alignments(reference, [ref])
    with pytest.raises(ValueError, match="coordinates differ"):
        main(arguments)


def test_rescoring_reads_every_saved_worker_without_torch(tmp_path):
    (tmp_path / "utils").symlink_to(ROOT / "egs2/TEMPLATE/asr1/utils")
    (tmp_path / "path.sh").write_text("")
    out = tmp_path / "exp/alignment/eval"
    out.mkdir(parents=True)
    refs = [
        Alignment("u1", 1, phones=[Interval("A", 0, 0.2)]),
        Alignment("u2", 1, phones=[Interval("B", 0, 0.2)]),
    ]
    write_alignments(tmp_path / "data/reference/eval.jsonl", refs)
    paths = [out / "hyp.1.jsonl", out / "hyp.2.jsonl"]
    for record, path in zip(refs, paths):
        write_alignments(path, [record])
    (out / "hypotheses.list").write_text("".join(f"{p}\n" for p in paths))
    # A broken torch import proves scoring cannot accidentally load the model.
    (tmp_path / "torch.py").write_text("raise RuntimeError('model access forbidden')\n")
    env = dict(os.environ, PYTHONPATH=f"{tmp_path}{os.pathsep}{ROOT}")
    subprocess.run(
        [
            "bash",
            str(ROOT / "egs2/TEMPLATE/align1/align.sh"),
            "--stage",
            "2",
            "--test-sets",
            "eval",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((out / "score/summary.json").read_text())
    assert result["phones"]["PBE_ms"] == 0
    assert result["phones"]["n_matched"] == 2


def test_textgrid_conversion_preserves_intervals_and_reports_missing(tmp_path):
    from praatio import textgrid

    grid = textgrid.Textgrid()
    grid.addTier(
        textgrid.IntervalTier(
            "words", [(0, 0.1234567, ""), (0.1234567, 1, "CAT")], 0, 1
        )
    )
    grid.addTier(
        textgrid.IntervalTier(
            "phones", [(0, 0.1234567, "sil"), (0.1234567, 1, "K")], 0, 1
        )
    )
    path = tmp_path / "u.TextGrid"
    grid.save(str(path), format="long_textgrid", includeBlankSpaces=True)
    text = tmp_path / "text"
    text.write_text("u CAT\nmissing CAT\n")
    output = tmp_path / "reference.jsonl"
    converter = runpy.run_path(
        str(ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils/prepare_alignment_reference.py")
    )["main"]
    converter(
        ["--textgrid-root", str(tmp_path), "--text", str(text), "--output", str(output)]
    )
    record = read_alignments(output)["u"]
    assert record.words[1].start == 0.1234567
    assert record.phones[0].label == "sil"
    assert json.loads(output.with_suffix(".coverage.json").read_text())[
        "missing_textgrid_ids"
    ] == ["missing"]
    with pytest.raises(ValueError, match="missing tier"):
        converter(
            [
                "--textgrid-root",
                str(tmp_path),
                "--text",
                str(text),
                "--phone-tier",
                "nonexistent",
                "--output",
                str(output),
            ]
        )


@pytest.mark.parametrize(
    "transcript,grid_words,expected",
    [
        # MFA splits OOV/possessive words at apostrophes; <unk> matches any word.
        (
            "QUILTER'S CAT",
            [("quilter", ["K", "W"]), ("s", ["Z"]), ("cat", ["K"])],
            [["K", "W", "Z"], ["K"]],
        ),
        ("MILLION'D", [("<unk>", ["spn"]), ("d", ["D"])], [["spn", "D"]]),
        ("BUSH'", [("bush", ["B", "SH"])], [["B", "SH"]]),
        ("THE CAT", [("the", ["DH"]), ("dog", ["D"])], "do not match"),
    ],
)
def test_pronunciations_follow_textgrid_word_grouping(
    tmp_path, transcript, grid_words, expected
):
    from praatio import textgrid

    word_entries, phone_entries, t = [(0, 0.1, "")], [(0, 0.1, "sil")], 0.1
    for word, phones in grid_words:
        start = t
        for phone in phones:
            phone_entries.append((t, t + 0.1, phone))
            t += 0.1
        word_entries.append((start, t, word))
    grid = textgrid.Textgrid()
    grid.addTier(textgrid.IntervalTier("words", word_entries, 0, t))
    grid.addTier(textgrid.IntervalTier("phones", phone_entries, 0, t))
    grid.save(
        str(tmp_path / "u.TextGrid"), format="long_textgrid", includeBlankSpaces=True
    )
    text = tmp_path / "text"
    text.write_text(f"u {transcript}\nmissing CAT\n")
    pronunciations = tmp_path / "pronunciations.jsonl"
    keys = tmp_path / "keys"
    converter = runpy.run_path(
        str(ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils/prepare_alignment_reference.py")
    )["main"]
    arguments = [
        "--textgrid-root",
        str(tmp_path),
        "--text",
        str(text),
        "--output",
        str(tmp_path / "reference.jsonl"),
        "--pronunciation-output",
        str(pronunciations),
        "--key-output",
        str(keys),
    ]
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            converter(arguments)
        assert not pronunciations.exists()
        return
    converter(arguments)
    assert json.loads(pronunciations.read_text()) == {
        "utt_id": "u",
        "words": transcript.split(),
        "phones": expected,
    }
    assert keys.read_text() == "u\n"


def test_recipe_stage_zero_converts_subsets_without_model_or_audio(tmp_path):
    from praatio import textgrid

    (tmp_path / "utils").symlink_to(ROOT / "egs2/TEMPLATE/asr1/utils")
    (tmp_path / "pyscripts").symlink_to(ROOT / "egs2/TEMPLATE/asr1/pyscripts")
    (tmp_path / "path.sh").write_text("")
    (tmp_path / "align.sh").symlink_to(ROOT / "egs2/TEMPLATE/align1/align.sh")
    (tmp_path / "torch.py").write_text("raise RuntimeError('model access forbidden')\n")
    for subset in ("test_clean", "dev_clean"):
        grid_dir = tmp_path / "grids" / subset.replace("_", "-")
        data_dir = tmp_path / "data" / subset
        grid_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        (data_dir / "text").write_text(f"{subset} CAT\n")
        grid = textgrid.Textgrid()
        for tier, label in (("words", "CAT"), ("phones", "K")):
            grid.addTier(textgrid.IntervalTier(tier, [(0, 1, label)], 0, 1))
        grid.save(
            str(grid_dir / f"{subset}.TextGrid"),
            format="long_textgrid",
            includeBlankSpaces=True,
        )
    subprocess.run(
        [
            "bash",
            str(ROOT / "egs2/librispeech/align1/run.sh"),
            "--stage",
            "0",
            "--stop-stage",
            "0",
            "--textgrid-root",
            str(tmp_path / "grids"),
            "--data-root",
            str(tmp_path / "data"),
            "--reference-root",
            str(tmp_path / "references"),
            "--test-sets",
            "test_clean dev_clean",
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=f"{tmp_path}{os.pathsep}{ROOT}"),
        check=True,
        capture_output=True,
        text=True,
    )
    for subset in ("test_clean", "dev_clean"):
        references = tmp_path / "references"
        record = read_alignments(references / f"{subset}.jsonl")[subset]
        assert record.words[0].label == "CAT" and record.phones[0].label == "K"
        assert json.loads(
            (references / f"{subset}.pronunciations.jsonl").read_text()
        ) == {"utt_id": subset, "words": ["CAT"], "phones": [["K"]]}
        assert (references / f"{subset}.keys").read_text() == f"{subset}\n"


@pytest.mark.parametrize(
    "options,message",
    [
        (["--stage", "0", "--stop-stage", "0"], "--textgrid-root is required"),
        (
            ["--stage", "0", "--stop-stage", "0", "--textgrid-root", "missing"],
            "TextGrid directory not found",
        ),
        (["--stage", "2", "--stop-stage", "0"], "Invalid stage range"),
    ],
)
def test_recipe_stage_zero_invalid_options(tmp_path, options, message):
    (tmp_path / "utils").symlink_to(ROOT / "egs2/TEMPLATE/asr1/utils")
    (tmp_path / "path.sh").write_text("")
    (tmp_path / "align.sh").symlink_to(ROOT / "egs2/TEMPLATE/align1/align.sh")
    result = subprocess.run(
        ["bash", str(ROOT / "egs2/librispeech/align1/run.sh"), *options],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    "entrypoint",
    [
        "TEMPLATE/align1/align.sh",
        "librispeech/align1/run.sh",
    ],
)
@pytest.mark.parametrize("stage,stop_stage", [(0, 2), (1, 1), (2, 2)])
def test_recipe_completes_each_stage_for_all_subsets(
    tmp_path, entrypoint, stage, stop_stage
):
    (tmp_path / "utils").symlink_to(ROOT / "egs2/TEMPLATE/asr1/utils")
    (tmp_path / "path.sh").write_text("")
    (tmp_path / "align.sh").symlink_to(ROOT / "egs2/TEMPLATE/align1/align.sh")
    for subset in ("test-clean", "dev-clean"):
        (tmp_path / "grids" / subset).mkdir(parents=True)
    for subset in ("test_clean", "dev_clean"):
        (tmp_path / "data/reference").mkdir(parents=True, exist_ok=True)
        (tmp_path / f"data/reference/{subset}.keys").write_text("u\n")
        (tmp_path / f"exp/alignment/{subset}").mkdir(parents=True)
        (tmp_path / f"exp/alignment/{subset}/hypotheses.list").write_text("hyp\n")
    # Fake python: record "<stage> <subset>" from the called entry point.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "stage=0\n"
        'while [ "$#" -gt 0 ]; do\n'
        '    case "$1" in\n'
        "        espnet2.bin.alignment_inference) stage=1 ;;\n"
        "        espnet2.bin.alignment_scoring) stage=2 ;;\n"
        '        --text) subset=$(basename "$(dirname "$2")") ;;\n'
        '        --reference) subset=$(basename "$2" .jsonl) ;;\n'
        "    esac\n"
        "    shift\n"
        "done\n"
        'printf "%s %s\\n" "$stage" "$subset" >> "$TRACE"\n'
    )
    python.chmod(0o755)
    trace = tmp_path / "trace"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "egs2" / entrypoint),
            "--stage",
            str(stage),
            "--stop-stage",
            str(stop_stage),
            "--test-sets",
            "test_clean dev_clean",
            "--textgrid-root",
            "grids",
        ],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            TRACE=str(trace),
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert trace.read_text().splitlines() == [
        f"{s} {subset}"
        for s in range(stage, stop_stage + 1)
        for subset in ("test_clean", "dev_clean")
    ]


def test_alignment_setup_generated_recipe_and_overwrite_protection(tmp_path):
    from praatio import textgrid

    template = tmp_path / "egs2/TEMPLATE/align1"
    shutil.copytree(ROOT / "egs2/TEMPLATE/align1", template, symlinks=True)
    (template.parent / "asr1").symlink_to(ROOT / "egs2/TEMPLATE/asr1")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/extra_path.sh").write_text("")
    recipe = tmp_path / "egs2/new_corpus/align1"
    setup = ["bash", str(template / "setup.sh"), str(recipe)]
    subprocess.run(setup, check=True, capture_output=True, text=True)
    for name in ("align.sh", "path.sh", "scripts", "pyscripts", "utils"):
        assert (recipe / name).is_symlink()
        assert (recipe / name).exists()
    (recipe / "local/path.sh").write_text("export ALIGNMENT_CUSTOM_PATH=1\n")
    data = recipe / "data/eval"
    data.mkdir(parents=True)
    (data / "text").write_text("u CAT\n")
    grids = recipe / "grids/nested"
    grids.mkdir(parents=True)
    grid = textgrid.Textgrid()
    for tier, label in (("words", "CAT"), ("phones", "K")):
        grid.addTier(textgrid.IntervalTier(tier, [(0, 1, label)], 0, 1))
    grid.save(
        str(grids / "u.TextGrid"), format="long_textgrid", includeBlankSpaces=True
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run(
        [
            "bash",
            "run.sh",
            "--stage",
            "0",
            "--stop-stage",
            "0",
            "--test-sets",
            "eval",
            "--data-root",
            "data",
            "--textgrid-root",
            "grids",
        ],
        cwd=recipe,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    reference = recipe / "data/reference/eval.jsonl"
    assert read_alignments(reference)["u"].phones[0].label == "K"
    out = recipe / "exp/alignment/eval"
    out.mkdir(parents=True)
    write_alignments(out / "hyp.jsonl", list(read_alignments(reference).values()))
    (out / "hypotheses.list").write_text("exp/alignment/eval/hyp.jsonl\n")
    subprocess.run(
        ["bash", "run.sh", "--stage", "2", "--test-sets", "eval"],
        cwd=recipe,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((out / "score/summary.json").read_text())
    assert summary["phones"]["PBE_ms"] == 0
    assert summary["words"]["WBE_ms"] == 0
    result = subprocess.run(setup, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Refusing to overwrite" in result.stderr
    assert (recipe / "local/path.sh").read_text() == "export ALIGNMENT_CUSTOM_PATH=1\n"
    invalid = subprocess.run(
        ["bash", str(template / "setup.sh"), str(tmp_path / "outside/align1")],
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 2
    assert "same egs2 tree" in invalid.stderr
