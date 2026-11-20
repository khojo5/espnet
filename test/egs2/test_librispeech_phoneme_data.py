import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from praatio import textgrid


class TestLibriSpeechPhonemeData(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.recipe = Path(self.temporary_directory.name)
        root = Path(__file__).resolve().parents[2]
        original = root / "egs2/librispeech/asr1"
        local = self.recipe / "local"
        local.mkdir()
        for name in ("textgrid_to_phoneme.py", "prepare_phoneme_data.sh"):
            shutil.copyfile(original / "local" / name, local / name)
        (self.recipe / "utils").symlink_to(original / "utils", target_is_directory=True)
        (self.recipe / "data").mkdir()
        binaries = self.recipe / "bin"
        binaries.mkdir()
        (binaries / "python").symlink_to(sys.executable)
        (self.recipe / "path.sh").write_text(
            'export PATH="$PWD/bin:$PATH"\n', encoding="utf-8"
        )
        modules = self.recipe / "modules"
        modules.mkdir()
        for name in ("g2p_en", "pyopenjtalk"):
            (modules / f"{name}.py").write_text(
                'raise AssertionError("TextGrid extraction imported G2P")\n',
                encoding="utf-8",
            )
        self.environment = os.environ.copy()
        self.environment["PYTHONPATH"] = os.pathsep.join((str(modules), str(root)))
        self.environment["PYTHONDONTWRITEBYTECODE"] = "1"
        self.textgrid_root = self.recipe / "textgrids"
        self.textgrid_root.mkdir()
        for uid in ("spk1-001", "spk2-001"):
            self.write_grid(
                uid,
                ["", "sil", "HH", "AH0", "sp", "L", "OW1", "W", "ER1", "L", "D", "spn"],
            )
        (self.textgrid_root / "unselected.TextGrid").write_text(
            "Not a TextGrid: unselected files must not be parsed.", encoding="utf-8"
        )

    def write_grid(self, uid, labels, tier_name="phones", point=False):
        directory = self.textgrid_root / "speaker" / "chapter"
        directory.mkdir(parents=True, exist_ok=True)
        grid = textgrid.Textgrid()
        if point:
            tier = textgrid.PointTier(tier_name, [(0, "HH")], 0, 1)
        else:
            tier = textgrid.IntervalTier(
                tier_name,
                [(i, i + 1, label) for i, label in enumerate(labels)],
                0,
                len(labels),
            )
        grid.addTier(tier)
        path = directory / f"{uid}.TextGrid"
        grid.save(str(path), format="long_textgrid", includeBlankSpaces=True)
        return path

    def prepare_set(self, name):
        source = self.recipe / "data" / name
        source.mkdir()
        files = {
            "text": "spk1-001 HELLO WORLD\nspk2-001 HELLO WORLD\n",
            "utt2spk": "spk1-001 spk1\nspk2-001 spk2\n",
            "spk2utt": "spk1 spk1-001\nspk2 spk2-001\n",
            "wav.scp": "spk1-001 /audio/one.flac\nspk2-001 /audio/two.flac\n",
            "utt2dur": "spk1-001 1.0\nspk2-001 2.0\n",
        }
        for name, contents in files.items():
            (source / name).write_text(contents, encoding="utf-8")
        return source

    def run_preparation(self, *sets, phone_tier="phones", missing_textgrid="error"):
        return subprocess.run(
            [
                "bash",
                "local/prepare_phoneme_data.sh",
                "--textgrid-root",
                str(self.textgrid_root),
                "--phone-tier",
                phone_tier,
                "--missing-textgrid",
                missing_textgrid,
                *sets,
            ],
            cwd=self.recipe,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def assert_no_temporary_data(self):
        self.assertEqual(list((self.recipe / "data").glob(".phoneme.*")), [])

    def test_default_sets_preserve_source_and_audio_references(self):
        sets = (
            "train_960",
            "dev",
            "test_clean",
            "test_other",
            "dev_clean",
            "dev_other",
        )
        snapshots = {}
        for name in sets:
            source = self.prepare_set(name)
            snapshots[name] = {p.name: p.read_bytes() for p in source.iterdir()}
        result = self.run_preparation()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in sets:
            source = self.recipe / "data" / name
            output = self.recipe / "data" / f"{name}_phn"
            self.assertEqual(
                (output / "text").read_text(encoding="utf-8"),
                "spk1-001 HH AH0 L OW1 W ER1 L D spn\n"
                "spk2-001 HH AH0 L OW1 W ER1 L D spn\n",
            )
            for filename, contents in snapshots[name].items():
                self.assertEqual((source / filename).read_bytes(), contents)
                if filename != "text":
                    self.assertEqual((output / filename).read_bytes(), contents)
        self.assert_no_temporary_data()
        self.assertEqual(result.stderr.count("Searching for TextGrids"), 1)

    def test_preflight_refuses_missing_input_and_existing_outputs(self):
        source = self.prepare_set("dev")
        result = self.run_preparation("dev", "missing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing or empty input", result.stderr)
        self.assertFalse((self.recipe / "data/dev_phn").exists())
        output = self.recipe / "data/dev_phn"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("existing", encoding="utf-8")
        result = self.run_preparation("dev")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output already exists", result.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing")
        output.rename(self.recipe / "existing")
        output.symlink_to(self.recipe / "missing_target", target_is_directory=True)
        result = self.run_preparation("dev")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(output.is_symlink())
        self.assertIn("HELLO WORLD", (source / "text").read_text(encoding="utf-8"))
        self.assert_no_temporary_data()

    def test_missing_and_duplicate_textgrids_are_not_published(self):
        source = self.prepare_set("dev")
        contents = (source / "text").read_bytes()
        path = self.textgrid_root / "speaker/chapter/spk2-001.TextGrid"
        saved = path.read_bytes()
        path.unlink()
        result = self.run_preparation("dev")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing TextGrids for 1 utterances: spk2-001", result.stderr)
        self.assertFalse((self.recipe / "data/dev_phn").exists())
        self.assert_no_temporary_data()
        path.write_bytes(saved)
        duplicate = self.textgrid_root / "spk2-001.textgrid"
        duplicate.write_bytes(saved)
        result = self.run_preparation("dev")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate TextGrid ID spk2-001", result.stderr)
        self.assertFalse((self.recipe / "data/dev_phn").exists())
        self.assertEqual((source / "text").read_bytes(), contents)
        self.assert_no_temporary_data()

    def test_invalid_tiers_and_labels_are_not_published(self):
        self.prepare_set("dev")
        for labels, name, point, message in (
            (["HH"], "words", False, "missing tier 'phones'"),
            (["", "sil", "sp"], "phones", False, "empty phoneme sequence"),
            (["HH AH0"], "phones", False, "invalid phone label"),
            (["HH"], "phones", True, "not an interval tier"),
        ):
            with self.subTest(message=message):
                self.write_grid("spk2-001", labels, tier_name=name, point=point)
                result = self.run_preparation("dev", missing_textgrid="g2p")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("spk2-001", result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse((self.recipe / "data/dev_phn").exists())
                self.assert_no_temporary_data()

    def mock_g2p(self, phones):
        (self.recipe / "modules/g2p_en.py").write_text(
            f"""class G2p:
    def __call__(self, text):
        if text != "HELLO WORLD":
            raise ValueError("unexpected G2P transcript: " + text)
        return {phones!r}
""",
            encoding="utf-8",
        )

    def test_missing_grid_uses_g2p_and_records_only_missing_ids(self):
        source = self.prepare_set("dev")
        original = (source / "text").read_bytes()
        (self.textgrid_root / "speaker/chapter/spk2-001.TextGrid").unlink()
        self.mock_g2p(["HH", "AH0", "L", "OW1", " ", "W", "ER1", "L", "D"])
        result = self.run_preparation("dev", missing_textgrid="g2p")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.recipe / "data/dev_phn"
        self.assertEqual(
            (output / "text").read_text(encoding="utf-8"),
            "spk1-001 HH AH0 L OW1 W ER1 L D spn\n" "spk2-001 HH AH0 L OW1 W ER1 L D\n",
        )
        self.assertEqual(
            (output / "g2p_fallback_utt_ids").read_text(encoding="utf-8"),
            "spk2-001\n",
        )
        self.assertIn("G2P fallback for missing TextGrid: spk2-001", result.stderr)
        self.assertEqual((source / "text").read_bytes(), original)
        self.assert_no_temporary_data()

    def test_g2p_option_does_not_import_backend_when_grids_present(self):
        self.prepare_set("dev")
        result = self.run_preparation("dev", missing_textgrid="g2p")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            (self.recipe / "data/dev_phn/g2p_fallback_utt_ids").read_bytes(), b""
        )
        self.assertNotIn("G2P fallback for missing TextGrid", result.stderr)

    def test_invalid_g2p_output_is_not_published(self):
        source = self.prepare_set("dev")
        original = (source / "text").read_bytes()
        (self.textgrid_root / "speaker/chapter/spk2-001.TextGrid").unlink()
        for phones in ([], [" "], ["HH AH0"]):
            with self.subTest(phones=phones):
                self.mock_g2p(phones)
                result = self.run_preparation("dev", missing_textgrid="g2p")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "spk2-001: empty or invalid G2P phoneme sequence", result.stderr
                )
                self.assertFalse((self.recipe / "data/dev_phn").exists())
                self.assertEqual((source / "text").read_bytes(), original)
                self.assert_no_temporary_data()

    def test_custom_tier_preserves_repeated_phones_and_spn(self):
        self.prepare_set("dev")
        for uid in ("spk1-001", "spk2-001"):
            self.write_grid(
                uid, ["SIL", "ah0", "ah0", "spn", "Sp"], tier_name="segments"
            )
        result = self.run_preparation("dev", phone_tier="segments")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            (self.recipe / "data/dev_phn/text").read_text(encoding="utf-8"),
            "spk1-001 ah0 ah0 spn\nspk2-001 ah0 ah0 spn\n",
        )

    def test_concurrent_destination_is_not_overwritten(self):
        self.prepare_set("dev")
        output = self.recipe / "data/dev_phn"
        with (self.recipe / "path.sh").open("a", encoding="utf-8") as fout:
            fout.write("""python() {
    command "$PWD/bin/python" "$@" || return "$?"
    mkdir data/dev_phn
    printf existing > data/dev_phn/sentinel
}
""")
        result = self.run_preparation("dev")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output already exists", result.stderr)
        self.assertEqual((output / "sentinel").read_text(encoding="utf-8"), "existing")
        self.assertFalse((output / "text").exists())
        self.assert_no_temporary_data()

    def test_invalid_and_duplicate_set_names(self):
        self.prepare_set("dev")
        for arguments in (("../dev",), ("dev/test",), ("--unknown",), ("dev", "dev")):
            with self.subTest(arguments=arguments):
                result = self.run_preparation(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.recipe / "data/dev_phn").exists())
                self.assert_no_temporary_data()

    def test_direct_extraction_preserves_text_order_and_refuses_overwrite(self):
        source = self.prepare_set("dev")
        (source / "text").write_text(
            "spk2-001 WORLD\nspk1-001 HELLO\n", encoding="utf-8"
        )
        output = self.recipe / "phones"
        command = [
            sys.executable,
            "local/textgrid_to_phoneme.py",
            "--textgrid-root",
            str(self.textgrid_root),
            "--input",
            str(source / "text"),
            "--output",
            str(output),
        ]
        result = subprocess.run(
            command,
            cwd=self.recipe,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        contents = output.read_bytes()
        self.assertEqual(
            [line.split()[0] for line in contents.decode().splitlines()],
            ["spk2-001", "spk1-001"],
        )
        result = subprocess.run(
            command,
            cwd=self.recipe,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output already exists", result.stderr)
        self.assertEqual(output.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
