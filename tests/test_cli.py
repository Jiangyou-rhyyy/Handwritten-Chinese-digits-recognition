import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_data import _write_tiny_archive


class CommandLineTests(unittest.TestCase):
    def test_public_commands_expose_help_without_starting_work(self):
        project_root = Path(__file__).parents[1]

        for script_name in ("app.py", "evaluate.py", "train.py"):
            with self.subTest(script=script_name):
                completed = subprocess.run(
                    [sys.executable, str(project_root / script_name), "--help"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout.lower())

    def test_evaluation_handles_an_archive_with_missing_classes(self):
        project_root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            archive = Path(directory) / "tiny.zip"
            _write_tiny_archive(archive)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "evaluate.py"),
                    "--data",
                    str(archive),
                    "--weights",
                    str(project_root / "pthmnist_cnn.pth"),
                    "--device",
                    "cpu",
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("0/0", completed.stdout)


if __name__ == "__main__":
    unittest.main()
