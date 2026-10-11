import subprocess
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
