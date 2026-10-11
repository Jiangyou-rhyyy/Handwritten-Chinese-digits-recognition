import tempfile
import unittest
from pathlib import Path

from train import PROJECT_ROOT, build_parser, validate_output_path


class TrainingCommandTests(unittest.TestCase):
    def test_default_output_does_not_replace_the_packaged_checkpoint(self):
        args = build_parser().parse_args([])

        self.assertNotEqual(args.output, PROJECT_ROOT / "pthmnist_cnn.pth")
        self.assertEqual(args.output, PROJECT_ROOT / "checkpoints" / "trained.pth")

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory) / "checkpoint.pth"
            output.write_bytes(b"existing checkpoint")

            with self.assertRaises(FileExistsError):
                validate_output_path(output, overwrite=False)

            validate_output_path(output, overwrite=True)


if __name__ == "__main__":
    unittest.main()
