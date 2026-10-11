import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from chinese_digits.data import ChineseMNISTDataset, writer_disjoint_split


def _write_tiny_archive(path):
    rows = [
        {"suite_id": 1, "sample_id": 1, "code": 1, "value": 0, "character": "零"},
        {"suite_id": 2, "sample_id": 1, "code": 2, "value": 1, "character": "一"},
    ]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Chinese MNIST/chinese_mnist.csv", csv_buffer.getvalue())
        for row in rows:
            image = Image.new("L", (64, 64), color=row["code"] * 10)
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="JPEG")
            filename = (
                f"Chinese MNIST/data/data/input_{row['suite_id']}_"
                f"{row['sample_id']}_{row['code']}.jpg"
            )
            archive.writestr(filename, image_buffer.getvalue())


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.archive_path = Path(self.temp_directory.name) / "tiny.zip"
        _write_tiny_archive(self.archive_path)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_dataset_reads_images_and_zero_based_labels_from_zip(self):
        dataset = ChineseMNISTDataset(self.archive_path)

        image, label = dataset[1]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(image.mode, "L")
        self.assertEqual(image.size, (64, 64))
        self.assertEqual(label, 1)
        self.assertEqual(dataset.records[1].suite_id, 2)

    def test_writer_split_prevents_the_same_writer_appearing_in_both_sets(self):
        dataset = ChineseMNISTDataset(self.archive_path)

        train_indices, test_indices = writer_disjoint_split(
            dataset.records, train_ratio=0.5, seed=7
        )
        train_writers = {dataset.records[index].suite_id for index in train_indices}
        test_writers = {dataset.records[index].suite_id for index in test_indices}

        self.assertEqual(len(train_indices), 1)
        self.assertEqual(len(test_indices), 1)
        self.assertTrue(train_writers.isdisjoint(test_writers))
        self.assertEqual(sorted(train_indices + test_indices), [0, 1])


if __name__ == "__main__":
    unittest.main()
