"""Chinese MNIST data loading and writer-disjoint splitting."""

import csv
import io
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChineseMNISTRecord:
    suite_id: int
    sample_id: int
    code: int
    value: int
    character: str
    archive_member: str

    @property
    def label(self) -> int:
        return self.code - 1


class ChineseMNISTDataset(Dataset):
    """Read the packaged Chinese MNIST ZIP archive without extracting it."""

    def __init__(
        self,
        archive_path: Union[str, Path],
        transform: Optional[Callable] = None,
    ) -> None:
        self.archive_path = Path(archive_path)
        self.transform = transform
        self._archive_buffer = io.BytesIO(self.archive_path.read_bytes())
        self._archive = zipfile.ZipFile(self._archive_buffer)
        self.records = self._read_records()

    def _read_records(self) -> List[ChineseMNISTRecord]:
        members = self._archive.namelist()
        csv_member = next(
            member
            for member in members
            if member.endswith("chinese_mnist.csv") and "__MACOSX" not in member
        )
        images_by_name = {
            Path(member).name: member
            for member in members
            if member.lower().endswith(".jpg") and "__MACOSX" not in member
        }
        reader = csv.DictReader(
            io.TextIOWrapper(self._archive.open(csv_member), encoding="utf-8-sig")
        )
        records = []
        for row in reader:
            suite_id = int(row["suite_id"])
            sample_id = int(row["sample_id"])
            code = int(row["code"])
            filename = f"input_{suite_id}_{sample_id}_{code}.jpg"
            records.append(
                ChineseMNISTRecord(
                    suite_id=suite_id,
                    sample_id=sample_id,
                    code=code,
                    value=int(row["value"]),
                    character=row["character"],
                    archive_member=images_by_name[filename],
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_bytes = self._archive.read(record.archive_member)
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label


def writer_disjoint_split(
    records: Sequence[ChineseMNISTRecord],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split indices by writer ID so no writer leaks across partitions."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    writer_ids = sorted({record.suite_id for record in records})
    if len(writer_ids) < 2:
        raise ValueError("at least two writers are required for a disjoint split")
    random.Random(seed).shuffle(writer_ids)
    split_at = max(1, min(len(writer_ids) - 1, round(len(writer_ids) * train_ratio)))
    train_writers = set(writer_ids[:split_at])
    train_indices = [
        index for index, record in enumerate(records) if record.suite_id in train_writers
    ]
    test_indices = [
        index for index, record in enumerate(records) if record.suite_id not in train_writers
    ]
    return train_indices, test_indices
