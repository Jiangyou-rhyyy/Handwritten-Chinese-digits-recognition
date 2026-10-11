"""Model evaluation helpers shared by the CLI and tests."""

from dataclasses import dataclass
from typing import Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ClassResult:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0


@dataclass(frozen=True)
class EvaluationResult:
    correct: int
    total: int
    per_class: Dict[int, ClassResult]

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> EvaluationResult:
    """Evaluate a classifier and return aggregate and per-class counts."""

    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    class_counts: Dict[int, list[int]] = {}
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(target_device)
            labels = labels.to(target_device)
            predictions = model(images).argmax(dim=1)
            matches = predictions.eq(labels)
            correct += int(matches.sum().item())
            total += labels.numel()
            for label, matched in zip(labels.tolist(), matches.tolist()):
                counts = class_counts.setdefault(label, [0, 0])
                counts[0] += int(matched)
                counts[1] += 1
    per_class = {
        label: ClassResult(correct=counts[0], total=counts[1])
        for label, counts in class_counts.items()
    }
    return EvaluationResult(correct=correct, total=total, per_class=per_class)
