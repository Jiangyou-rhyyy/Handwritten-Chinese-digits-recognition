"""CNN architecture and checkpoint loading utilities."""

from pathlib import Path
from typing import Union

import torch
from torch import nn


CLASS_NAMES = (
    "零",
    "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
    "十",
    "百",
    "千",
    "万",
    "亿",
)

CLASS_VALUES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10_000, 100_000_000)


class ChineseDigitCNN(nn.Module):
    """Compact 15-class CNN used by the packaged checkpoint."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def load_model(
    weights: Union[str, Path], device: Union[str, torch.device] = "cpu"
) -> ChineseDigitCNN:
    """Load a compatible state dictionary and return an evaluation model."""

    target_device = torch.device(device)
    model = ChineseDigitCNN().to(target_device)
    state = torch.load(Path(weights), map_location=target_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
