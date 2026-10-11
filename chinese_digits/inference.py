"""Image preprocessing and prediction helpers."""

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import nn

from .model import CLASS_NAMES, CLASS_VALUES


@dataclass(frozen=True)
class Prediction:
    index: int
    character: str
    value: int
    confidence: float


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert one image into the normalized 1x28x28 model input."""

    grayscale = image.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
    pixels = np.array(grayscale, dtype=np.float32, copy=True)
    return torch.from_numpy(pixels).unsqueeze(0).div(127.5).sub(1.0)


def predict_image(
    model: nn.Module,
    image: Image.Image,
    device: Union[str, torch.device] = "cpu",
) -> Prediction:
    """Run inference for one image and return its decoded prediction."""

    target_device = torch.device(device)
    inputs = preprocess_image(image).unsqueeze(0).to(target_device)
    model = model.to(target_device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    confidence, index_tensor = probabilities.max(dim=0)
    index = int(index_tensor.item())
    return Prediction(
        index=index,
        character=CLASS_NAMES[index],
        value=CLASS_VALUES[index],
        confidence=float(confidence.item()),
    )
