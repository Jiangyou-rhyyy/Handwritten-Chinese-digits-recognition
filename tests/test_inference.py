import unittest

import numpy as np
import torch
from PIL import Image
from torch import nn

from chinese_digits.inference import predict_image, preprocess_image


class _AlwaysLastClass(nn.Module):
    def forward(self, inputs):
        logits = torch.zeros((inputs.shape[0], 15), device=inputs.device)
        logits[:, -1] = 10.0
        return logits


class InferenceTests(unittest.TestCase):
    def test_preprocessing_produces_normalized_model_input(self):
        image = Image.new("L", (64, 64), color=255)

        tensor = preprocess_image(image)

        self.assertEqual(tensor.shape, (1, 28, 28))
        self.assertTrue(torch.allclose(tensor, torch.ones_like(tensor)))

    def test_preprocessing_matches_the_bilinear_resize_used_for_training(self):
        pixels = np.zeros((7, 7), dtype=np.uint8)
        pixels[2:5, 3] = 255
        image = Image.fromarray(pixels, mode="L")
        resized = image.resize((28, 28), Image.Resampling.BILINEAR)
        expected = torch.from_numpy(np.array(resized, dtype=np.float32, copy=True))
        expected = expected.unsqueeze(0).div(127.5).sub(1.0)

        tensor = preprocess_image(image)

        self.assertTrue(torch.equal(tensor, expected))

    def test_prediction_returns_character_value_and_confidence(self):
        prediction = predict_image(
            _AlwaysLastClass(), Image.new("L", (28, 28), color=0), device="cpu"
        )

        self.assertEqual(prediction.index, 14)
        self.assertEqual(prediction.character, "亿")
        self.assertEqual(prediction.value, 100_000_000)
        self.assertGreater(prediction.confidence, 0.99)


if __name__ == "__main__":
    unittest.main()
