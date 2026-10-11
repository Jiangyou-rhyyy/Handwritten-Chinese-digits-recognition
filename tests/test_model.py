from pathlib import Path
import unittest

import torch

from chinese_digits.model import CLASS_NAMES, ChineseDigitCNN, load_model


class ModelTests(unittest.TestCase):
    def test_model_returns_one_logit_per_supported_character(self):
        model = ChineseDigitCNN()

        output = model(torch.zeros(2, 1, 28, 28))

        self.assertEqual(output.shape, (2, 15))
        self.assertEqual(
            CLASS_NAMES,
            (
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
            ),
        )


    def test_packaged_checkpoint_loads_into_the_shared_model(self):
        weights = Path(__file__).parents[1] / "pthmnist_cnn.pth"

        model = load_model(weights, device="cpu")
        output = model(torch.zeros(1, 1, 28, 28))

        self.assertEqual(output.shape, (1, 15))
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
