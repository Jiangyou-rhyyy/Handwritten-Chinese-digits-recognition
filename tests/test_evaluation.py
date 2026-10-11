import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from chinese_digits.evaluation import evaluate_model


class _AlwaysClassOne(nn.Module):
    def forward(self, inputs):
        logits = torch.zeros((inputs.shape[0], 15), device=inputs.device)
        logits[:, 1] = 5.0
        return logits


class EvaluationTests(unittest.TestCase):
    def test_evaluation_reports_accuracy_and_per_class_counts(self):
        images = torch.zeros(4, 1, 28, 28)
        labels = torch.tensor([0, 1, 1, 2])
        dataset = TensorDataset(images, labels)

        result = evaluate_model(_AlwaysClassOne(), dataset, batch_size=2, device="cpu")

        self.assertEqual(result.total, 4)
        self.assertEqual(result.correct, 2)
        self.assertEqual(result.accuracy, 50.0)
        self.assertEqual(result.per_class[1].correct, 2)
        self.assertEqual(result.per_class[1].total, 2)


if __name__ == "__main__":
    unittest.main()
