"""Evaluate a checkpoint against every image in the packaged dataset."""

import argparse
from pathlib import Path

import torch

from chinese_digits.data import ChineseMNISTDataset
from chinese_digits.evaluation import evaluate_model
from chinese_digits.inference import preprocess_image
from chinese_digits.model import CLASS_NAMES, CLASS_VALUES, load_model


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run transparent dataset-wide checkpoint evaluation."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "MNIST7646.zip",
        help="path to the Chinese MNIST ZIP archive",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "pthmnist_cnn.pth",
        help="path to a compatible PyTorch state dictionary",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cpu or cuda",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = ChineseMNISTDataset(args.data, transform=preprocess_image)
    model = load_model(args.weights, device=args.device)
    result = evaluate_model(
        model, dataset, batch_size=args.batch_size, device=args.device
    )

    print("Dataset-wide checkpoint evaluation (not a held-out test split)")
    print(f"Accuracy: {result.accuracy:.2f}% ({result.correct:,}/{result.total:,})")
    print("\nClass     Value       Correct     Accuracy")
    print("-" * 47)
    for index, character in enumerate(CLASS_NAMES):
        class_result = result.per_class[index]
        print(
            f"{character:<9} {CLASS_VALUES[index]:>9,}     "
            f"{class_result.correct:>4}/{class_result.total:<4}     "
            f"{class_result.accuracy:>6.2f}%"
        )


if __name__ == "__main__":
    main()
