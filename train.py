"""Train the compact CNN with a deterministic writer-disjoint split."""

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from chinese_digits.data import ChineseMNISTDataset, writer_disjoint_split
from chinese_digits.evaluation import evaluate_model
from chinese_digits.model import ChineseDigitCNN


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the 15-class Chinese digit CNN."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "MNIST7646.zip",
        help="path to the Chinese MNIST ZIP archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "pthmnist_cnn.pth",
        help="where to save the best state dictionary",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cpu or cuda",
    )
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = build_parser().parse_args()
    from torchvision import transforms

    seed_everything(args.seed)
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    metadata = ChineseMNISTDataset(args.data)
    train_indices, test_indices = writer_disjoint_split(
        metadata.records, train_ratio=0.8, seed=args.seed
    )
    training_data = Subset(
        ChineseMNISTDataset(args.data, transform=train_transform), train_indices
    )
    test_data = Subset(
        ChineseMNISTDataset(args.data, transform=evaluation_transform), test_indices
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    device = torch.device(args.device)
    model = ChineseDigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0

    print(
        f"Training on {len(training_data):,} images; evaluating on "
        f"{len(test_data):,} writer-disjoint images ({device})."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        evaluation = evaluate_model(
            model, test_data, batch_size=args.batch_size, device=device
        )
        average_loss = running_loss / len(train_loader)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: loss={average_loss:.4f}, "
            f"writer-disjoint accuracy={evaluation.accuracy:.2f}%"
        )
        if evaluation.accuracy > best_accuracy:
            best_accuracy = evaluation.accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)

    print(f"Best checkpoint: {best_accuracy:.2f}% -> {args.output}")


if __name__ == "__main__":
    main()
