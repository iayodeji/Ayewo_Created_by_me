import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 for malaria cell classification.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to cell_images directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (minimum 5).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--output", type=str, default="malaria_model.pth", help="Output model path.")
    args = parser.parse_args()
    if args.epochs < 5:
        parser.error("--epochs must be at least 5.")
    return args


def build_model() -> nn.Module:
    try:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    except (RuntimeError, OSError, ValueError):
        model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    if train_len == 0 or val_len == 0:
        raise ValueError("Dataset is too small. Provide enough images for train/validation split.")
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        val_acc = (correct / total) * 100 if total else 0.0
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_train_loss:.4f} - val_acc: {val_acc:.2f}%")

    output_path = Path(args.output)
    torch.save(model.state_dict(), output_path)
    print(f"Saved fine-tuned model to {output_path.resolve()}")


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments)
