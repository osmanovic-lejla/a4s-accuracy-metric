import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
from torchvision.models import resnet18, resnet34, resnet50


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    Multi-class generalization of Brier Score.

    y_true: shape (N,) with class indices
    proba:  shape (N, C) with per-class probabilities
    """
    n_samples, n_classes = proba.shape
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(n_samples), y_true] = 1.0
    # Mean over samples of sum over classes
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def get_cifar_loaders(
    root: str = "tests/data/cifar10",
    n_train: int = 2000,
    n_test: int = 1000,
    batch_size: int = 128,
):
    """
    Download CIFAR-10 if needed & return small train/test loaders.
    We use a subset (n_train / n_test) to keep tests reasonably fast.
    """

    transform = T.Compose(
        [
            T.ToTensor(),  # [0,1]
            # (Optionally you could normalize, but for this demo it's not required)
        ]
    )

    train_full = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    test_full = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )

    n_train = min(n_train, len(train_full))
    n_test = min(n_test, len(test_full))

    train_subset = Subset(train_full, range(n_train))
    test_subset = Subset(test_full, range(n_test))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_model(model: nn.Module, train_loader, device, epochs: int = 1):
    """
    Very small training loop just to make the model non-random.
    We keep epochs=1 to avoid long test runtime.
    """
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate_brier(model: nn.Module, test_loader, device) -> float:
    """
    Run model on test set and compute multi-class Brier Score.
    """

    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)  # logits
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    proba = np.concatenate(all_probs, axis=0)  # (N, C)
    y_true = np.concatenate(all_labels, axis=0)  # (N,)

    score = multiclass_brier_score(y_true, proba)
    return score


def save_sample_predictions(
    model: nn.Module,
    test_loader,
    device,
    folder: str,
    n_images: int = 10,
):
    """
    Save a few example images with true/pred labels as PNG files.

    Images are saved to:
        tests/data/cifar_images/<model_name>/img_<i>_trueY_predZ.png
    """

    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.eval()
    saved = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            for i in range(images.size(0)):
                if saved >= n_images:
                    return

                img = images[i].cpu()  # still in [0,1]
                true_label = int(labels[i].item())
                pred_label = int(preds[i].item())

                filename = f"img_{saved}_true{true_label}_pred{pred_label}.png"
                vutils.save_image(img, out_dir / filename)
                saved += 1


def test_cifar_resnets_brier_score_saved_to_csv_and_images():
    """
    End-to-end experiment test:
      - download CIFAR-10 if needed
      - train 3 ResNet models (18/34/50) on a small subset
      - compute multi-class Brier Score for each
      - save results into tests/data/measures/brier_score_cifar_resnets.csv
      - save ~10 sample images per model into tests/data/cifar_images/<model_name>/

    This is for experimentation & reporting, reusing the same Brier Score concept
    as the integrated A4S metric.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Data
    train_loader, test_loader = get_cifar_loaders(
        root="tests/data/cifar10",
        n_train=2000,
        n_test=1000,
        batch_size=128,
    )

    # 2) Models to compare
    def make_resnet18():
        return resnet18(weights=None, num_classes=10)

    def make_resnet34():
        return resnet34(weights=None, num_classes=10)

    def make_resnet50():
        return resnet50(weights=None, num_classes=10)

    model_builders = {
        "ResNet18": make_resnet18,
        "ResNet34": make_resnet34,
        "ResNet50": make_resnet50,
    }

    results = []

    # 3) Train & evaluate
    for name, builder in model_builders.items():
        model = builder()
        train_one_model(model, train_loader, device, epochs=1)
        score = evaluate_brier(model, test_loader, device)

        # Save a few example images for visual inspection later
        img_folder = f"tests/data/cifar_images/{name}"
        save_sample_predictions(model, test_loader, device, folder=img_folder, n_images=10)

        results.append(
            {
                "model": name,
                "brier_score": float(score),
            }
        )

    # 4) Save Brier Score results to CSV
    measures_dir = Path("tests/data/measures")
    measures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = measures_dir / "brier_score_cifar_resnets.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    # Basic sanity checks for test
    assert csv_path.exists()
    assert len(results) == 3
    for row in results:
        assert isinstance(row["brier_score"], float)