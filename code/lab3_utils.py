"""Flowers: original CNN, transfer learning, and a compact design example."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FLOWER_CLASSES = {"daisy", "dandelion", "rose", "sunflower", "tulip"}


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def image_transform(image_size=224, augment=False):
    ops = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def prepare_flowers(folder="data", validation_target=1000, seed=0,
                    image_size=224, augment=False):
    """Keep byte-identical files in one split; validation size is approximate."""
    folder = Path(folder).resolve()
    train_dataset = datasets.ImageFolder(folder, transform=image_transform(image_size, augment))
    val_dataset = datasets.ImageFolder(folder, transform=image_transform(image_size, False))
    if set(train_dataset.classes) != FLOWER_CLASSES:
        raise ValueError("Expected five class directories: " + ", ".join(sorted(FLOWER_CLASSES)))
    if train_dataset.samples != val_dataset.samples:
        raise ValueError("Training and validation file inventories differ")
    count = len(train_dataset)
    if not 5 <= validation_target <= count - 5:
        raise ValueError("validation_target must leave at least five images in each part")
    hashes = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p, _ in train_dataset.samples]
    labels = np.asarray(train_dataset.targets)
    labels_per_hash = {}
    for digest, label in zip(hashes, labels):
        labels_per_hash.setdefault(digest, set()).add(int(label))
    if any(len(v) > 1 for v in labels_per_hash.values()):
        raise ValueError("Identical files have different class labels; resolve before splitting")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_target / count,
                                 random_state=seed)
    tr, va = next(splitter.split(np.arange(count), labels, hashes))
    if set(labels[tr]) != set(range(5)) or set(labels[va]) != set(range(5)):
        raise ValueError("A split is missing classes; revise the protocol before training")
    rows = []
    tr_set = set(tr)
    for i, ((path, target), digest) in enumerate(zip(train_dataset.samples, hashes)):
        rows.append(dict(path=str(Path(path).relative_to(folder)), target=int(target),
            sha256=digest, split="training" if i in tr_set else "validation"))
    return dict(train_dataset=train_dataset, validation_dataset=val_dataset,
                train_indices=tr, validation_indices=va, inventory=pd.DataFrame(rows),
                classes=train_dataset.classes, class_to_idx=train_dataset.class_to_idx,
                config=dict(seed=seed, image_size=image_size, augment=augment,
                    validation_target=validation_target, validation_actual=len(va),
                    training_actual=len(tr), duplicate_files=count-len(set(hashes)),
                    mean=list(MEAN), std=list(STD), split="GroupShuffleSplit on SHA-256"))


def _seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loaders(data, batch_size=32, seed=0, num_workers=0):
    """Create loaders anew for each model so the random sequence starts anew."""
    loaders = {}
    for name, dataset, indices in [
        ("training", data["train_dataset"], data["train_indices"]),
        ("validation", data["validation_dataset"], data["validation_indices"])]:
        loaders[name] = DataLoader(Subset(dataset, indices), batch_size=batch_size,
            shuffle=name == "training", num_workers=num_workers,
            worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(seed),
            pin_memory=torch.cuda.is_available(), drop_last=False)
    return loaders


def original_cnn():
    """Exact layer dimensions from the received lab, for RGB 224x224 input."""
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(), nn.Linear(256 * 28 * 28, 1024), nn.ReLU(),
        nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 5))
    model.lab_description = {"architecture": "original_cnn", "pretrained": False}
    return model


def compact_example():
    """A design example; the student's own model must include a justified change."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 5))
    model.lab_description = {"architecture": "compact_example", "pretrained": False}
    return model


def transfer_resnet18(pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.frozen_backbone = True
    model.lab_description = {"architecture": "resnet18_frozen", "pretrained": pretrained,
        "weights": "IMAGENET1K_V1" if pretrained else None}
    return model


def parameter_counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_epoch(model, loader, device="cpu", optimizer=None):
    training = optimizer is not None
    if training and getattr(model, "frozen_backbone", False):
        model.eval()  # Freeze BatchNorm buffers as well as parameter gradients.
        model.fc.train()
    else:
        model.train(training)
    total_loss, total, actual, predicted = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape != (len(labels), 5):
                raise ValueError("Expected five logits per image")
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss; inspect inputs and learning rate")
            if training:
                loss.backward()
                optimizer.step()
            bs = len(labels)
            total_loss += float(loss.detach()) * bs
            total += bs
            actual.extend(labels.detach().cpu().tolist())
            predicted.extend(logits.detach().argmax(1).cpu().tolist())
    if not total:
        raise ValueError("Empty data loader")
    return dict(loss=total_loss / total, accuracy=float(accuracy_score(actual, predicted)),
        macro_f1=float(f1_score(actual, predicted, labels=range(5), average="macro", zero_division=0)),
        actual=actual, predicted=predicted, n=total)


def fit_flowers(model, data, folder, epochs=10, batch_size=32, learning_rate=1e-3,
                seed=0, device=None):
    """Use a freshly initialized model; saves the best validation macro-F1 epoch."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Positive epochs, batch size and learning rate are required")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loaders = make_loaders(data, batch_size=batch_size, seed=seed)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=learning_rate)
    data["inventory"].to_csv(folder / "split.csv", index=False)
    records, best, best_epoch = [], -1.0, None
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, loaders["training"], device, optimizer)
        va = run_epoch(model, loaders["validation"], device)
        records.append(dict(epoch=epoch, train_loss=tr["loss"], val_loss=va["loss"],
            train_accuracy=tr["accuracy"], val_accuracy=va["accuracy"],
            train_macro_f1=tr["macro_f1"], val_macro_f1=va["macro_f1"]))
        pd.DataFrame(records).to_csv(folder / "history.csv", index=False)
        if va["macro_f1"] > best:
            best, best_epoch = va["macro_f1"], epoch
            torch.save(model.state_dict(), folder / "best_weights.pth")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    model.load_state_dict(torch.load(folder / "best_weights.pth", map_location=device,
                                     weights_only=True))
    final = run_epoch(model, loaders["validation"], device)
    val_rows = data["inventory"].iloc[data["validation_indices"]].copy()
    val_rows["predicted"] = final["predicted"]
    val_rows.to_csv(folder / "validation_predictions.csv", index=False)
    report = classification_report(final["actual"], final["predicted"], labels=range(5),
        target_names=data["classes"], zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(folder / "validation_report.csv")
    pd.DataFrame(confusion_matrix(final["actual"], final["predicted"], labels=range(5)),
        index=data["classes"], columns=data["classes"]).to_csv(folder / "confusion_matrix.csv")
    manifest = dict(data=data["config"], class_to_idx=data["class_to_idx"],
        architecture=getattr(model, "lab_description", {"architecture": type(model).__name__}),
        parameters=parameter_counts(model), epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, optimizer="Adam", seed=seed, device=str(device),
        best_epoch=best_epoch, selection="highest validation macro-F1; earliest tie",
        seconds_fit_and_checkpoint_io=seconds,
        validation_metrics={k: final[k] for k in ("loss", "accuracy", "macro_f1")},
        versions={p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "pandas")},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
        allow_nan=False), encoding="utf-8")
    (folder / "architecture.txt").write_text(str(model), encoding="utf-8")
    return dict(model=model, history=pd.DataFrame(records), metrics=manifest["validation_metrics"],
                manifest=manifest, validation=final)


def denormalize(image):
    mean = image.new_tensor(MEAN)[:, None, None]
    std = image.new_tensor(STD)[:, None, None]
    return (image.detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1)


def plot_learning(result):
    import matplotlib.pyplot as plt
    h = result["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for phase, label in [("train", "training"), ("val", "validation")]:
        axes[0].plot(h["epoch"], h[phase + "_loss"], label=label)
        axes[1].plot(h["epoch"], h[phase + "_macro_f1"], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(xlabel="Epoch", ylabel="Macro-F1")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    return fig
