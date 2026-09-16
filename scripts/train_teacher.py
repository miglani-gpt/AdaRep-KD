import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from adarep_kd.data.cifar100 import create_cifar100_dataloaders
from adarep_kd.models import resnet32
from adarep_kd.training import Trainer
from adarep_kd.utils.reproducibility import set_seed


# ============================================================
# Experiment configuration
# ============================================================

SEED = 42

BATCH_SIZE = 128
EPOCHS = 200

LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_CLASSES = 100

DATA_DIR = Path("data/raw")

CHECKPOINT_DIR = Path("outputs/checkpoints")
LOG_DIR = Path("outputs/logs")

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "teacher_resnet32_best.pt"
)

HISTORY_PATH = (
    LOG_DIR / "teacher_resnet32_history.json"
)


# ============================================================
# Main experiment
# ============================================================

def main():
    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("AdaRep-KD Teacher Baseline")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {SEED}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader, test_loader = (
        create_cifar100_dataloaders(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=0,
            split_seed=SEED,
            pin_memory=torch.cuda.is_available(),
        )
    )

    print("Dataset:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Validation: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    print()

    # --------------------------------------------------------
    # Teacher model
    # --------------------------------------------------------

    model = resnet32(
        num_classes=NUM_CLASSES
    )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    loss_fn = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_extra={
            "model": "ResNet-32",
            "dataset": "CIFAR-100",
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
        },
    )

    # --------------------------------------------------------
    # Evaluate best checkpoint on test set
    # --------------------------------------------------------

    test_metrics = trainer.validate(test_loader)

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    history["test_loss"] = test_metrics["loss"]
    history["test_accuracy"] = test_metrics["accuracy"]

    with open(HISTORY_PATH, "w") as file:
        json.dump(history, file, indent=2)

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Teacher Baseline Complete")
    print("=" * 60)

    print(
        f"Best validation-loss epoch: "
        f"{history['best_val_loss_epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{history['best_val_loss']:.4f}"
    )

    print(
        f"Validation accuracy at best-loss epoch: "
        f"{history['val_accuracy_at_best_val_loss']:.4f}"
    )

    print(
        f"Maximum validation accuracy: "
        f"{history['max_val_accuracy']:.4f}"
    )

    print(
        f"Maximum validation-accuracy epoch: "
        f"{history['max_val_accuracy_epoch']}"
    )

    print(
        f"Test loss: "
        f"{test_metrics['loss']:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print()
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"History: {HISTORY_PATH}")

if __name__ == "__main__":
    main()