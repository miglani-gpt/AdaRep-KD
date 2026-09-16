from pathlib import Path
import json

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

HISTORY_PATH = Path(
    "outputs/logs/teacher_resnet32_history.json"
)

FIGURE_DIR = Path("outputs/figures")


# ============================================================
# Load training history
# ============================================================

def load_history(path: Path) -> dict:
    with open(path, "r") as file:
        return json.load(file)


# ============================================================
# Plot loss
# ============================================================

def plot_loss(history: dict, output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    best_val_loss_epoch = history["best_val_loss_epoch"]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.axvline(
        best_val_loss_epoch,
        linestyle="--",
        label=f"Best Val Loss Epoch ({best_val_loss_epoch})",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet-32 Teacher — Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# Plot accuracy
# ============================================================

def plot_accuracy(history: dict, output_path: Path) -> None:
    epochs = range(1, len(history["train_accuracy"]) + 1)
    max_val_accuracy_epoch = history["max_val_accuracy_epoch"]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.axvline(
        max_val_accuracy_epoch,
        linestyle="--",
        label=f"Max Val Accuracy Epoch ({max_val_accuracy_epoch})",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet-32 Teacher — Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# Plot learning rate
# ============================================================

def plot_learning_rate(history: dict, output_path: Path) -> None:
    epochs = range(1, len(history["learning_rate"]) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["learning_rate"],
        label="Learning Rate",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("ResNet-32 Teacher — Learning Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    history = load_history(HISTORY_PATH)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_loss(
        history,
        FIGURE_DIR / "teacher_resnet32_loss.png",
    )

    plot_accuracy(
        history,
        FIGURE_DIR / "teacher_resnet32_accuracy.png",
    )

    plot_learning_rate(
        history,
        FIGURE_DIR / "teacher_resnet32_learning_rate.png",
    )

    print("Teacher training plots saved to:")
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
