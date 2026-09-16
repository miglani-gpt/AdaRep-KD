from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits, _ = self.model(images)

            loss = self.loss_fn(logits, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits, _ = self.model(images)

            loss = self.loss_fn(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_metrics: Dict[str, float],
        extra: Optional[Dict] = None,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = (
                self.scheduler.state_dict()
            )

        if extra is not None:
            checkpoint.update(extra)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> Dict:
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if (
            self.scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        return checkpoint

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_path: Optional[Path] = None,
        checkpoint_extra: Optional[Dict] = None,
    ) -> Dict[str, list]:

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        best_val_loss = float("inf")
        best_val_loss_epoch = None
        val_accuracy_at_best_val_loss = None

        max_val_accuracy = float("-inf")
        max_val_accuracy_epoch = None

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            history["train_loss"].append(
                train_metrics["loss"]
            )
            history["train_accuracy"].append(
                train_metrics["accuracy"]
            )
            history["val_loss"].append(
                val_metrics["loss"]
            )
            history["val_accuracy"].append(
                val_metrics["accuracy"]
            )
            history["learning_rate"].append(current_lr)

            current_epoch = epoch + 1

            # ----------------------------------------------------
            # Track best validation loss
            # ----------------------------------------------------

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_loss_epoch = current_epoch
                val_accuracy_at_best_val_loss = val_metrics["accuracy"]

                if checkpoint_path is not None:
                    self.save_checkpoint(
                        path=checkpoint_path,
                        epoch=current_epoch,
                        val_metrics=val_metrics,
                        extra=checkpoint_extra,
                    )

            # ----------------------------------------------------
            # Track maximum validation accuracy
            # ----------------------------------------------------

            if val_metrics["accuracy"] > max_val_accuracy:
                max_val_accuracy = val_metrics["accuracy"]
                max_val_accuracy_epoch = current_epoch

            print(
                f"Epoch [{current_epoch}/{epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train Acc: {train_metrics['accuracy']:.4f} "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val Acc: {val_metrics['accuracy']:.4f} "
                f"LR: {current_lr:.6f}"
            )

        # --------------------------------------------------------
        # Restore best validation-loss checkpoint
        # --------------------------------------------------------

        if checkpoint_path is not None and best_val_loss_epoch is not None:
            self.load_checkpoint(checkpoint_path)

        # --------------------------------------------------------
        # Store summary statistics
        # --------------------------------------------------------

        history["best_val_loss"] = best_val_loss
        history["best_val_loss_epoch"] = best_val_loss_epoch
        history["val_accuracy_at_best_val_loss"] = (
            val_accuracy_at_best_val_loss
        )

        history["max_val_accuracy"] = max_val_accuracy
        history["max_val_accuracy_epoch"] = max_val_accuracy_epoch

        return history