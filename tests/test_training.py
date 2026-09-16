import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adarep_kd.models import resnet8
from adarep_kd.training import Trainer


def create_dummy_dataloader(
    num_samples=16,
    num_classes=10,
    batch_size=4,
):
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(
        0,
        num_classes,
        (num_samples,),
    )

    dataset = TensorDataset(images, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def create_trainer(
    num_classes=10,
    learning_rate=0.01,
    scheduler=None,
):
    model = resnet8(num_classes=num_classes)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
        scheduler=scheduler,
    )


# ============================================================
# Initialization
# ============================================================


def test_trainer_initializes_model():
    trainer = create_trainer()

    assert trainer.model is not None
    assert trainer.loss_fn is not None
    assert trainer.optimizer is not None
    assert trainer.device == torch.device("cpu")


# ============================================================
# train_epoch
# ============================================================


def test_train_epoch_returns_metrics():
    trainer = create_trainer()

    dataloader = create_dummy_dataloader()

    metrics = trainer.train_epoch(dataloader)

    assert "loss" in metrics
    assert "accuracy" in metrics

    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)

    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_training_updates_model_parameters():
    trainer = create_trainer()

    dataloader = create_dummy_dataloader()

    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    trainer.train_epoch(dataloader)

    changed = any(
        not torch.equal(
            before[name],
            parameter,
        )
        for name, parameter in trainer.model.named_parameters()
    )

    assert changed


def test_train_epoch_sets_training_mode():
    trainer = create_trainer()

    dataloader = create_dummy_dataloader()

    trainer.model.eval()

    trainer.train_epoch(dataloader)

    assert trainer.model.training


# ============================================================
# validate
# ============================================================


def test_validate_returns_metrics_without_training():
    trainer = create_trainer()

    dataloader = create_dummy_dataloader()

    metrics = trainer.validate(dataloader)

    assert "loss" in metrics
    assert "accuracy" in metrics

    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)

    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_validate_does_not_update_model_parameters():
    trainer = create_trainer()

    dataloader = create_dummy_dataloader()

    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    trainer.validate(dataloader)

    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(
            before[name],
            parameter,
        )


def test_validate_sets_evaluation_mode():
    trainer = create_trainer()

    dataloader = create_dummy_dataloader()

    trainer.model.train()

    trainer.validate(dataloader)

    assert not trainer.model.training


# ============================================================
# fit
# ============================================================


def test_fit_returns_complete_history():
    trainer = create_trainer()

    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
    )

    assert len(history["train_loss"]) == 3
    assert len(history["train_accuracy"]) == 3
    assert len(history["val_loss"]) == 3
    assert len(history["val_accuracy"]) == 3
    assert len(history["learning_rate"]) == 3

    assert "best_val_loss" in history
    assert "best_val_loss_epoch" in history
    assert "val_accuracy_at_best_val_loss" in history
    assert "max_val_accuracy" in history
    assert "max_val_accuracy_epoch" in history


def test_fit_tracks_best_validation_metrics():
    trainer = create_trainer()

    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
    )

    assert history["best_val_loss"] == min(
        history["val_loss"]
    )

    assert history["max_val_accuracy"] == max(
        history["val_accuracy"]
    )

    best_loss_epoch = (
        history["val_loss"].index(
            history["best_val_loss"]
        ) + 1
    )

    max_accuracy_epoch = (
        history["val_accuracy"].index(
            history["max_val_accuracy"]
        ) + 1
    )

    assert (
        history["best_val_loss_epoch"]
        == best_loss_epoch
    )

    assert (
        history["max_val_accuracy_epoch"]
        == max_accuracy_epoch
    )

    assert (
        history["val_accuracy_at_best_val_loss"]
        == history["val_accuracy"][best_loss_epoch - 1]
    )


def test_scheduler_steps_once_per_epoch():
    trainer = create_trainer(
        learning_rate=0.01,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        trainer.optimizer,
        milestones=[1],
        gamma=0.01,
    )

    trainer.scheduler = scheduler

    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
    )

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(
        0.0001
    )


# ============================================================
# Checkpointing
# ============================================================


def test_fit_saves_best_checkpoint(tmp_path):
    trainer = create_trainer()

    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()

    checkpoint_path = tmp_path / "best.pt"

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert history["best_val_loss_epoch"] is not None
    assert history["best_val_loss"] is not None
    assert history["val_accuracy_at_best_val_loss"] is not None
    assert history["max_val_accuracy"] is not None
    assert history["max_val_accuracy_epoch"] is not None


def test_checkpoint_contains_required_state(tmp_path):
    trainer = create_trainer()

    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()

    checkpoint_path = tmp_path / "best.pt"

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert "epoch" in checkpoint
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "val_loss" in checkpoint
    assert "val_accuracy" in checkpoint


def test_checkpoint_contains_scheduler_state(tmp_path):
    trainer = create_trainer()

    trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        trainer.optimizer,
        milestones=[1],
        gamma=0.1,
    )

    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()

    checkpoint_path = tmp_path / "best.pt"

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert "scheduler_state_dict" in checkpoint


def test_checkpoint_extra_metadata_is_saved(tmp_path):
    trainer = create_trainer()

    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()

    checkpoint_path = tmp_path / "best.pt"

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        checkpoint_path=checkpoint_path,
        checkpoint_extra={
            "model": "ResNet-8",
            "dataset": "CIFAR-100",
            "seed": 42,
        },
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    assert checkpoint["model"] == "ResNet-8"
    assert checkpoint["dataset"] == "CIFAR-100"
    assert checkpoint["seed"] == 42


def test_load_checkpoint_restores_model_state(tmp_path):
    trainer = create_trainer()

    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()

    checkpoint_path = tmp_path / "best.pt"

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    expected_state = copy.deepcopy(
        checkpoint["model_state_dict"]
    )

    # Modify the model after checkpoint creation.
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)

    trainer.load_checkpoint(checkpoint_path)

    for name, parameter in trainer.model.state_dict().items():
        assert torch.equal(
            parameter,
            expected_state[name],
        )