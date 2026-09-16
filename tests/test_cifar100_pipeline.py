from pathlib import Path

from adarep_kd.data.cifar100 import (
    create_cifar100_datasets,
    create_cifar100_dataloaders,
)


DATA_DIR = Path("data/raw")
SPLIT_SEED = 42


def test_dataset_sizes():
    train, val, test = create_cifar100_datasets(
        DATA_DIR,
        split_seed=SPLIT_SEED,
    )

    assert len(train) == 45_000
    assert len(val) == 5_000
    assert len(test) == 10_000


def test_train_validation_split_is_deterministic():
    train_1, val_1, _ = create_cifar100_datasets(
        DATA_DIR,
        split_seed=SPLIT_SEED,
    )

    train_2, val_2, _ = create_cifar100_datasets(
        DATA_DIR,
        split_seed=SPLIT_SEED,
    )

    assert train_1.indices == train_2.indices
    assert val_1.indices == val_2.indices


def test_train_validation_split_has_no_overlap():
    train, val, _ = create_cifar100_datasets(
        DATA_DIR,
        split_seed=SPLIT_SEED,
    )

    train_indices = set(train.indices)
    val_indices = set(val.indices)

    assert train_indices.isdisjoint(val_indices)


def test_train_validation_split_covers_original_training_set():
    train, val, _ = create_cifar100_datasets(
        DATA_DIR,
        split_seed=SPLIT_SEED,
    )

    combined_indices = set(train.indices) | set(val.indices)

    assert len(combined_indices) == 50_000
    assert combined_indices == set(range(50_000))


def test_test_set_isolated():
    train, val, test = create_cifar100_datasets(
        DATA_DIR,
        split_seed=SPLIT_SEED,
    )

    assert len(train) == 45_000
    assert len(val) == 5_000
    assert len(test) == 10_000

    assert train.dataset is not test
    assert val.dataset is not test


def test_image_shape_and_labels():
    train, val, test = create_cifar100_datasets(
        DATA_DIR,
        split_seed=SPLIT_SEED,
    )

    train_image, train_label = train[0]
    val_image, val_label = val[0]
    test_image, test_label = test[0]

    assert train_image.shape == (3, 32, 32)
    assert val_image.shape == (3, 32, 32)
    assert test_image.shape == (3, 32, 32)

    assert 0 <= train_label < 100
    assert 0 <= val_label < 100
    assert 0 <= test_label < 100


def test_dataloader_batch_shapes():
    train_loader, val_loader, test_loader = create_cifar100_dataloaders(
        DATA_DIR,
        batch_size=128,
        num_workers=0,
    )

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))

    assert train_images.shape == (128, 3, 32, 32)
    assert val_images.shape == (128, 3, 32, 32)
    assert test_images.shape == (128, 3, 32, 32)

    assert train_labels.shape == (128,)
    assert val_labels.shape == (128,)
    assert test_labels.shape == (128,)


def test_dataloader_dataset_sizes():
    train_loader, val_loader, test_loader = create_cifar100_dataloaders(
        DATA_DIR,
        batch_size=128,
        num_workers=0,
    )

    assert len(train_loader.dataset) == 45_000
    assert len(val_loader.dataset) == 5_000
    assert len(test_loader.dataset) == 10_000


def test_split_changes_with_different_seed():
    train_1, val_1, _ = create_cifar100_datasets(
        DATA_DIR,
        split_seed=42,
    )

    train_2, val_2, _ = create_cifar100_datasets(
        DATA_DIR,
        split_seed=123,
    )

    assert train_1.indices != train_2.indices
    assert val_1.indices != val_2.indices