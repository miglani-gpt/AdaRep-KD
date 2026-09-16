from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# CIFAR-100 normalization statistics
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

DEFAULT_SPLIT_SEED = 42
DEFAULT_BATCH_SIZE = 128


def get_cifar100_transforms():

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    return train_transform, eval_transform


def create_cifar100_datasets(
    data_dir,
    split_seed=DEFAULT_SPLIT_SEED,
):


    data_dir = Path(data_dir)

    train_transform, eval_transform = get_cifar100_transforms()

    # Load the training data twice so that train and validation
    # can have different transformations while sharing the same samples.
    train_data = datasets.CIFAR100(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=False,
    )

    val_data = datasets.CIFAR100(
        root=data_dir,
        train=True,
        transform=eval_transform,
        download=False,
    )

    test_data = datasets.CIFAR100(
        root=data_dir,
        train=False,
        transform=eval_transform,
        download=False,
    )

    # Deterministic 45k / 5k split.
    generator = torch.Generator()
    generator.manual_seed(split_seed)

    indices = torch.randperm(
        len(train_data),
        generator=generator,
    ).tolist()

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_dataset = Subset(train_data, train_indices)
    val_dataset = Subset(val_data, val_indices)

    return train_dataset, val_dataset, test_data


def create_cifar100_dataloaders(
    data_dir,
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=0,
    split_seed=DEFAULT_SPLIT_SEED,
    pin_memory=None,
):
   
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_dataset, val_dataset, test_dataset = create_cifar100_datasets(
        data_dir=data_dir,
        split_seed=split_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader