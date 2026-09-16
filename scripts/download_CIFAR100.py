from pathlib import Path

from torchvision import datasets


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_data_dir = project_root / "data" / "raw"

    raw_data_dir.mkdir(parents=True, exist_ok=True)

    datasets.CIFAR100(
        root=raw_data_dir,
        train=True,
        download=True,
    )

    datasets.CIFAR100(
        root=raw_data_dir,
        train=False,
        download=True,
    )

    print(f"CIFAR-100 downloaded to: {raw_data_dir}")


if __name__ == "__main__":
    main()