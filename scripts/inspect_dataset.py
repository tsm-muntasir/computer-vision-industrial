
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_images(folder: Path) -> list[Path]:
    """Return supported image files under a folder."""
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def inspect_images(image_paths: list[Path]) -> tuple[list[dict], list[str]]:
    """Collect image metadata and identify unreadable files."""
    records = []
    errors = []

    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.verify()

            # Reopen after verify() before accessing image properties.
            with Image.open(path) as image:
                records.append({
                    "path": str(path),
                    "filename": path.name,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "format": image.format,
                })

        except (OSError, ValueError, UnidentifiedImageError) as exc:
            errors.append(f"{path}: {exc}")

    return records, errors


def create_sample_grid(
    folders: list[tuple[str, Path]],
    output_path: Path,
    max_per_class: int,
) -> None:
    """Save a grid showing example images from each folder."""
    samples = []

    for label, folder in folders:
        images = find_images(folder)[:max_per_class]

        for image_path in images:
            samples.append((label, image_path))

    if not samples:
        print("No images found. Check the dataset directory.")
        return

    columns = min(4, len(samples))
    rows = (len(samples) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.5 * rows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, (label, image_path) in zip(axes.flat, samples):
        try:
            with Image.open(image_path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(label, fontsize=9)
            ax.axis("off")
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            ax.set_title(f"Unreadable: {image_path.name}")
            print(f"Could not display {image_path}: {exc}")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSample grid saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an MVTec AD category."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/mvtec_anomaly_detection"),
    )
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--max-per-class", type=int, default=4)
    args = parser.parse_args()

    category_dir = args.data_dir / args.category
    train_dir = category_dir / "train"
    test_dir = category_dir / "test"

    if not category_dir.is_dir():
        raise FileNotFoundError(
            f"Category folder not found: {category_dir}\n"
            "Download and extract MVTec AD, then check --data-dir."
        )

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected train/ and test/ under {category_dir}"
        )

    folders = []

    for split_dir in (train_dir, test_dir):
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                folders.append((f"{split_dir.name}/{class_dir.name}", class_dir))

    all_records = []
    all_errors = []

    print("=" * 65)
    print(f"MVTec AD DATA AUDIT — {args.category}")
    print("=" * 65)

    for label, folder in folders:
        paths = find_images(folder)
        records, errors = inspect_images(paths)

        for record in records:
            record["split_class"] = label

        all_records.extend(records)
        all_errors.extend(errors)

        print(
            f"{label:25} "
            f"found={len(paths):5} "
            f"readable={len(records):5} "
            f"errors={len(errors):5}"
        )

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if all_records:
        dataframe = pd.DataFrame(all_records)
        csv_path = reports_dir / f"{args.category}_image_inventory.csv"
        dataframe.to_csv(csv_path, index=False)

        print(f"\nTotal readable images: {len(dataframe)}")
        print("\nImage dimensions:")
        print(dataframe.groupby(
            ["width", "height"]
        ).size().sort_values(ascending=False).head(10))

        print("\nImage color modes:")
        print(dataframe["mode"].value_counts())

        print(f"\nInventory saved to: {csv_path}")
    else:
        print("\nNo readable images were found.")

    if all_errors:
        error_path = reports_dir / f"{args.category}_image_errors.txt"
        error_path.write_text("\n".join(all_errors), encoding="utf-8")
        print(f"Unreadable image details saved to: {error_path}")

    create_sample_grid(
        folders,
        reports_dir / f"{args.category}_sample_grid.png",
        args.max_per_class,
    )

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
