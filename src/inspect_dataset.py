from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import matplotlib

matplotlib.use("Agg")  # Save figures in headless environments too.
import matplotlib.pyplot as plt
import numpy as np


SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png"}

## to calculate the sha
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_for_display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    raise ValueError(f"Unexpected channel count: {image.shape[2]}")


def inspect_dataset(data_dir: Path, output_dir: Path, samples_per_class: int) -> dict:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_dir}")
    if samples_per_class < 1:
        raise ValueError("--samples must be at least 1")

    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found inside: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory: list[dict] = []
    examples: dict[str, list[tuple[Path, np.ndarray]]] = defaultdict(list)
    hashes: dict[str, list[dict]] = defaultdict(list)
    all_file_counts: Counter[str] = Counter()
    valid_file_counts: Counter[str] = Counter()
    rejected: list[dict] = []

    for class_dir in [data_dir, *class_dirs]:
        label = class_dir.name if class_dir != data_dir else ""
        paths = class_dir.glob("*") if class_dir == data_dir else class_dir.rglob("*")
        for path in sorted(item for item in paths if item.is_file()):
            all_file_counts[label] += 1
            row = {
                "path": path.as_posix(),
                "label": label,
                "extension": path.suffix.lower(),
                "supported_extension": path.suffix.lower() in SUPPORTED_EXTENSIONS,
                "readable": False,
                "width": "",
                "height": "",
                "channels": "",
                "dtype": "",
                "file_bytes": "",
                "sha256": "",
                "error": "",
            }
            if label not in {"glioma", "meningioma", "notumor", "pituitary"}:
                row["error"] = "File is outside a supported class folder"
                rejected.append({"path": row["path"], "label": label, "reason": row["error"]})
                inventory.append(row)
                continue
            if not row["supported_extension"]:
                row["error"] = "Unsupported image extension"
                rejected.append({"path": row["path"], "label": label, "reason": row["error"]})
                inventory.append(row)
                continue

            image = None
            try:
                row["file_bytes"] = path.stat().st_size
                # imdecode + fromfile handles paths more reliably than imread on some systems.
                encoded = np.fromfile(path, dtype=np.uint8)
                if encoded.size:
                    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
                if image is None:
                    row["error"] = "OpenCV could not decode the image"
                else:
                    row["height"], row["width"] = image.shape[:2]
                    row["channels"] = 1 if image.ndim == 2 else image.shape[2]
                    row["dtype"] = str(image.dtype)
                    row["sha256"] = sha256_file(path)
                    row["readable"] = True
                    valid_file_counts[label] += 1
                    hashes[row["sha256"]].append({"path": row["path"], "label": label})
                    if len(examples[label]) < samples_per_class:
                        examples[label].append((path, image_for_display(image)))
            except (OSError, cv2.error, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"

            if row["error"]:
                rejected.append({"path": row["path"], "label": label, "reason": row["error"]})
            inventory.append(row)

    inventory_path = output_dir / "image_inventory.csv"
    inventory_fields = [
        "path", "label", "extension", "supported_extension", "readable",
        "width", "height", "channels", "dtype", "file_bytes", "sha256", "error",
    ]
    with inventory_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=inventory_fields)
        writer.writeheader()
        writer.writerows(inventory)

    rejected_path = output_dir / "rejected_files.csv"
    with rejected_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["path", "label", "reason"])
        writer.writeheader()
        writer.writerows(rejected)

    duplicate_groups = [
        {"sha256": digest, "copies": len(items), "files": items}
        for digest, items in sorted(hashes.items())
        if len(items) > 1
    ]
    duplicate_report = {
        "duplicate_groups": len(duplicate_groups),
        "extra_copies": sum(group["copies"] - 1 for group in duplicate_groups),
        "groups_spanning_labels": sum(
            len({item["label"] for item in group["files"]}) > 1 for group in duplicate_groups
        ),
        "groups_spanning_Tr_Te_prefixes": sum(
            {"Tr", "Te"}.issubset({Path(item["path"]).name.split("-")[0] for item in group["files"]})
            for group in duplicate_groups
        ),
        "groups": duplicate_groups,
    }
    (output_dir / "duplicate_audit.json").write_text(
        json.dumps(duplicate_report, indent=2) + "\n", encoding="utf-8"
    )

    labels = [path.name for path in class_dirs]
    counts_figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, [valid_file_counts[label] for label in labels], color="#5b8db8")
    axis.set(title="Readable images per class", xlabel="Class folder", ylabel="Image count")
    axis.tick_params(axis="x", rotation=15)
    counts_figure.tight_layout()
    counts_figure.savefig(output_dir / "class_counts.png", dpi=150)
    plt.close(counts_figure)

    columns = max(1, samples_per_class)
    grid, axes = plt.subplots(
        len(labels), columns, figsize=(3 * columns, 2.7 * len(labels)), squeeze=False
    )
    for row_index, label in enumerate(labels):
        for column_index in range(columns):
            axis = axes[row_index][column_index]
            axis.axis("off")
            if column_index < len(examples[label]):
                path, display_image = examples[label][column_index]
                axis.imshow(display_image, cmap="gray" if display_image.ndim == 2 else None)
                axis.set_title(f"{label}\n{path.name}", fontsize=8)
    grid.suptitle("Sample MRI images by folder label")
    grid.tight_layout()
    grid.savefig(output_dir / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close(grid)

    dimension_counts = Counter(
        f"{row['width']}x{row['height']}x{row['channels']}"
        for row in inventory
        if row["readable"]
    )
    report = {
        "data_dir": data_dir.as_posix(),
        "total_files": len(inventory),
        "readable_images": sum(valid_file_counts.values()),
        "rejected_files": len(rejected),
        "files_per_class": dict(all_file_counts),
        "readable_per_class": dict(valid_file_counts),
        "image_shapes": dict(dimension_counts),
        **duplicate_report,
        "outputs": [
            inventory_path.as_posix(),
            rejected_path.as_posix(),
            (output_dir / "duplicate_audit.json").as_posix(),
            (output_dir / "class_counts.png").as_posix(),
            (output_dir / "sample_images.png").as_posix(),
        ],
        "note": "Inspection only: source images were not deleted, moved, resized, or normalized.",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repository_root / "Data")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "reports")
    parser.add_argument("--samples", type=int, default=5, help="Examples per class in the grid")
    args = parser.parse_args()

    report = inspect_dataset(args.data_dir, args.output_dir, args.samples)
    print(f"Inspected {report['total_files']} files; {report['readable_images']} decoded successfully.")
    print("Readable images per class:")
    for label, count in report["readable_per_class"].items():
        print(f"  {label}: {count}")
    print(
        f"Exact duplicates: {report['duplicate_groups']} groups, "
        f"{report['extra_copies']} extra copies; "
        f"{report['groups_spanning_Tr_Te_prefixes']} groups span Tr/Te prefixes."
    )
    print(f"Reports saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
