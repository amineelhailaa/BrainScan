from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_TO_INDEX = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3,
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42


def load_image(path):
    path = Path(path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")

    try:
        image_bytes = np.fromfile(path, dtype=np.uint8)
        image = None
        if image_bytes.size > 0:
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as error:
        raise ValueError(f"Cannot read image {path}: {error}") from error

    if image is None:
        raise ValueError(f"Cannot decode image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    scale = 224/ max(height, width)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    if scale < 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=interpolation
    )

    padded_image = np.zeros((224, 224, 3), dtype=np.uint8)
    top = (224 - new_height) // 2
    left = (224 - new_width) // 2
    padded_image[top:top + new_height, left:left + new_width] = resized_image

    return padded_image














def load_batch(records, root=".", *, seed=SEED, limit=32):
    records = list(records)
    root = Path(root)

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")

    # Shuffle record positions, keeping each image paired with its label.
    random_generator = np.random.default_rng(seed)
    indices = random_generator.permutation(len(records))
    if limit is not None:
        indices = indices[:limit]

    images = []
    labels = []
    paths = []
    failures = []

    for index in indices:
        record = records[index]
        path = record["path"]
        label = CLASS_TO_INDEX[record["label"]]

        try:
            image = load_image(root / path)
        except ValueError as error:
            failures.append({"path": path, "reason": str(error)})
            continue

        # Add all three together only when the image loaded successfully.
        images.append(image)
        labels.append(label)
        paths.append(path)

    if len(images) > 0:
        X = np.stack(images)
    else:
        X = np.empty((0, 224, 224, 3), dtype=np.uint8)
    y = np.array(labels, dtype=np.int64)

    assert len(X) == len(y) == len(paths)
    assert X.shape == (len(y), 224, 224, 3)
    assert np.isfinite(X).all()

    return X, y, paths, failures
