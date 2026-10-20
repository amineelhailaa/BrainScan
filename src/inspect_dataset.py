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
