"""Visualization helpers for report-ready comparison figures."""

from pathlib import Path

import cv2
import numpy as np

from .io_utils import save_image


def _to_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def put_title(image, title):
    """Add a compact English title band above an image."""
    bgr = _to_bgr(image)
    height, width = bgr.shape[:2]
    band = np.full((40, width, 3), 255, dtype=np.uint8)
    cv2.putText(band, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([band, bgr])


def _resize_to_height(image, target_height):
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / float(height)
    return cv2.resize(image, (int(round(width * scale)), target_height),
                      interpolation=cv2.INTER_AREA)


def stack_images_horizontally(images, titles):
    """Stack images horizontally with titles."""
    titled = [put_title(img, title) for img, title in zip(images, titles)]
    min_height = min(img.shape[0] for img in titled)
    resized = [_resize_to_height(img, min_height) for img in titled]
    return cv2.hconcat(resized)


def stack_images_grid(images, titles, cols=3):
    """Stack images in a titled grid."""
    titled = [put_title(img, title) for img, title in zip(images, titles)]
    max_height = max(img.shape[0] for img in titled)
    max_width = max(img.shape[1] for img in titled)
    cells = []
    for img in titled:
        canvas = np.full((max_height, max_width, 3), 255, dtype=np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        cells.append(canvas)

    rows = []
    for start in range(0, len(cells), cols):
        row = cells[start:start + cols]
        while len(row) < cols:
            row.append(np.full((max_height, max_width, 3), 255, dtype=np.uint8))
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def save_comparison_figure(images, titles, output_path):
    """Save a horizontal comparison figure."""
    figure = stack_images_horizontally(images, titles)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(output_path, figure)
    return figure

