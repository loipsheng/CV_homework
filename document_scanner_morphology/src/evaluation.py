"""Unsupervised quality metrics for scanned document results."""

import csv
from pathlib import Path

import cv2
import numpy as np


def calculate_contrast(gray):
    """Return grayscale standard deviation as a contrast metric."""
    return float(np.std(gray))


def calculate_sharpness(gray):
    """Return Laplacian variance as a sharpness metric."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def foreground_ratio(binary):
    """Return ratio of black foreground pixels in a binary document image."""
    return float(np.mean(binary < 128))


def edge_density(gray):
    """Return ratio of Canny edge pixels."""
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def evaluate_image_results(image_name, enhanced, fixed, otsu, sauvola):
    """Calculate metrics for enhanced image and three binary outputs."""
    return {
        "image_name": image_name,
        "enhanced_contrast": calculate_contrast(enhanced),
        "enhanced_sharpness": calculate_sharpness(enhanced),
        "fixed_foreground_ratio": foreground_ratio(fixed),
        "otsu_foreground_ratio": foreground_ratio(otsu),
        "sauvola_foreground_ratio": foreground_ratio(sauvola),
        "fixed_edge_density": edge_density(fixed),
        "otsu_edge_density": edge_density(otsu),
        "sauvola_edge_density": edge_density(sauvola),
    }


def save_metrics_csv(metrics_list, output_csv):
    """Save metric dictionaries to a CSV file."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_name",
        "enhanced_contrast",
        "enhanced_sharpness",
        "fixed_foreground_ratio",
        "otsu_foreground_ratio",
        "sauvola_foreground_ratio",
        "fixed_edge_density",
        "otsu_edge_density",
        "sauvola_edge_density",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in metrics_list:
            writer.writerow({key: row.get(key, "") for key in fields})

