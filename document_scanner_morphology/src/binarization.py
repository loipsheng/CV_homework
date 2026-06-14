"""Document binarization methods implemented with OpenCV and NumPy."""

import cv2
import numpy as np


def fixed_threshold(gray, threshold=127):
    """Apply fixed global thresholding with black text on white background."""
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def otsu_threshold(gray):
    """Apply Otsu adaptive global thresholding.

    Returns:
        Tuple (binary_image, threshold_value).
    """
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, threshold


def sauvola_threshold(gray, window_size=25, k=0.2, R=128):
    """Apply Sauvola local adaptive thresholding.

    Formula:
        T(x, y) = m(x, y) * [1 + k * (s(x, y) / R - 1)]

    m and s are local mean and standard deviation computed by cv2.boxFilter.
    Pixels darker than T are treated as foreground text (black).
    """
    if window_size % 2 == 0:
        window_size += 1
    gray_float = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_float, ddepth=-1, ksize=(window_size, window_size),
                         normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(gray_float * gray_float, ddepth=-1,
                            ksize=(window_size, window_size), normalize=True,
                            borderType=cv2.BORDER_REPLICATE)
    variance = np.maximum(mean_sq - mean * mean, 0)
    std = np.sqrt(variance)
    threshold = mean * (1 + k * (std / R - 1))
    return np.where(gray_float > threshold, 255, 0).astype(np.uint8)


def postprocess_binary(binary, kernel_size=3):
    """Clean binary image with opening and closing.

    Opening removes small isolated noise; closing reconnects broken strokes.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def compare_binarization(gray):
    """Return fixed, Otsu, and Sauvola binarization results."""
    fixed = fixed_threshold(gray)
    otsu, _ = otsu_threshold(gray)
    sauvola = sauvola_threshold(gray)
    return fixed, otsu, sauvola

