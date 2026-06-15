"""Morphological shadow removal and text enhancement."""

import cv2
import numpy as np


def _ensure_odd(kernel_size):
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def create_kernel(kernel_size):
    """Create a rectangular morphology structuring element.

    Args:
        kernel_size: Positive odd kernel size. Small kernels may fail to model
            broad shadows; overly large kernels can remove local text details.

    Returns:
        OpenCV rectangular structuring element.
    """
    kernel_size = _ensure_odd(kernel_size)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))


def remove_shadow(gray, kernel_size=31):
    """Remove uneven illumination with morphological closing.

    Closing estimates slow-changing bright background illumination. Dividing
    the original grayscale image by this background normalizes shadows.

    Args:
        gray: Input grayscale document image.
        kernel_size: Background-estimation structuring element size.

    Returns:
        Shadow-corrected grayscale image.
    """
    kernel = create_kernel(kernel_size)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    corrected = cv2.divide(gray, background, scale=255)
    return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def top_hat_enhance(gray, kernel_size=15):
    """Enhance dark text detail on bright background using top-hat transform.

    Opening removes small bright structures; top-hat extracts the difference
    between the original image and its opening.
    """
    kernel = create_kernel(kernel_size)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def black_hat_enhance(gray, kernel_size=15):
    """Highlight dark text or shadow details using black-hat transform.

    Black-hat computes the difference between a closing result and the original
    image, making dark strokes more visible.
    """
    kernel = create_kernel(kernel_size)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def enhance_document(gray, kernel_size=15):
    """Combine shadow removal and morphology to improve document readability.

    Args:
        gray: Input grayscale document image.
        kernel_size: Morphological kernel size.

    Returns:
        Enhanced grayscale image.
    """
    shadow_removed = remove_shadow(gray, max(31, kernel_size * 2 + 1))
    top_hat = top_hat_enhance(shadow_removed, kernel_size)
    black_hat = black_hat_enhance(shadow_removed, kernel_size)
    enhanced = cv2.add(shadow_removed, top_hat)
    enhanced = cv2.subtract(enhanced, black_hat)
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

