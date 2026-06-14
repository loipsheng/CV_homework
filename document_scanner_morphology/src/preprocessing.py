"""Preprocessing operations for document images."""

import cv2


def _ensure_odd(kernel_size):
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def to_gray(image):
    """Convert a BGR image to grayscale.

    Args:
        image: Input BGR image.

    Returns:
        Single-channel grayscale image.
    """
    if image is None:
        raise ValueError("image must not be None")
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_gray(gray, kernel_size=5):
    """Denoise a grayscale image with Gaussian blur.

    Gaussian filtering suppresses small noise before Canny edge detection,
    which helps avoid false edges caused by sensor noise or paper texture.

    Args:
        gray: Single-channel grayscale image.
        kernel_size: Odd Gaussian kernel size.

    Returns:
        Blurred grayscale image.
    """
    kernel_size = _ensure_odd(kernel_size)
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def equalize_hist(gray):
    """Enhance low-contrast grayscale images with histogram equalization.

    Args:
        gray: Single-channel grayscale image.

    Returns:
        Contrast-enhanced grayscale image.
    """
    return cv2.equalizeHist(gray)


def preprocess(image, config):
    """Run grayscale conversion, denoising, and histogram equalization.

    Args:
        image: Input BGR image.
        config: Module or object that provides GAUSSIAN_KERNEL.

    Returns:
        Tuple (gray, blurred, equalized).
    """
    gray = to_gray(image)
    blurred = denoise_gray(gray, getattr(config, "GAUSSIAN_KERNEL", 5))
    equalized = equalize_hist(blurred)
    return gray, blurred, equalized

