"""Input and output helpers for batch document image processing."""

from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(input_dir):
    """Return sorted image paths in an input directory.

    Args:
        input_dir: Directory that contains jpg, jpeg, png, or bmp images.

    Returns:
        A sorted list of pathlib.Path objects.
    """
    folder = Path(input_dir)
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path):
    """Read an image from disk with OpenCV.

    Args:
        path: Image file path.

    Returns:
        BGR image as a NumPy array.

    Raises:
        ValueError: If OpenCV cannot decode the image.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def save_image(path, image):
    """Save an image and create parent directories automatically.

    Args:
        path: Output file path.
        image: NumPy image array.

    Raises:
        ValueError: If OpenCV fails to write the file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise ValueError(f"Failed to save image: {output_path}")


def resize_for_processing(image, max_width=1200):
    """Resize a large image while preserving aspect ratio.

    Args:
        image: Input BGR or grayscale image.
        max_width: Maximum width used by the processing pipeline.

    Returns:
        Tuple of (resized_image, scale), where scale maps resized coordinates
        back to the original image. If no resize is needed, scale is 1.0.
    """
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy(), 1.0

    scale = max_width / float(width)
    new_size = (max_width, int(round(height * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale

