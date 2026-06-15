"""Perspective correction for document quadrilaterals."""

import cv2
import numpy as np

from .document_detection import order_points


def compute_output_size(ordered_pts):
    """Compute output width and height from ordered document corners.

    Args:
        ordered_pts: Points ordered as top-left, top-right, bottom-right,
            bottom-left.

    Returns:
        Tuple (width, height).
    """
    pts = np.asarray(ordered_pts, dtype=np.float32).reshape(4, 2)
    tl, tr, br, bl = pts
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    width = max(1, int(round(max(width_top, width_bottom))))
    height = max(1, int(round(max(height_left, height_right))))
    return width, height


def four_point_transform(image, pts):
    """Warp an arbitrary document quadrilateral into a front-facing rectangle.

    Perspective transformation maps a quadrilateral region in the source image
    to a standard rectangle. It corrects tilt and perspective distortion caused
    by handheld phone capture.

    Args:
        image: Input image.
        pts: Four detected document corners.

    Returns:
        Tuple (warped_image, perspective_matrix).
    """
    if image is None:
        raise ValueError("image must not be None")
    rect = order_points(pts)
    width, height = compute_output_size(rect)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix

