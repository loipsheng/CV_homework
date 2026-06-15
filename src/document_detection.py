"""Document boundary detection using Canny edges and contours."""

import cv2
import numpy as np


def detect_edges(gray_or_blur, low_threshold=50, high_threshold=150):
    """Detect image edges with the Canny operator.

    Args:
        gray_or_blur: Grayscale or blurred grayscale image.
        low_threshold: Lower hysteresis threshold.
        high_threshold: Upper hysteresis threshold.

    Returns:
        Binary edge image.
    """
    return cv2.Canny(gray_or_blur, low_threshold, high_threshold)


def order_points(pts):
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array-like points with shape (4, 2).

    Returns:
        NumPy array with shape (4, 2), dtype float32.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def find_document_contour(edges, image_shape):
    """Locate the document quadrilateral from a Canny edge map.

    Contours are sorted by area. For each contour, approxPolyDP simplifies
    the curve; epsilon controls the maximum distance between the original
    contour and its polygonal approximation. A larger epsilon gives a simpler
    polygon, while a smaller epsilon preserves more contour detail.

    Args:
        edges: Binary edge image.
        image_shape: Shape of the source image.

    Returns:
        Ordered document corners as shape (4, 2).

    Raises:
        ValueError: If no contour can be found.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found for document detection")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = image_shape[0] * image_shape[1]

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.03:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2))

    largest = contours[0]
    if cv2.contourArea(largest) <= 0:
        raise ValueError("Document contour detection failed: largest contour has zero area")
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return order_points(box)


def draw_document_contour(image, corners):
    """Draw detected document border and corner points.

    Args:
        image: Source BGR image.
        corners: Four document corner points.

    Returns:
        Visualization image.
    """
    vis = image.copy()
    pts = order_points(corners).astype(np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
    for idx, (x, y) in enumerate(pts):
        cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(vis, str(idx + 1), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return vis

