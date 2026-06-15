"""End-to-end document scanning pipeline."""

from pathlib import Path

import cv2

from . import config as default_config
from .binarization import (
    fixed_threshold,
    otsu_threshold,
    postprocess_binary,
    sauvola_threshold,
)
from .document_detection import detect_edges, draw_document_contour, find_document_contour
from .evaluation import evaluate_image_results
from .io_utils import resize_for_processing, save_image
from .morphology_enhancement import enhance_document, remove_shadow
from .perspective import four_point_transform
from .preprocessing import preprocess, to_gray
from .visualization import save_comparison_figure


def _out(output_dir, key, stem, suffix):
    folder = default_config.OUTPUT_DIRS[key]
    return Path(output_dir) / folder / f"{stem}_{suffix}.png"


def process_document(image, image_name, output_dir, config=default_config, debug=False):
    """Process one document image and save all intermediate results.

    Args:
        image: Input BGR image.
        image_name: File name or display name.
        output_dir: Root output directory.
        config: Configuration module/object.
        debug: Print extra progress messages.

    Returns:
        Metrics dictionary for the processed image, or None if detection fails.
    """
    stem = Path(image_name).stem
    resized, _ = resize_for_processing(image, getattr(config, "MAX_WIDTH", 1200))
    gray, blurred, _ = preprocess(resized, config)

    save_image(_out(output_dir, "gray", stem, "gray"), gray)
    save_image(_out(output_dir, "blur", stem, "blur"), blurred)

    edges = detect_edges(
        blurred,
        getattr(config, "CANNY_LOW", 50),
        getattr(config, "CANNY_HIGH", 150),
    )
    save_image(_out(output_dir, "edges", stem, "edges"), edges)

    try:
        corners = find_document_contour(edges, resized.shape)
        contour_vis = draw_document_contour(resized, corners)
        warped, _ = four_point_transform(resized, corners)
    except Exception as exc:
        print(f"Warning: skip {image_name}, document detection failed: {exc}")
        return None

    save_image(_out(output_dir, "contours", stem, "contour"), contour_vis)
    save_image(_out(output_dir, "warped", stem, "warped"), warped)

    warped_gray = to_gray(warped)
    shadow_removed = remove_shadow(
        warped_gray, getattr(config, "SHADOW_KERNEL_SIZE", 31)
    )
    enhanced = enhance_document(
        shadow_removed, getattr(config, "MORPH_KERNEL_SIZE", 15)
    )
    save_image(_out(output_dir, "shadow_removed", stem, "shadow_removed"), shadow_removed)
    save_image(_out(output_dir, "enhanced", stem, "enhanced"), enhanced)

    fixed = fixed_threshold(enhanced, getattr(config, "FIXED_THRESHOLD", 127))
    otsu, otsu_value = otsu_threshold(enhanced)
    sauvola = sauvola_threshold(
        enhanced,
        getattr(config, "SAUVOLA_WINDOW_SIZE", 25),
        getattr(config, "SAUVOLA_K", 0.2),
        getattr(config, "SAUVOLA_R", 128),
    )
    fixed = postprocess_binary(fixed)
    otsu = postprocess_binary(otsu)
    sauvola = postprocess_binary(sauvola)

    save_image(_out(output_dir, "binary_fixed", stem, "fixed"), fixed)
    save_image(_out(output_dir, "binary_otsu", stem, "otsu"), otsu)
    save_image(_out(output_dir, "binary_sauvola", stem, "sauvola"), sauvola)

    comparison_path = _out(output_dir, "comparisons", stem, "compare")
    save_comparison_figure(
        [resized, contour_vis, warped, enhanced, fixed, otsu, sauvola],
        ["Original", "Contour", "Warped", "Enhanced", "Fixed", f"Otsu {otsu_value:.1f}", "Sauvola"],
        comparison_path,
    )

    if debug:
        print(f"Processed {image_name}: saved {comparison_path}")
    return evaluate_image_results(image_name, enhanced, fixed, otsu, sauvola)

