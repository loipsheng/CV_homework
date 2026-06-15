"""Central configuration for the document scanner pipeline."""

MAX_WIDTH = 1200
GAUSSIAN_KERNEL = 5
CANNY_LOW = 50
CANNY_HIGH = 150
MORPH_KERNEL_SIZE = 15
SHADOW_KERNEL_SIZE = 31
FIXED_THRESHOLD = 127
SAUVOLA_WINDOW_SIZE = 25
SAUVOLA_K = 0.2
SAUVOLA_R = 128

OUTPUT_DIRS = {
    "gray": "01_gray",
    "blur": "02_blur",
    "edges": "03_edges",
    "contours": "04_contours",
    "warped": "05_warped",
    "shadow_removed": "06_shadow_removed",
    "enhanced": "07_enhanced",
    "binary_fixed": "08_binary_fixed",
    "binary_otsu": "09_binary_otsu",
    "binary_sauvola": "10_binary_sauvola",
    "comparisons": "comparisons",
}

