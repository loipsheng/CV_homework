"""Parameter ablation experiments for the document scanner."""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.binarization import postprocess_binary, sauvola_threshold
from src.document_detection import detect_edges
from src.evaluation import calculate_contrast, calculate_sharpness, edge_density, foreground_ratio
from src.io_utils import list_images, read_image, resize_for_processing, save_image
from src.morphology_enhancement import enhance_document, remove_shadow
from src.preprocessing import denoise_gray, to_gray
from src.visualization import save_comparison_figure


def _write_csv(rows, output_csv):
    fields = ["image_name", "experiment", "params", "contrast", "sharpness", "foreground_ratio", "edge_density"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_ablation(input_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(input_dir)[:3]
    rows = []
    if not image_paths:
        print(f"No images found in {input_dir}. Ablation CSV will be empty.")
        _write_csv(rows, output_dir / "ablation_results.csv")
        return

    canny_params = [(30, 100), (50, 150), (80, 200)]
    kernel_sizes = [7, 15, 31]
    sauvola_params = [(w, k) for w in [15, 25, 35] for k in [0.2, 0.3, 0.5]]

    for path in image_paths:
        image = read_image(path)
        resized, _ = resize_for_processing(image, config.MAX_WIDTH)
        gray = to_gray(resized)
        blurred = denoise_gray(gray, config.GAUSSIAN_KERNEL)
        stem = path.stem

        canny_images = []
        canny_titles = []
        for low, high in canny_params:
            edges = detect_edges(blurred, low, high)
            save_image(output_dir / f"{stem}_canny_{low}_{high}.png", edges)
            canny_images.append(edges)
            canny_titles.append(f"Canny {low}/{high}")
            rows.append({
                "image_name": path.name,
                "experiment": "canny",
                "params": f"low={low},high={high}",
                "contrast": "",
                "sharpness": "",
                "foreground_ratio": "",
                "edge_density": float((edges > 0).mean()),
            })
        save_comparison_figure(canny_images, canny_titles, output_dir / f"{stem}_canny_ablation.png")

        kernel_images = []
        kernel_titles = []
        for kernel_size in kernel_sizes:
            shadow_removed = remove_shadow(gray, max(31, kernel_size * 2 + 1))
            enhanced = enhance_document(shadow_removed, kernel_size)
            save_image(output_dir / f"{stem}_kernel_{kernel_size}.png", enhanced)
            kernel_images.append(enhanced)
            kernel_titles.append(f"Kernel {kernel_size}")
            rows.append({
                "image_name": path.name,
                "experiment": "morphology_kernel",
                "params": f"kernel_size={kernel_size}",
                "contrast": calculate_contrast(enhanced),
                "sharpness": calculate_sharpness(enhanced),
                "foreground_ratio": "",
                "edge_density": edge_density(enhanced),
            })
        save_comparison_figure(kernel_images, kernel_titles, output_dir / f"{stem}_kernel_ablation.png")

        enhanced = enhance_document(remove_shadow(gray, config.SHADOW_KERNEL_SIZE), config.MORPH_KERNEL_SIZE)
        sauvola_images = []
        sauvola_titles = []
        for window, k in sauvola_params:
            binary = sauvola_threshold(enhanced, window, k, config.SAUVOLA_R)
            binary = postprocess_binary(binary)
            save_image(output_dir / f"{stem}_sauvola_w{window}_k{k}.png", binary)
            sauvola_images.append(binary)
            sauvola_titles.append(f"W{window} k{k}")
            rows.append({
                "image_name": path.name,
                "experiment": "sauvola",
                "params": f"window_size={window},k={k}",
                "contrast": "",
                "sharpness": "",
                "foreground_ratio": foreground_ratio(binary),
                "edge_density": edge_density(binary),
            })
        save_comparison_figure(sauvola_images[:6], sauvola_titles[:6], output_dir / f"{stem}_sauvola_ablation.png")

    _write_csv(rows, output_dir / "ablation_results.csv")
    print(f"Ablation finished. Results saved to {output_dir}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run parameter ablation experiments.")
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="outputs/ablation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args.input_dir, args.output_dir)

