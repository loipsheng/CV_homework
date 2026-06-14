"""Generate report-ready figures from pipeline outputs."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import list_images, read_image, save_image
from src.visualization import save_comparison_figure


def _read_if_exists(path):
    if path.exists():
        return read_image(path)
    return None


def main():
    input_dir = ROOT / "data" / "raw"
    output_dir = ROOT / "outputs"
    figures_dir = ROOT / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        print("No raw images found. Run_all created report/figures but no figures were generated.")
        return

    subprocess.run([sys.executable, str(ROOT / "main.py"), "--input_dir", str(input_dir), "--output_dir", str(output_dir)], check=False)
    subprocess.run([sys.executable, str(ROOT / "experiments" / "ablation.py"), "--input_dir", str(input_dir), "--output_dir", str(output_dir / "ablation")], check=False)

    stem = images[0].stem
    original = read_image(images[0])
    pipeline_paths = [
        output_dir / "03_edges" / f"{stem}_edges.png",
        output_dir / "04_contours" / f"{stem}_contour.png",
        output_dir / "05_warped" / f"{stem}_warped.png",
        output_dir / "06_shadow_removed" / f"{stem}_shadow_removed.png",
        output_dir / "10_binary_sauvola" / f"{stem}_sauvola.png",
    ]
    pipeline_images = [original] + [img for img in (_read_if_exists(p) for p in pipeline_paths) if img is not None]
    if len(pipeline_images) >= 2:
        save_comparison_figure(
            pipeline_images,
            ["Original", "Edges", "Corners", "Warped", "Shadow Removed", "Final Binary"][:len(pipeline_images)],
            figures_dir / "pipeline_example.png",
        )

    binary_images = []
    binary_titles = []
    for folder, suffix, title in [
        ("08_binary_fixed", "fixed", "Fixed"),
        ("09_binary_otsu", "otsu", "Otsu"),
        ("10_binary_sauvola", "sauvola", "Sauvola"),
    ]:
        img = _read_if_exists(output_dir / folder / f"{stem}_{suffix}.png")
        if img is not None:
            binary_images.append(img)
            binary_titles.append(title)
    if binary_images:
        save_comparison_figure(binary_images, binary_titles, figures_dir / "binarization_comparison.png")

    morph_images = []
    morph_titles = []
    for folder, suffix, title in [
        ("01_gray", "gray", "Gray"),
        ("06_shadow_removed", "shadow_removed", "Shadow Removed"),
        ("07_enhanced", "enhanced", "Enhanced"),
    ]:
        img = _read_if_exists(output_dir / folder / f"{stem}_{suffix}.png")
        if img is not None:
            morph_images.append(img)
            morph_titles.append(title)
    if morph_images:
        save_comparison_figure(morph_images, morph_titles, figures_dir / "morphology_comparison.png")

    for source, target in [
        (output_dir / "ablation" / f"{stem}_canny_ablation.png", figures_dir / "canny_ablation.png"),
        (output_dir / "ablation" / f"{stem}_sauvola_ablation.png", figures_dir / "sauvola_ablation.png"),
    ]:
        img = _read_if_exists(source)
        if img is not None:
            save_image(target, img)

    print(f"Report figures saved to {figures_dir}.")


if __name__ == "__main__":
    main()

