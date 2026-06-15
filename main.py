"""Batch entry point for the traditional document scanner."""

import argparse
from pathlib import Path

from src import config
from src.evaluation import save_metrics_csv
from src.io_utils import list_images, read_image
from src.pipeline import process_document


def parse_args():
    parser = argparse.ArgumentParser(
        description="Document scanner using traditional OpenCV and NumPy methods."
    )
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_width", type=int, default=config.MAX_WIDTH)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config.MAX_WIDTH = args.max_width

    image_paths = list_images(args.input_dir)
    if not image_paths:
        print(f"No images found in {args.input_dir}. Put jpg/jpeg/png/bmp files there and run again.")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        save_metrics_csv([], Path(args.output_dir) / "metrics.csv")
        return

    metrics = []
    for path in image_paths:
        try:
            image = read_image(path)
            result = process_document(image, path.name, args.output_dir, config, args.debug)
            if result is not None:
                metrics.append(result)
        except Exception as exc:
            print(f"Warning: skip {path.name}, unexpected error: {exc}")

    save_metrics_csv(metrics, Path(args.output_dir) / "metrics.csv")
    print(f"Done. Processed {len(metrics)}/{len(image_paths)} images. Metrics saved to {Path(args.output_dir) / 'metrics.csv'}.")


if __name__ == "__main__":
    main()

