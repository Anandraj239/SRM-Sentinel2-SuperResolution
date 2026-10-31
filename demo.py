"""Open the result images included with this repository."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEMO_IMAGES = (
    ("LR / SR / HR comparison", RESULTS / "comparison_result.png"),
    ("Full pipeline", RESULTS / "final_pipeline_result.png"),
    ("Super-resolution output", RESULTS / "sentinel2_SR_output.png"),
    ("Segmentation overlay", RESULTS / "segmentation" / "sentinel2_overlay.png"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="list available result images without opening them",
    )
    args = parser.parse_args()

    print("SRM project result demo")
    print("These are saved artifacts; this script does not run model inference.")
    available = [(label, path) for label, path in DEMO_IMAGES if path.is_file()]
    missing = [path for _, path in DEMO_IMAGES if not path.is_file()]

    for label, path in available:
        print(f"\n{label}: {path.relative_to(ROOT)}")
        if not args.no_open:
            if not webbrowser.open(path.as_uri()):
                print("  Could not open an image viewer; open the path manually.")
            input("  Press Enter to continue...")

    for path in missing:
        print(f"Missing result image: {path.relative_to(ROOT)}")

    metrics = RESULTS / "metrics_report.txt"
    if metrics.is_file():
        print("\nReported metrics:")
        print(metrics.read_text(encoding="utf-8"))

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
