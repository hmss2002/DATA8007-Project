import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data8007_rl.plotting import plot_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Create comparison figures and result summaries.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    manifest = plot_results(args.log_dir, args.output_dir)
    print(manifest)


if __name__ == "__main__":
    main()
