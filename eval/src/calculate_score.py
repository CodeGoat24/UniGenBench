# -*- coding: utf-8 -*-
"""Calculate evaluation scores from a result CSV file."""
import argparse
from eval_common import calculate_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate UniGenBench evaluation scores")
    parser.add_argument("--result_csv", type=str, default="./results/flux_output.csv",
                        help="Path to the result CSV file")
    parser.add_argument("--json_path", type=str, default=None,
                        help="Optional path to save score summary as JSON")
    args = parser.parse_args()
    calculate_scores(args.result_csv, json_path=args.json_path)
