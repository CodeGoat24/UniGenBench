# -*- coding: utf-8 -*-
"""Evaluate generated images using offline vLLM model (Chinese)."""
import argparse
from eval_common import run_evaluation, add_common_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniGenBench Evaluation - Offline Model (Chinese)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--api_url", type=str, required=True,
                        help="vLLM server URL")
    add_common_args(parser)
    args = parser.parse_args()

    run_evaluation(
        data_path=args.data_path,
        csv_file=args.csv_file,
        lang="zh",
        backend="vllm",
        api_url=args.api_url,
        num_processes=args.num_processes,
        images_per_prompt=args.images_per_prompt,
        image_suffix=args.image_suffix,
        max_retries=args.max_retries,
        resume=args.resume,
        category=args.category,
    )
