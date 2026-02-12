# -*- coding: utf-8 -*-
"""Evaluate generated images using Gemini API (English)."""
import argparse
from eval_common import run_evaluation, add_common_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniGenBench Evaluation - Gemini (English)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key for Gemini")
    parser.add_argument("--base_url", type=str, required=True,
                        help="Base URL for Gemini-compatible API")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro",
                        help="Model name (default: gemini-2.5-pro)")
    add_common_args(parser)
    args = parser.parse_args()

    run_evaluation(
        data_path=args.data_path,
        csv_file=args.csv_file,
        lang="en",
        backend="gemini",
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        num_processes=args.num_processes,
        images_per_prompt=args.images_per_prompt,
        image_suffix=args.image_suffix,
        max_retries=args.max_retries,
        resume=args.resume,
        category=args.category,
    )
