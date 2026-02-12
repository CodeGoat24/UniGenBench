#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  UniGenBench · Gemini Evaluation Script
#
#  Usage:
#    bash eval_gemini.sh --model <MODEL> --categories <cat1> [cat2 ...]
#
#  Categories (--categories):
#    en        English short prompts  (unigenbench_short.csv)
#    en_long   English long  prompts  (unigenbench_en_long.csv)
#    zh        Chinese short prompts  (unigenbench_short.csv)
#    zh_long   Chinese long  prompts  (unigenbench_zh_long.csv)
#    all       All of the above
#
#  Examples:
#    bash eval_gemini.sh --model FLUX.1-dev --categories en zh
#    bash eval_gemini.sh --model gpt-image-1 --categories all
#    bash eval_gemini.sh --model FLUX.1-dev --categories en_long --resume
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Defaults (edit these or override via CLI) ───────────────────────
MODEL="FLUX.1-dev"
API_KEY="${GEMINI_API_KEY:-}"
BASE_URL="${GEMINI_BASE_URL:-}"
GEMINI_MODEL="gemini-2.5-pro"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
EVAL_DATA_DIR="${SCRIPT_DIR}/../eval_data"

NUM_PROCESSES=20
IMAGES_PER_PROMPT=4
IMAGE_SUFFIX=".png"
MAX_RETRIES=10
RESUME_FLAG=""

CATEGORIES=()
DEFAULT_CATEGORIES=(en zh)

# ─── Argument Parsing ────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash $0 [OPTIONS]

Required:
  --api_key       Gemini API key (or set GEMINI_API_KEY env var)
  --base_url      Gemini API base URL (or set GEMINI_BASE_URL env var)

Options:
  --model         Image generation model name       (default: FLUX.1-dev)
  --gemini_model  Gemini evaluation model name      (default: gemini-2.5-pro)
  --categories    Prompt types to evaluate:
                    en, en_long, zh, zh_long, all   (default: en zh)
  --eval_data_dir Base directory for evaluation data (default: ../eval_data)
  --num_processes Number of parallel workers         (default: 20)
  --images_per_prompt  Images per prompt             (default: 4)
  --image_suffix  Image file extension               (default: .png)
  --max_retries   Max retries per evaluation         (default: 10)
  --resume        Resume from previous progress
  -h, --help      Show this help message

Examples:
  bash $0 --api_key YOUR_KEY --base_url https://api.example.com --model FLUX.1-dev --categories en zh
  bash $0 --api_key YOUR_KEY --base_url https://api.example.com --categories all --resume
  bash $0 --api_key YOUR_KEY --base_url https://api.example.com --categories en_long zh_long --num_processes 32
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)           MODEL="$2";             shift 2 ;;
        --gemini_model)    GEMINI_MODEL="$2";      shift 2 ;;
        --api_key)         API_KEY="$2";           shift 2 ;;
        --base_url)        BASE_URL="$2";          shift 2 ;;
        --eval_data_dir)   EVAL_DATA_DIR="$2";    shift 2 ;;
        --num_processes)   NUM_PROCESSES="$2";     shift 2 ;;
        --images_per_prompt) IMAGES_PER_PROMPT="$2"; shift 2 ;;
        --image_suffix)    IMAGE_SUFFIX="$2";      shift 2 ;;
        --max_retries)     MAX_RETRIES="$2";       shift 2 ;;
        --resume)          RESUME_FLAG="--resume"; shift ;;
        --categories)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CATEGORIES+=("$1"); shift
            done
            ;;
        -h|--help) usage ;;
        *) echo "Error: unknown option '$1' (use -h for help)" >&2; exit 1 ;;
    esac
done

# ─── Validate ────────────────────────────────────────────────────────
if [[ -z "$API_KEY" ]]; then
    echo "Error: --api_key is required (or set GEMINI_API_KEY env var)" >&2; exit 1
fi
if [[ -z "$BASE_URL" ]]; then
    echo "Error: --base_url is required (or set GEMINI_BASE_URL env var)" >&2; exit 1
fi

if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
    CATEGORIES=("${DEFAULT_CATEGORIES[@]}")
    echo "No --categories specified, using default: ${CATEGORIES[*]}"
fi

# Expand "all"
if [[ " ${CATEGORIES[*]} " =~ " all " ]]; then
    CATEGORIES=(en en_long zh zh_long)
fi

ALL_VALID="en en_long zh zh_long"
for cat in "${CATEGORIES[@]}"; do
    if [[ ! " ${ALL_VALID} " =~ " ${cat} " ]]; then
        echo "Error: unknown category '${cat}'. Valid: ${ALL_VALID}, all" >&2; exit 1
    fi
done

# ─── Category → config mapping ──────────────────────────────────────
#   category  →  (lang,  csv_file,  data_subdir)
declare -A CAT_LANG CAT_CSV CAT_DIR

CAT_LANG[en]="en"
CAT_CSV[en]="${DATA_DIR}/test_prompts_en.csv"
CAT_DIR[en]="${EVAL_DATA_DIR}/en/${MODEL}"

CAT_LANG[en_long]="en"
CAT_CSV[en_long]="${DATA_DIR}/test_prompts_en_long.csv"
CAT_DIR[en_long]="${EVAL_DATA_DIR}/en_long/${MODEL}"

CAT_LANG[zh]="zh"
CAT_CSV[zh]="${DATA_DIR}/test_prompts_zh.csv"
CAT_DIR[zh]="${EVAL_DATA_DIR}/zh/${MODEL}"

CAT_LANG[zh_long]="zh"
CAT_CSV[zh_long]="${DATA_DIR}/test_prompts_zh_long.csv"
CAT_DIR[zh_long]="${EVAL_DATA_DIR}/zh_long/${MODEL}"

# ─── Run each category ──────────────────────────────────────────────
for cat in "${CATEGORIES[@]}"; do
    lang="${CAT_LANG[$cat]}"
    csv_file="${CAT_CSV[$cat]}"
    data_path="${CAT_DIR[$cat]}"
    eval_script="${SCRIPT_DIR}/src/gemini_${lang}_eval.py"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Category:      ${cat}"
    echo "  Language:      ${lang}"
    echo "  Model:         ${MODEL}"
    echo "  Gemini Model:  ${GEMINI_MODEL}"
    echo "  Data Path:     ${data_path}"
    echo "  CSV File:      ${csv_file}"
    echo "  Processes:     ${NUM_PROCESSES}"
    echo "  Resume:        ${RESUME_FLAG:-no}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python "${eval_script}" \
        --data_path        "${data_path}" \
        --api_key          "${API_KEY}" \
        --base_url         "${BASE_URL}" \
        --model            "${GEMINI_MODEL}" \
        --csv_file         "${csv_file}" \
        --category         "${cat}" \
        --num_processes    "${NUM_PROCESSES}" \
        --images_per_prompt "${IMAGES_PER_PROMPT}" \
        --image_suffix     "${IMAGE_SUFFIX}" \
        --max_retries      "${MAX_RETRIES}" \
        ${RESUME_FLAG}

    echo ""
    echo "  [Done] ${cat}"
    echo ""
done

echo "All evaluations completed!"
