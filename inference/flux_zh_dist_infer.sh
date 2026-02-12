GPU_NUM=8

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH=black-forest-labs/FLUX.1-dev
MODEL_NAME=FLUX.1-dev
OUTPUT_DIR="${REPO_ROOT}/eval_data/zh/${MODEL_NAME}"
PROMPT_PATH="${REPO_ROOT}/data/test_prompts_zh.csv"

mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node="${GPU_NUM}" --master_port 19000 \
    "${SCRIPT_DIR}/flux_zh_multi_node_inference.py" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_dir "${PROMPT_PATH}" \
    --model_path "${MODEL_PATH}"
