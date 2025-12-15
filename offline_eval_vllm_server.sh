# UniGenBench-EvalModel-qwen-72b-v1

# vllm serve CodeGoat24/UniGenBench-EvalModel-qwen-72b-v1 \
#     --host localhost \
#     --trust-remote-code \
#     --served-model-name QwenVL \
#     --gpu-memory-utilization 0.9 \
#     --tensor-parallel-size 4 \
#     --pipeline-parallel-size 1 \
#     --limit-mm-per-prompt.image 2 \
#     --port 8080


# UniGenBench-EvalModel-qwen3vl-32b-v1 (recommended, support deploying on 8 gpus)

vllm serve CodeGoat24/UniGenBench-EvalModel-qwen3vl-32b-v1 \
    --host localhost \
    --trust-remote-code \
    --served-model-name QwenVL \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8 \ 
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt.image 2 \
    --port 8080