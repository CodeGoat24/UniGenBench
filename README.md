<div align="center">
    <h1 align="center"> UniGenBench: Unified Text-to-Image Generation Benchmark
    </h1>


Hunyuan, Tencent & [UnifiedReward](https://github.com/CodeGoat24/UnifiedReward) Team

<a href="https://arxiv.org/pdf/2508.20751">
<img src='https://img.shields.io/badge/arXiv-UniGenBench-blue' alt='Paper PDF'></a>

<a href="https://codegoat24.github.io/UnifiedReward/Pref-GRPO">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/CodeGoat24/FLUX.1-dev-PrefGRPO) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-yellow)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard)



</div>


## 🔥 News
😊 We are actively gathering feedback from the community to improve our benchmark. **We welcome your input and encourage you to stay updated through our repository**!!

📝 **To add your own model to the leaderboard**, please send an Email to [Yibin Wang](https://codegoat24.github.io/), then we will help with the evaluation and updating the leaderboard.

Please leave us a star ⭐ if you find our benchmark helpful.
- [2025/9] 🔥🔥 We release **all generated images from the T2I models** evaluated in our UniGenBench on [UniGenBench-Eval-Images](https://huggingface.co/datasets/CodeGoat24/UniGenBench-Eval-Images). Feel free to use any evaluation model that is convenient and suitable for you to assess and compare the performance of your models.
- [2025/9] 🔥🔥 **GPT-4o**, **Imagen-4-Ultra**, **Nano Banana**, **Seedream-3.0**, **Qwen-Image**, **FLUX-Kontext-[Max/Pro]** are added to the 🏅[Leaderboard](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard).
- [2025/8] 🔥🔥 We release [paper](https://arxiv.org/pdf/2508.20751), [project page](https://codegoat24.github.io/UnifiedReward/Pref-GRPO), and UniGenBench [Leaderboard](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard).

## Introduction

We propose <b>UniGenBench</b>, a unified and versatile benchmark for image generation that integrates diverse prompt themes with a comprehensive suite of fine-grained evaluation criteria. 



![](assets/teaser.png)
### ✨ Highlights:

- **Comprehensive and fine-grained evaluation**: covering 10 **primary** dimensions and 27 **sub-dimensions**, enabling systematic and fine-grained assessment of diverse model capabilities.

- **Rich prompt theme coverage**: organized into 5 **primary** themes and 20 **sub-themes**, comprehensively spanning both realistic and imaginative generation scenarios.

- **Efficient yet comprehensive**: unlike other benchmarks, UniGenBench requires only **600 prompts**, with each prompt targeting **1–5 specific testpoint**, ensuring both coverage and efficiency.

- **Reliable MLLM Evaluation**: Each testpoint of the prompt is accompanied by a **detailed description**, explaining how the testpoint is reflected in the prompt, assisting MLLM in conducting precise evaluations.

- **Bilingual support**: providing test prompts in both **English** and **Chinese**, together with evaluation pipelines for both languages, thus enabling fair and broad cross-lingual benchmarking.

![](assets/comparison.png)

![](assets/pipeline.jpg)


![unigenbench_leadboard](assets/leaderboard.png)

## 📑 Prompt Introduction
Each prompt in our benchmark is recorded as a row in a `.csv` file, combining with structured annotations for evaluation.  

- **index** 
- **prompt**: The full English prompt to be tested  
- **sub_dims**: A JSON-encoded field that organizes rich metadata, including:  
  - **Primary / Secondary Categories** – prompt theme (e.g., *Creative Divergence → Imaginative Thinking*)  
  - **Subjects** – the main entities involved in the prompt (e.g., *Animal*)  
  - **Sentence Structure** – the linguistic form of the prompt (e.g., *Descriptive*)  
  - **Testpoints** – key aspects to evaluate (e.g., *Style*, *World Knowledge*, *Attribute - Quantity*)  
  - **Testpoint Description** – evaluation cues extracted from the prompt (e.g., *classical ink painting*, *Egyptian pyramids*, *two pandas*)  

- **English** Test set: `data/test_prompts_en.csv`
- **Chinese** Test set: `data/test_prompts_zh.csv`
- **Training** set: `train_prompt.txt`


## 🚀 Inference
We provide reference code for **multi-node inference** based on *FLUX.1-dev*.  
```bash
# English Prompt
bash inference/flux_en_dist_infer.sh

# Chinese Prompt
bash inference/flux_zh_dist_infer.sh
``` 
For each test prompt, **4 images** are generated and stored in the following folder structure:

```
output_directory/
  ├── 0_0.png
  ├── 0_1.png
  ├── 0_2.png
  ├── 0_3.png
  ├── 1_0.png
  ├── 1_1.png
  ...
```
 
The file naming follows the pattern `promptID_imageID.png`


## ✨ Evaluation with Gemini2.5-pro
We are using the API version:

> **gemini-2.5-pro**:
>
> Release stage: General Availability (GA)
>
> Release date: June 17, 2025
>
### 1. Evaluation
```
#!/bin/bash

# API
API_KEY="sk-xxxxxxx"
BASE_URL=""

DATA_PATH="flux_output"  # Directory of generated images
CSV_FILE="data/test_prompts_en.csv" # English test prompt file

# English Evaluation
python eval/gemini_en_eval.py \
  --data_path "$DATA_PATH" \
  --api_key "$API_KEY" \
  --base_url "$BASE_URL" \
  --csv_file "$CSV_FILE"

# Chinese Evaluation
CSV_FILE="data/test_prompts_zh.csv" # Chinese test prompt file

python eval/gemini_zh_eval.py \
  --data_path "$DATA_PATH" \
  --api_key "$API_KEY" \
  --base_url "$BASE_URL" \
  --csv_file "$CSV_FILE"
```

- After evaluation, scores across all dimensions will be **printed to the console**.  
- A detailed `.csv` **results file** will also be saved in the `./results` directory.  
 

### 2. Calculate Score

You can also load the results file to re-print or further analyze the scores. 
```
python eval/calculate_score.py
```


## ✨ Evaluation with Qwen2.5-VL-72b
### 1. Deploy vLLM server

1. Install vLLM
```bash
pip install vllm==0.9.0.1 transformers==4.52.4
```
2. Start server
```bash
echo ${LOCAL_IP}

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --host ${LOCAL_IP} \
    --trust-remote-code \
    --served-model-name QwenVL \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt image=2 \
    --port 8080 
```

### 2. Evaluation
```
#!/bin/bash

# vLLM request url
API_URL=http://${LOCAL_IP}:8080

DATA_PATH="flux_output"  # Directory of generated images
CSV_FILE="data/test_prompts_en.csv" # English test prompt file

# English Evaluation
python eval/qwenvl_72b_en_eval.py \
  --data_path "$DATA_PATH" \
  --api_url "$API_URL" \
  --csv_file "$CSV_FILE"

# Chinese Evaluation
CSV_FILE="data/test_prompts_zh.csv" # Chinese test prompt file

python eval/qwenvl_72b_zh_eval.py \
  --data_path "$DATA_PATH" \
  --api_url "$API_URL" \
  --csv_file "$CSV_FILE"
```

- After evaluation, scores across all dimensions will be **printed to the console**.  
- A detailed `.csv` **results file** will also be saved in the `./results` directory.  
 

### 3. Calculate Score

You can also load the results file to re-print or further analyze the scores. 
```
python eval/calculate_score.py
```


## 📧 Contact
If you have any comments or questions, please open a new issue or feel free to contact [Yibin Wang](https://codegoat24.github.io).


## ⭐ Citation
```bibtex
@article{Pref-GRPO&UniGenBench,
  title={Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning},
  author={Wang, Yibin and Li, Zhimin and Zang, Yuhang and Zhou, Yujie and Bu, Jiazi and Wang, Chunyu and Lu, Qinglin, and Jin, Cheng and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2508.20751},
  year={2025}
}
```
