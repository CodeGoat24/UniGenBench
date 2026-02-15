<div align="center">
<img width="724" height="313" alt="" src="https://github.com/user-attachments/assets/50593fdc-ed3d-48b7-8e83-c234b64da2e0" />

</div>
<div align="center">
    <h1 align="center"> UniGenBench++: A Unified Semantic Evaluation Benchmark for Text-to-Image Generation
    </h1>


[UnifiedReward](https://github.com/CodeGoat24/UnifiedReward) Team

<a href="https://arxiv.org/pdf/2508.20751">
<img src='https://img.shields.io/badge/arXiv-UniGenBench-blue' alt='Paper PDF'></a>






<a href="https://arxiv.org/pdf/2510.18701">
<img src='https://img.shields.io/badge/Technical Report-UniGenBench++-blue' alt='Paper PDF'></a>
<br>

<a href="https://codegoat24.github.io/UnifiedReward/Pref-GRPO">
<img src='https://img.shields.io/badge/Website-UniGenBench-orange' alt='Project Page'></a>



<a href="https://codegoat24.github.io/UniGenBench">
<img src='https://img.shields.io/badge/Website-UniGenBench++-orange' alt='Project Page'></a>


[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Eval_Images-yellow)](https://huggingface.co/datasets/CodeGoat24/UniGenBench-Eval-Images)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Offline_Eval_Model-yellow)](https://huggingface.co/CodeGoat24/UniGenBench-EvalModel-qwen-72b-v1) 

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20UniGenBench%20-Leaderboard_(English)-brown)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20UniGenBench%20-Leaderboard_(Chinese)-red)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_Chinese)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20UniGenBench%20-Leaderboard_(English%20Long)-orange)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_English_Long)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20UniGenBench%20-Leaderboard_(Chinese%20Long)-pink)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_Chinese_Long)

</div>


## 🔥 News
😊 We are actively gathering feedback from the community to improve our benchmark. **We welcome your input and encourage you to stay updated through our repository**!!

📝 **To add your own model to the leaderboard**, please send an Email to [Yibin Wang](https://codegoat24.github.io/), then we will help with the evaluation and updating the leaderboard.

Please leave us a star ⭐ if you find our benchmark helpful.

- [2026/02] 🔥🔥  **GPT-4o-1.5**, **Seedream-4.5**, and **FLUX.2-(klein/pro/flex/max)** are added to all 🏅Leaderboard.

- [2025/11] 🔥🔥 **Nano Banana Pro**, **FLUX.2-dev** and **Z-Image** are added to all 🏅Leaderboard.

- [2025/11] 🔥🔥🔥 We release the offline evaluation model [UniGenBench-EvalModel-qwen3vl-32b-v1](https://huggingface.co/CodeGoat24/UniGenBench-EvalModel-qwen3vl-32b-v1).

- [2025/10] 🔥🔥🔥 We release the offline evaluation model [UniGenBench-EvalModel-qwen-72b-v1](https://huggingface.co/CodeGoat24/UniGenBench-EvalModel-qwen-72b-v1), which achieves an average accuracy of 94% compared to evaluations by Gemini 2.5 Pro.
<img width="1121" height="432" alt="image" src="https://github.com/user-attachments/assets/5d5de340-6f31-4fbf-a37d-3181387dce7b" />

- [2025/9] 🔥🔥 **Lumina-DiMOO**, **OmniGen2**, **Infinity**, **X-Omni**, **OneCAT**, **Echo-4o**, and **MMaDA** are added to all 🏅Leaderboard.

- [2025/9] 🔥🔥 **Seedream-4.0**, **Nano Banana**, **GPT-4o**, **Qwen-Image**, **FLUX-Kontext-[Max/Pro]** are added to all 🏅Leaderboard.

- [2025/9] 🔥🔥 We release UniGenBench 🏅[Leaderboard (**Chinese**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_Chinese), 🏅[Leaderboard (**English Long**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_English_Long) and 🏅[Leaderboard (**Chinese Long**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard_Chinese_Long). We will continue to update them regularly. The test prompts are provided in `./data`.
- [2025/9] 🔥🔥 We release **all generated images from the T2I models** evaluated in our UniGenBench on [UniGenBench-Eval-Images](https://huggingface.co/datasets/CodeGoat24/UniGenBench-Eval-Images). Feel free to use any evaluation model that is convenient and suitable for you to assess and compare the performance of your models.
- [2025/8] 🔥🔥 We release [paper](https://arxiv.org/pdf/2508.20751), [project page](https://codegoat24.github.io/UnifiedReward/Pref-GRPO), and UniGenBench 🏅[Leaderboard (**English**)](https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard).




## Introduction

We propose <b>UniGenBench</b>, a unified and versatile benchmark for image generation that integrates diverse prompt themes with a comprehensive suite of fine-grained evaluation criteria. 


<img width="994" height="745" alt="image" src="https://github.com/user-attachments/assets/9b281b2c-d0b0-4c34-8f47-2772a62b7bb9" />


### ✨ Highlights:

- **Comprehensive and Fine-grained Evaluation**: covering 10 **primary dimensions** and 27 **sub-dimensions**, enabling systematic and fine-grained assessment of diverse model capabilities.

- **Rich Prompt Theme Coverage**: organized into 5 **primary themes** and 20 **sub-themes**, comprehensively spanning both realistic and imaginative generation scenarios.

- **Efficient yet Comprehensive**: unlike other benchmarks, UniGenBench requires only **600 prompts**, with each prompt targeting **1–10** specific testpoint, ensuring both coverage and efficiency.

- **Stremlined MLLM Evaluation**: Each testpoint of the prompt is accompanied by a **detailed description**, explaining how the testpoint is reflected in the prompt, assisting MLLM in conducting precise evaluations.

- **Bilingual and Length-variant Prompt Support**: providing both **English** and **Chinese** test prompts in **short** and **long** forms, together with evaluation pipelines for both languages, thus enabling fair and broad cross-lingual benchmarking.

- **Reliable Evaluation Model for Offline Assessment**: To facilitate community use, we train a **robust evaluation model that supports offline assessment** of T2I model outputs.

<img width="1000" height="168" alt="image" src="https://github.com/user-attachments/assets/5ab00a77-7924-42e2-8a32-edaf3eb872cf" />




![](assets/pipeline.jpg)


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

| Category | File | Description |
|----------|------|-------------|
| English Short | `data/test_prompts_en.csv` | 600 short English prompts |
| English Long | `data/test_prompts_en_long.csv` | Long-form English prompts |
| Chinese Short | `data/test_prompts_zh.csv` | 600 short Chinese prompts |
| Chinese Long | `data/test_prompts_zh_long.csv` | Long-form Chinese prompts |
| Training | `data/train_prompt.txt` | Training prompts |


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


## 📂 Expected Image Directory Structure

The evaluation scripts expect generated images organized as follows:

```
eval_data/
  ├── en/
  │   └── FLUX.1-dev/          # --model name
  │       ├── 0_0.png
  │       ├── 0_1.png
  │       ├── ...
  │       └── 599_3.png
  ├── en_long/
  │   └── FLUX.1-dev/
  ├── zh/
  │   └── FLUX.1-dev/
  └── zh_long/
      └── FLUX.1-dev/
```

File naming: `{promptID}_{imageID}.png` (4 images per prompt by default).

You can customize the base directory via `--eval_data_dir`, images per prompt via `--images_per_prompt`, and file extension via `--image_suffix`.


## ✨ Evaluation with Gemini 2.5 Pro

We use **gemini-2.5-pro** (GA, June 17, 2025) via OpenAI-compatible API.

### 1. Evaluation
```bash
# Set API credentials (or pass via --api_key / --base_url)
export GEMINI_API_KEY="sk-xxxxxxx"
export GEMINI_BASE_URL="https://..."

# Evaluate English & Chinese short prompts
bash eval/eval_gemini.sh --model FLUX.1-dev --categories en zh

# Evaluate all categories (en, en_long, zh, zh_long)
bash eval/eval_gemini.sh --model FLUX.1-dev --categories all

# Resume from previous progress
bash eval/eval_gemini.sh --model FLUX.1-dev --categories en --resume
```

Available categories: `en` (English short), `en_long` (English long), `zh` (Chinese short), `zh_long` (Chinese long), `all`.

Run `bash eval/eval_gemini.sh -h` for all options (`--num_processes`, `--images_per_prompt`, etc.).

### 2. Output

After evaluation, for each category:
- Scores across all dimensions are **printed to the console**
- A detailed **CSV results file** is saved: `./results/{model}_{category}.csv`
- A **JSON score summary** is saved: `./results/{model}_{category}.json`

### 3. Re-calculate Scores
```bash
python eval/src/calculate_score.py --result_csv ./results/FLUX.1-dev_en.csv --json_path ./results/FLUX.1-dev_en.json
```


## ✨ Evaluation with UniGenBench-EvalModel

### 1. Deploy vLLM Server

Install dependencies:
```bash
pip install vllm>=0.11.0 qwen-vl-utils==0.0.14
```

Start server:
```bash
# UniGenBench-EvalModel-qwen-72b-v1
vllm serve CodeGoat24/UniGenBench-EvalModel-qwen-72b-v1 \
    --host localhost --port 8080 \
    --served-model-name QwenVL \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt.image 2

# UniGenBench-EvalModel-qwen3vl-32b-v1 (recommended, supports 8 GPUs)
vllm serve CodeGoat24/UniGenBench-EvalModel-qwen3vl-32b-v1 \
    --host localhost --port 8080 \
    --served-model-name QwenVL \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8 \
    --limit-mm-per-prompt.image 2
```

### 2. Evaluation
```bash
# Evaluate English & Chinese short prompts
bash eval/eval_vllm.sh --model FLUX.1-dev --categories en zh

# Evaluate all categories
bash eval/eval_vllm.sh --model FLUX.1-dev --categories all

# Custom server URL and resume
bash eval/eval_vllm.sh --model FLUX.1-dev --categories en_long zh_long \
    --api_url http://gpu-server:8080 --resume
```

Run `bash eval/eval_vllm.sh -h` for all options.

### 3. Output

Same as Gemini evaluation — results are saved to `./results/{model}_{category}.csv` and `./results/{model}_{category}.json`.

### 4. Re-calculate Scores
```bash
python eval/src/calculate_score.py --result_csv ./results/FLUX.1-dev_en.csv --json_path ./results/FLUX.1-dev_en.json
```


## 📧 Contact
If you have any comments or questions, please open a new issue or feel free to contact [Yibin Wang](https://codegoat24.github.io).


## ⭐ Citation
```bibtex
@article{UniGenBench++,
  title={UniGenBench++: A Unified Semantic Evaluation Benchmark for Text-to-Image Generation},
  author={Wang, Yibin and Li, Zhimin and Zang, Yuhang and Bu, Jiazi and Zhou, Yujie and Xin, Yi and He, Junjun and Wang, Chunyu and Lu, Qinglin and Jin, Cheng and others},
  journal={arXiv preprint arXiv:2510.18701},
  year={2025}
}

@article{Pref-GRPO&UniGenBench,
  title={Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning},
  author={Wang, Yibin and Li, Zhimin and Zang, Yuhang and Zhou, Yujie and Bu, Jiazi and Wang, Chunyu and Lu, Qinglin and Jin, Cheng and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2508.20751},
  year={2025}
}
```

## 🏅 Evaluation Leaderboards
<div align="center">

### English Short Prompt Evaluation

<img width="1055" height="662" alt="en_short" src="https://github.com/user-attachments/assets/3f6ce637-aa05-4232-a17d-1852e6e77067" />

### English Long Prompt Evaluation

<img width="1055" height="662" alt="en_long" src="https://github.com/user-attachments/assets/20df3fae-e6be-4546-b75b-c4d52d8ba5c4" />

### Chinese Short Prompt Evaluation

<img width="1055" height="662" alt="zh_short" src="https://github.com/user-attachments/assets/ac29a9eb-b839-4764-bd48-517d523751ca" />

### Chinese Long Prompt Evaluation

<img width="1055" height="662" alt="zh_long" src="https://github.com/user-attachments/assets/e886f3eb-cef2-4a68-af23-f4e9b4059106" />





