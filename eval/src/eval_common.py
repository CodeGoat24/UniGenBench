# -*- coding: utf-8 -*-
"""
Common utilities for UniGenBench evaluation pipeline.

This module provides shared logic for all evaluation backends (Gemini, vLLM)
and languages (English, Chinese), including:
  - Checkpoint explanation dictionaries
  - System prompt templates
  - Response parsing with robust error handling
  - Score calculation
  - Resume / checkpoint support
  - Main evaluation pipeline
"""
import json
import os
import re
import ast
import base64
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
from mimetypes import guess_type


# ============================================================
# Checkpoint Explanation Dictionaries
# ============================================================

EXPLANATION_DICT_EN = {
    "Relationship - Comparison": "Comparison of attributes between two entities",
    "Relationship - Composition": "An entity is composed of one or more other entities",
    "Relationship - Inclusion": "A container contains an entity; the container can also be a plane, e.g., a snake in a painting on a wall",
    "Relationship - Similarity": "Existence of similarities between different entities",

    "Compound - Imagination": "Things that are impossible in real life",
    "Compound - Feature Matching": "Different entities possess different types of attribute features",

    "Attribute - Size": "Assessment of the subject's size, height, length, thickness, width, or tallness/shortness",
    "Attribute - Expression": "Distinguishing expressions from facial actions; expressions must convey a clear emotion",
    "Attribute - Quantity": "Focuses on the challenge of depicting three or more items accurately",
    "Attribute - Material": "Evaluation of different material types and textures",
    "Attribute - Color": "Assessment of different colors",
    "Attribute - Shape": "Assessment of different shapes",

    "Entity Layout - Two-Dimensional Space": "Arrangement and positioning of entities in two-dimensional space",
    "Entity Layout - Three-Dimensional Space": "Arrangement and positioning of entities in three-dimensional space",

    "Action - Full-body (Character/Anthropomorphic)": "Full-body actions by characters or anthropomorphized entities, such as running, diving, breakdancing, swinging, or hanging upside down",
    "Action - Hand (Character/Anthropomorphic)": "Focuses on hand structure—checking if fingers are missing, broken, or distorted",
    "Action - Animal": "Actions performed by animals",
    "Action - Contact Interaction": "Physical interactions between entities",
    "Action - Non-contact Interaction": "For example, two people making eye contact—testing if the model can accurately depict such interactions",
    "Action - State": "A sustained state of an entity, typically expressed with a verb",

    "Grammar - Negation": "Tests the model's understanding of negation grammar",
    "Grammar - Pronoun Reference": "Tests if the model can resolve ambiguous pronoun references correctly",
    "Grammar - Consistency": "Evaluation of shared attributes among entities",

    "World Knowledge": "Covers knowledge of celebrities, architecture, basic domain knowledge, and internet slang. Celebrities with modern copyright risk should be avoided",

    "Style": "Art, painting, photography, design styles, and corresponding artist names",
    "Text Generation": "The text content model needed to accurately generate without any omissions or extra words",

    "Logical Reasoning": "Requires the model to deeply understand the intent and perform reasoning",
}

EXPLANATION_DICT_ZH = {
    "关系-比较关系": "两者的属性对比",
    "关系-构成关系": "一个实体由另一种或几种实体构成",
    "关系-包含关系": "容器对实体的包含关系，容器也可以是平面的，比如：墙上的画里有一只蛇",
    "关系-相似关系": "不同实体中存在的相似关系",

    "复合考点-想象力": "现实生活中不可能发生的事情",
    "复合考点-不同实体特征匹配": "不同实体拥有不同类的属性特征",

    "实体布局-三维空间": "对于三维空间实体的摆放布局",
    "实体布局-二维空间": "对于二维空间实体的摆放布局",

    "属性-大小": "对主体 大小/高低/长短/粗细/宽窄/高矮",
    "属性-表情": "区分表情和脸部动作，脸部动作组成表情，但表情是一定要体现出某种情绪的。",
    "属性-数量": "重点考察三个或三个以上的数字难点",
    "属性-材质": "考察不同材质",
    "属性-颜色": "考察不同颜色",
    "属性-形状": "考察不同形状",

    "动作-人物/拟人全身动作": "人物或拟人全身性的动作，比如奔跑、跳水、跳街舞、荡秋千、倒挂金钩等",
    "动作-人物/拟人手部动作": "针对手部结构的考点，考核手指是否有缺失、崩坏等问题",
    "动作-动物动作": "动物的动作",
    "动作-实体间有接触互动": "各种实体间的有接触互动",
    "动作-实体间无接触互动": "比如两个人对视，考核模型能否把对视关系画对",
    "动作-状态": "实体持续的状态，一般是一个动词。",

    "语法-否定": "考察模型对于否定语法的掌握程度",
    "语法-代词指代": "这里的代词通常是有一些迷惑性的，考察模型能否正确对应",
    "语法-统一性": "实体共同属性的考察",

    "世界知识": "名人、建筑、基础的领域知识、网络流行语。其中名人不要使用当代有版权风险的名人",

    "风格": "艺术、绘画、摄影、设计风格，及对应艺术家名称",

    "逻辑推理": "需要模型深入理解意图并进行一定的推理",

    "文本生成": "考察模型能否准确生成不同语言，字体和长、短文字",
}


# ============================================================
# System Prompt Templates
# ============================================================

SYSTEM_PROMPT_EN_TEMPLATE = '''You are a precise and objective English-language image description system. I will provide you with a prompt for image generation, as well as the corresponding generated image. You will be given a set of evaluation criteria (checkpoints) and their explanations that define the relevance between the prompt and the image. You must evaluate whether the generated image fulfills the requirements implied by each checkpoint in the prompt.

            For each image, follow the steps below in order:

            1. The prompt for the generated image is: 「{prompt}」. You are to analyze the image content in detail from the angles specified in {testpoint}. Detailed definitions of these checkpoints are provided here: {explanation}. The specific description of each checkpoint in the context of the prompt is: {test_explanation}. You must analyze whether the image meets the requirements for each checkpoint individually.

            2. Based on the above analysis, determine whether the generated image satisfies each checkpoint in terms of its visual alignment with the prompt. If the image meets the requirements of a checkpoint, assign a score of 1 to that checkpoint; otherwise, assign a score of 0.

            Constraints:
            - Only describe content that is directly visible; do not interpret, speculate, or infer any background story.
            - Focus solely on visually verifiable details.
            - Omit any uncertain or ambiguous elements.
            - Even if mentioned in the input, do not describe abstract entities, emotions, or speculative ideas.

            Please strictly follow the output format below:

            <description>
                <prompt>{prompt}</prompt>
                <checkpoint>{testpoint}</checkpoint>
                <analysis>A list using square brackets `[]`, where each element is a string of detailed analysis corresponding to one checkpoint, as required in Step 1. **Ensure the list length matches the number of checkpoints**. Each element should be a string representing the analysis for that specific checkpoint.</analysis>
                <score>A list using square brackets `[]`, where each element is a binary score (0 or 1) corresponding to a checkpoint, as required in Step 2. **Ensure the list length matches the number of checkpoints**. Each element should be either 0 or 1, indicating whether the checkpoint was satisfied.</score>
            </description>
            '''

SYSTEM_PROMPT_ZH_TEMPLATE = '''你是一个精确且客观的中文图像描述系统。我会给你一段生成图像的提示词，以及对应的生成图像，同时对于生成图像与提示词之间相关性的考点及对应说明，你需要逐个考点来判断生成的图像是否遵从了提示词中所包含的对应考点要求。

        针对每张图像，你需要按照顺序完成如下的任务：
        1. 这张生成图像对应的提示词为「{prompt}」，你需要根据{testpoint}中的这些角度逐个对图像内容进行更进一步的详细分析，考点的详细说明如下：{explanation}，各个考点在这条prompt中对应的描述说明如下：{test_explanation}, 你需要根据考点逐一判断生成图像是否满足了考点对应的要求
        2. 综合上述回答，你需要逐个考点判断生成的图像在考点关注维度上是否符合输入的prompt，如果满足要求则该考点得分为1，否则为0

        约束条件：
        - 仅描述直接可见的内容；不要进行解读、推测或暗示背景故事。
        - 专注于能够确定性陈述的视觉细节。
        - 省略不确定或不清晰的细节。
        - 即使输入中存在，也不要描述抽象实体、情感或推测。

        请严格遵循以下输出格式：

        <description>
            <prompt>{prompt}</prompt>
            <checkpoint>{testpoint}</checkpoint>
            <analysis>按照步骤1对于给定考点进行逐项详细分析，格式为一个方括号列表，**确保列表的长度与考点的数量相等**，每个元素为一个字符串，表示对于对应考点的分析</analysis>
            <score>按照步骤2逐个对考点进行打分，格式为一个方括号列表，**确保列表的长度与考点的数量相等**，每个元素为0或者1，表示对应考点是否完成</score>
        </description>
        '''


# ============================================================
# Language Configurations
# ============================================================

LANG_CONFIGS = {
    "en": {
        "explanation_dict": EXPLANATION_DICT_EN,
        "system_prompt_template": SYSTEM_PROMPT_EN_TEMPLATE,
        "prompt_column": "prompt_en",
        "subdim_column": "sub_dims_en",
        "testpoint_key": "Testpoints",
        "desc_key": "Testpoint Description",
        "checkpoint_header": "Checkpoints Defination",
        "checkpoint_desc_header": "Checkpoints Description",
    },
    "zh": {
        "explanation_dict": EXPLANATION_DICT_ZH,
        "system_prompt_template": SYSTEM_PROMPT_ZH_TEMPLATE,
        "prompt_column": "prompt_zh",
        "subdim_column": "sub_dims_zh",
        "testpoint_key": "考点",
        "desc_key": "考点对应描述",
        "checkpoint_header": "考点说明",
        "checkpoint_desc_header": "考点描述说明",
    },
}


# ============================================================
# Utility Functions
# ============================================================

def local_image_to_data_url(image_path):
    """Convert a local image file to a base64-encoded data URL."""
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_system_prompt(prompt, testpoint, test_desc, lang):
    """Build the full system prompt for a single evaluation task."""
    config = LANG_CONFIGS[lang]
    explanation_dict = config["explanation_dict"]

    # Build checkpoint explanation
    explanation = f"{config['checkpoint_header']}:\u300c"
    for point in testpoint:
        if point not in explanation_dict:
            raise ValueError(f"Checkpoint '{point}' not found in {lang} explanation dict")
        explanation += f"\n{point}: {explanation_dict[point]}"
    explanation += "\n\u300d"

    # Build checkpoint description
    test_explanation = f"{config['checkpoint_desc_header']}:\u300c"
    for idx, point in enumerate(testpoint):
        test_explanation += f"\n{point}: {test_desc[idx]}"
    test_explanation += "\n\u300d"

    return config["system_prompt_template"].format(
        prompt=prompt,
        testpoint=testpoint,
        explanation=explanation,
        test_explanation=test_explanation,
    )


def parse_evaluation_response(text, testpoint):
    """Parse XML-formatted model response to extract analysis and score.

    Returns:
        (analysis, score) tuple on success, None on failure.
    """
    if text is None:
        return None

    analysis_match = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
    score_match = re.search(r"<score>(.*?)</score>", text, re.DOTALL)

    if analysis_match is None or score_match is None:
        print("Warning: <analysis> or <score> tags not found in response")
        return None

    try:
        analysis = ast.literal_eval(analysis_match.group(1).strip())
        score = ast.literal_eval(score_match.group(1).strip())
    except (ValueError, SyntaxError) as e:
        print(f"Warning: failed to parse response: {e}")
        return None

    if len(testpoint) != len(analysis) or len(testpoint) != len(score):
        print(
            f"Warning: length mismatch - testpoint({len(testpoint)}) "
            f"vs analysis({len(analysis)}) vs score({len(score)})"
        )
        return None

    return analysis, score


def make_result(index, testpoint, prompt, img_path, output_text, result_json=None):
    """Create a standardized evaluation result dictionary."""
    return {
        "index": index,
        "testpoint": testpoint,
        "prompt": prompt,
        "img_path": img_path,
        "output": output_text,
        "result_json": result_json,
    }


# ============================================================
# Backend: Gemini (OpenAI-compatible API)
# ============================================================

def call_evaluation_gemini(task):
    """Evaluate a single image using Gemini API via OpenAI-compatible client."""
    from openai import OpenAI

    index = task["index"]
    prompt = task["prompt"]
    testpoint = task["testpoint"]
    test_desc = task["test_desc"]
    img_path = task["img_path"]
    lang = task["lang"]
    max_retries = task["max_retries"]
    api_key = task["api_key"]
    base_url = task["base_url"]
    model_name = task.get("model_name", "gemini-2.5-pro")

    system_prompt = build_system_prompt(prompt, testpoint, test_desc, lang)
    base64_image = local_image_to_data_url(img_path)
    client = OpenAI(api_key=api_key, base_url=base_url)

    text = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}},
                    ],
                }],
                max_tokens=4096,
            )
            text = response.choices[0].message.content
        except Exception as e:
            print(f"Gemini API error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            text = None

        if text is None:
            continue

        parsed = parse_evaluation_response(text, testpoint)
        if parsed is not None:
            analysis, score = parsed
            result_json = {
                "prompt": prompt,
                "testpoint": testpoint,
                "analysis": analysis,
                "score": score,
            }
            return make_result(index, testpoint, prompt, img_path, text, result_json)

    # All retries exhausted
    return make_result(index, testpoint, prompt, img_path, text)


# ============================================================
# Backend: vLLM (offline model server)
# ============================================================

def call_evaluation_vllm(task):
    """Evaluate a single image using a local vLLM server."""
    from vllm_request import evaluate_batch

    index = task["index"]
    prompt = task["prompt"]
    testpoint = task["testpoint"]
    test_desc = task["test_desc"]
    img_path = task["img_path"]
    lang = task["lang"]
    max_retries = task["max_retries"]
    api_url = task["api_url"]

    system_prompt = build_system_prompt(prompt, testpoint, test_desc, lang)
    payload = [{"images": [img_path], "problem": system_prompt}]

    text = None
    for attempt in range(max_retries + 1):
        result = evaluate_batch(payload, api_url)[0]

        if not result.get("success"):
            print(f"vLLM request failed (attempt {attempt + 1}/{max_retries + 1})")
            continue

        text = result["model_output"]
        parsed = parse_evaluation_response(text, testpoint)
        if parsed is not None:
            analysis, score = parsed
            result_json = {
                "prompt": prompt,
                "testpoint": testpoint,
                "analysis": analysis,
                "score": score,
            }
            return make_result(index, testpoint, prompt, img_path, text, result_json)

    # All retries exhausted
    return make_result(index, testpoint, prompt, img_path, text)


# ============================================================
# Score Calculation
# ============================================================

def calculate_scores(csv_path_or_df, json_path=None):
    """Calculate and print accuracy statistics grouped by dimension.

    Args:
        csv_path_or_df: Path to a result CSV file, or a pandas DataFrame.
        json_path:      If provided, save the score summary to this JSON file.

    Returns:
        dict with "primary_dimensions" and "sub_dimensions" score breakdowns.
    """
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df

    big_class_stats = defaultdict(lambda: [0, 0])
    small_class_stats = defaultdict(lambda: [0, 0])

    for _, row in df.iterrows():
        try:
            checkpoints = ast.literal_eval(row["testpoint"])
            if isinstance(row["result_json"], str):
                scores = ast.literal_eval(row["result_json"])["score"]
            else:
                scores = row["score"]
            if not isinstance(scores, list):
                scores = ast.literal_eval(row["score"])

            for cp, score in zip(checkpoints, scores):
                if "-" in cp:
                    big_class = cp.split("-", 1)[0]
                    small_class = cp
                else:
                    big_class = small_class = cp

                big_class_stats[big_class][1] += 1
                small_class_stats[small_class][1] += 1
                if score == 1:
                    big_class_stats[big_class][0] += 1
                    small_class_stats[small_class][0] += 1
        except Exception as e:
            print(f"Warning: skipping row due to error: {e}")
            continue

    # Build structured result
    result = {"primary_dimensions": {}, "sub_dimensions": {}}

    print("\n\U0001f4d8 Primary Dimension Evaluation Results:")
    for big_class, (correct, total) in big_class_stats.items():
        acc = correct / total if total > 0 else 0
        print(f"  - {big_class}: {correct}/{total} = {acc:.2%}")
        result["primary_dimensions"][big_class] = {
            "correct": correct, "total": total, "accuracy": round(acc, 4),
        }

    print("\n\U0001f4d7 Sub Dimension Evaluation Results:")
    for small_class in sorted(small_class_stats.keys()):
        correct, total = small_class_stats[small_class]
        acc = correct / total if total > 0 else 0
        print(f"  - {small_class}: {correct}/{total} = {acc:.2%}")
        result["sub_dimensions"][small_class] = {
            "correct": correct, "total": total, "accuracy": round(acc, 4),
        }

    # Save to JSON
    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"\nScore summary saved to: {json_path}")

    return result


# ============================================================
# Resume / Checkpoint
# ============================================================

def load_completed_results(jsonl_path):
    """Load previously completed results from a JSONL progress file."""
    completed = {}
    if not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed[row["img_path"]] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def run_evaluation(
    data_path,
    csv_file,
    lang,
    backend,
    api_key=None,
    base_url=None,
    api_url=None,
    model_name=None,
    num_processes=20,
    images_per_prompt=4,
    image_suffix=".png",
    max_retries=10,
    resume=False,
    category=None,
):
    """Run the full evaluation pipeline.

    Args:
        data_path:         Directory containing generated images.
        csv_file:          CSV file with prompts and checkpoints.
        lang:              Language code ('en' or 'zh').
        backend:           Backend type ('gemini' or 'vllm').
        api_key:           API key (gemini backend).
        base_url:          Base URL for API (gemini backend).
        api_url:           vLLM server URL (vllm backend).
        model_name:        Model name for API calls (gemini backend).
        num_processes:     Number of parallel worker processes.
        images_per_prompt: Number of generated images per prompt.
        image_suffix:      Image file extension (e.g. '.png').
        max_retries:       Maximum retry attempts per evaluation.
        resume:            Resume from previous progress file.
        category:          Category label (e.g. 'en', 'en_long', 'zh', 'zh_long')
                           used for output file naming. Falls back to lang if not set.
    """
    if lang not in LANG_CONFIGS:
        raise ValueError(f"Unsupported language '{lang}'. Choose from: {list(LANG_CONFIGS.keys())}")

    lang_config = LANG_CONFIGS[lang]
    file_name = os.path.basename(os.path.normpath(data_path))
    tag = category or lang
    out_file = f"./results/{file_name}_{tag}.csv"
    progress_file = f"{out_file}.progress.jsonl"

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Handle resume vs fresh start
    completed = {}
    if resume and os.path.exists(progress_file):
        completed = load_completed_results(progress_file)
        print(f"Resuming: loaded {len(completed)} completed results")
    else:
        for fpath in [out_file, progress_file]:
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"Removed existing file: {fpath}")

    # Read CSV and prepare tasks
    df = pd.read_csv(csv_file)
    df["index"] = df["index"].apply(lambda x: int(x))

    tasks = []
    for i in tqdm(range(len(df)), desc="Preparing tasks"):
        index = df.iloc[i]["index"]
        prompt = df.iloc[i][lang_config["prompt_column"]]
        subdim_dicts = df.iloc[i][lang_config["subdim_column"]]
        parsed = json.loads(subdim_dicts)
        test_point = parsed[lang_config["testpoint_key"]]
        test_desc = parsed[lang_config["desc_key"]]

        for j in range(images_per_prompt):
            img_path = os.path.join(data_path, f"{index}_{j}{image_suffix}")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")

            if img_path in completed:
                continue

            task = {
                "index": index,
                "prompt": prompt,
                "testpoint": test_point,
                "test_desc": test_desc,
                "img_path": img_path,
                "lang": lang,
                "max_retries": max_retries,
            }
            if backend == "gemini":
                task["api_key"] = api_key
                task["base_url"] = base_url
                task["model_name"] = model_name or "gemini-2.5-pro"
            elif backend == "vllm":
                task["api_url"] = api_url

            tasks.append(task)

    eval_fn = call_evaluation_gemini if backend == "gemini" else call_evaluation_vllm
    all_results = list(completed.values())

    if not tasks:
        print("All tasks already completed.")
    else:
        print(f"Running {len(tasks)} tasks with {num_processes} processes...")
        pool = Pool(processes=min(num_processes, len(tasks)))
        try:
            with open(progress_file, "a", encoding="utf-8") as pf:
                for result in tqdm(pool.imap(eval_fn, tasks), total=len(tasks), desc="Evaluating"):
                    row = {
                        "index": str(int(result["index"])),
                        "prompt": result["prompt"],
                        "testpoint": str(result["testpoint"]),
                        "output": result["output"],
                        "result_json": json.dumps(result["result_json"], ensure_ascii=False, indent=4),
                        "img_path": result["img_path"],
                    }
                    pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    pf.flush()
                    all_results.append(row)
        finally:
            pool.close()
            pool.join()

    # Write final CSV
    pd.DataFrame(all_results).to_csv(out_file, index=False)
    print(f"\nFinished! Evaluation results saved to: {out_file}")

    # Print score summary and save JSON
    json_file = f"./results/{file_name}_{tag}.json"
    calculate_scores(out_file, json_path=json_file)

    # Clean up progress file after successful completion
    if os.path.exists(progress_file):
        os.remove(progress_file)


def add_common_args(parser):
    """Add shared CLI arguments to an argument parser."""
    parser.add_argument("--csv_file", type=str, default="data/test_prompts_en.csv",
                        help="CSV file containing prompts and checkpoints")
    parser.add_argument("--num_processes", type=int, default=20,
                        help="Number of parallel worker processes (default: 20)")
    parser.add_argument("--images_per_prompt", type=int, default=4,
                        help="Number of generated images per prompt (default: 4)")
    parser.add_argument("--image_suffix", type=str, default=".png",
                        help="Image file extension (default: .png)")
    parser.add_argument("--max_retries", type=int, default=10,
                        help="Maximum retry attempts per evaluation (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous progress instead of starting fresh")
    parser.add_argument("--category", type=str, default=None,
                        help="Category label for output naming (e.g. en, en_long, zh, zh_long)")
