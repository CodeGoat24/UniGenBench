# -*- coding: utf-8 -*-
import json
import os
import pandas as pd
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from tqdm import tqdm
from multiprocessing import Pool
import ast
import re
from collections import defaultdict
import base64
from mimetypes import guess_type
from vllm_request import evaluate_batch
import argparse

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def call_evaluation(args):
    index, prompt, testpoint, test_desc, img_path, api_url = args

    explanation_dict = {
        '关系-比较关系':'两者的属性对比',
        '关系-构成关系':'一个实体由另一种或几种实体构成',
        '关系-包含关系':'容器对实体的包含关系，容器也可以是平面的，比如：墙上的画里有一只蛇',
        '关系-相似关系':'不同实体中存在的相似关系',
        '复合考点-想象力':'现实生活中不可能发生的事情',
        '复合考点-不同实体特征匹配':'不同实体拥有不同类的属性特征',
        '实体布局-三维空间':'对于三维空间实体的摆放布局', 
        '实体布局-二维空间':'对于二维空间实体的摆放布局', 
        '属性-大小':'对主体 大小/高低/长短/粗细/宽窄/高矮',
        '属性-表情':'区分表情和脸部动作，脸部动作组成表情，但表情是一定要体现出某种情绪的。',
        '属性-数量':'重点考察三个或三个以上的数字难点',
        '属性-材质':'考察不同材质',
        '动作-人物/拟人全身动作':'人物或拟人全身性的动作，比如奔跑、跳水、跳街舞、荡秋千、倒挂金钩等',
        '动作-人物/拟人手部动作':'针对手部结构的考点，考核手指是否有缺失、崩坏等问题',
        '动作-动物动作':'动物的动作',
        '动作-实体间有接触互动':'各种实体间的有接触互动',
        '动作-实体间无接触互动':'比如两个人对视，考核模型能否把对视关系画对',
        '动作-状态':'实体持续的状态，一般是一个动词。',
        '语法-否定':'考察模型对于否定语法的掌握程度',
        '语法-代词指代':'这里的代词通常是有一些迷惑性的，考察模型能否正确对应',
        '语法-统一性':'实体共同属性的考察',
        '世界知识':'名人、建筑、基础的领域知识、网络流行语。其中名人不要使用当代有版权风险的名人',
        '风格':'艺术、绘画、摄影、设计风格，及对应艺术家名称',
        '逻辑推理': '需要模型深入理解意图并进行一定的推理',
        '文本生成':'考察模型能否准确生成不同语言，字体和长、短文字',
        }

    explanation = "考点说明：「「"

    for point in testpoint:
        if point not in explanation_dict:
            print(f'{point} do not exist!')
            raise()

        explanation += f"\n{point}: {explanation_dict[point]}"
    explanation += "\n」"

    test_explanation = "考点描述说明：「"
    for idx, point in enumerate(testpoint):
        test_explanation += f"\n{point}: {test_desc[idx]}"
    test_explanation += "\n」"

    retry_num = 0
    while True:
        system_prompt = f'''你是一个精确且客观的中文图像描述系统。我会给你一段生成图像的提示词，以及对应的生成图像，同时对于生成图像与提示词之间相关性的考点及对应说明，你需要逐个考点来判断生成的图像是否遵从了提示词中所包含的对应考点要求。

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
        


        payload = [
            {
                "images": [
                    img_path
                ],
                "problem": system_prompt,
            },
        ]


        result = evaluate_batch(payload, api_url)[0]
        
        if result['success']:        
            text = result['model_output']
            print(text)
        else:
            print('fail to request vLLM server!')
            continue


        if text is not None:
            try:
                analysis_match = re.search(r'<analysis>(.*?)</analysis>', text, re.DOTALL)
                score_match = re.search(r'<score>(.*?)</score>', text, re.DOTALL)

                analysis_str = analysis_match.group(1).strip()
                analysis = ast.literal_eval(analysis_str)
                
                score_str = score_match.group(1).strip()
                score = ast.literal_eval(score_str)

                if len(testpoint) != len(analysis) or len(testpoint) != len(score):
                    if retry_num < 10:
                        retry_num += 1
                        continue
                    else:
                        return dict(
                            index = index,
                            testpoint = testpoint,
                            prompt = prompt,
                            img_path=img_path,
                            output = text,
                            result_json = None,
                        )
            except Exception as e:
                print(e)
                if retry_num < 10:
                    retry_num += 1
                    continue
                else:
                    return dict(
                        index = index,
                        testpoint = testpoint,
                        prompt = prompt,
                        img_path=img_path,
                        output = text,
                        result_json = None,
                    )

            result_json = {
                'prompt': prompt,
                'testpoint': testpoint,
                'analysis': analysis,
                'score': score,
            }


            return dict(
                index = index,
                testpoint = testpoint,
                prompt = prompt,
                img_path=img_path,
                output = text,
                result_json = result_json,
            )
        else:
            print("None")
            if retry_num < 10:
                retry_num += 1
                continue
            else:
                return dict(
                    index = index,
                    testpoint = testpoint,
                    prompt = prompt,
                    img_path=img_path,
                    output = text,
                    result_json = None,
                )


def main(data_path: str, api_url: str, csv_file: str):

    file_name = data_path.split('/')[-1]


    out_file = f'./results/{file_name}_zh.csv'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if os.path.exists(out_file):
        os.remove(out_file) 
        print(f"remove existing file: {out_file}")
    suffix = '.png'

    df = pd.read_csv(csv_file)
    df['index'] = df['index'].apply(lambda x: int(x))

    args = []
    for i in tqdm(range(len(df)), total=len(df)):
        index = df.iloc[i]['index']
        
        prompt = df.iloc[i]['prompt_zh']
        subdim_dicts = df.iloc[i]['sub_dims_zh']

        test_point = json.loads(subdim_dicts)['考点']
        test_desc = json.loads(subdim_dicts)['考点对应描述']

        for j in range(4):
            img_path = os.path.join(data_path, f"{index}_{j}{suffix}")

            if not os.path.exists(img_path):
                raise()
                    
            args.append((index, prompt, test_point, test_desc, img_path, api_url))


    pool = Pool(processes=20)
    try:
        for result in tqdm(pool.imap(call_evaluation, args), total=len(args)):
  
            new_row = pd.DataFrame([{
                'index': str(int(result['index'])),
                'prompt': result['prompt'],
                'testpoint': str(result['testpoint']),
                'output': result['output'],
                'result_json': json.dumps(result['result_json'], ensure_ascii=False, indent=4),
                'img_path': result['img_path']
            }])

            if not os.path.exists(out_file):
                new_row.to_csv(out_file, index=False)
            else:
                existing_df = pd.read_csv(out_file)
                updated_df = pd.concat([existing_df[existing_df['index'] != str(result['index'])], new_row])
                updated_df.to_csv(out_file, index=False)
            
    finally:
        pool.close()
        pool.join()

    print(f"Finished! Evaluation results are saved to: {out_file}")

    # Calculate scores
    df = pd.read_csv(out_file)

    big_class_stats = defaultdict(lambda: [0, 0])  
    small_class_stats = defaultdict(lambda: [0, 0]) 

    for _, row in df.iterrows():
        checkpoints = ast.literal_eval(row['testpoint'])
        try:
            scores = ast.literal_eval(row['result_json'])['score'] if isinstance(row['result_json'], str) else row['score']
    
            if not isinstance(scores, list):
                scores = ast.literal_eval(row['score'])
    
            for cp, score in zip(checkpoints, scores):
    
                if '-' in cp:
                    big_class, small_class = cp.split('-', 1)[0], cp
                else:
                    big_class = small_class = cp
    
                big_class_stats[big_class][1] += 1
                small_class_stats[small_class][1] += 1
                if score == 1:
                    big_class_stats[big_class][0] += 1
                    small_class_stats[small_class][0] += 1
        except:
            continue

    print("📘 Primary Dimension Evaluation Results:")
    for big_class, (correct, total) in big_class_stats.items():
        acc = correct / total if total > 0 else 0
        print(f"  - {big_class}: {correct}/{total} = {acc:.2%}")

    print("\n📗 Sub Dimension Evaluation Results:")
    for small_class in sorted(small_class_stats.keys()):
        correct, total = small_class_stats[small_class]
        acc = correct / total if total > 0 else 0
        print(f"  - {small_class}: {correct}/{total} = {acc:.2%}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation with Qwen2.5-VL-72b")
    parser.add_argument("--data_path", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--api_url", type=str, required=True, help="vLLM request url")
    parser.add_argument("--csv_file", type=str, default="data/test_prompts_en.csv", help="CSV file containing prompts")

    args = parser.parse_args()
    main(args.data_path, args.api_url, args.csv_file)
