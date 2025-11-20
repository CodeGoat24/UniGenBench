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
        'Attribute - Color': 'Assessment of different colors',
        'Attribute - Shape': 'Assessment of different shapes',
        
        'Entity Layout - Two-Dimensional Space': 'Arrangement and positioning of entities in two-dimensional space',
        'Entity Layout - Three-Dimensional Space': 'Arrangement and positioning of entities in three-dimensional space',

        "Action - Full-body (Character/Anthropomorphic)": "Full-body actions by characters or anthropomorphized entities, such as running, diving, breakdancing, swinging, or hanging upside down",
        "Action - Hand (Character/Anthropomorphic)": "Focuses on hand structure—checking if fingers are missing, broken, or distorted",
        "Action - Animal": "Actions performed by animals",
        "Action - Contact Interaction": "Physical interactions between entities",
        "Action - Non-contact Interaction": "For example, two people making eye contact—testing if the model can accurately depict such interactions",
        "Action - State": "A sustained state of an entity, typically expressed with a verb",
        
        "Grammar - Negation": "Tests the model’s understanding of negation grammar",
        "Grammar - Pronoun Reference": "Tests if the model can resolve ambiguous pronoun references correctly",
        "Grammar - Consistency": "Evaluation of shared attributes among entities",
        
        "World Knowledge": "Covers knowledge of celebrities, architecture, basic domain knowledge, and internet slang. Celebrities with modern copyright risk should be avoided",
        
        "Style": "Art, painting, photography, design styles, and corresponding artist names",
        'Text Generation': 'The text content model needed to accurately generate without any omissions or extra words',
        
        "Logical Reasoning": "Requires the model to deeply understand the intent and perform reasoning",
    }

    explanation = "Checkpoints Defination:「"

    for point in testpoint:
        if point not in explanation_dict:
            print(f'{point} do not exist!')
            raise()

        explanation += f"\n{point}: {explanation_dict[point]}"
    explanation += "\n」"

    test_explanation = "Checkpoints Description:「"
    for idx, point in enumerate(testpoint):
        test_explanation += f"\n{point}: {test_desc[idx]}"
    test_explanation += "\n」"

    retry_num = 0
    while True:
        system_prompt = f'''You are a precise and objective English-language image description system. I will provide you with a prompt for image generation, as well as the corresponding generated image. You will be given a set of evaluation criteria (checkpoints) and their explanations that define the relevance between the prompt and the image. You must evaluate whether the generated image fulfills the requirements implied by each checkpoint in the prompt.

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


        if text is not None :
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


    out_file = f'./results/{file_name}_en.csv'
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
        
        prompt = df.iloc[i]['prompt_en']
        subdim_dicts = df.iloc[i]['sub_dims_en']

        test_point = json.loads(subdim_dicts)['Testpoints']
        test_desc = json.loads(subdim_dicts)['Testpoint Description']

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
