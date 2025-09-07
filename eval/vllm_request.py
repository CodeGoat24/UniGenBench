import requests
import base64
import os
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import argparse
import pandas as pd
import concurrent.futures
from multiprocessing import Manager, Lock
import time
import random


class VLMessageClient:
    def __init__(self, api_url):
        self.api_url = api_url
        self.session = requests.Session() 

    def _encode_image(self, image):
        with Image.open(image) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")


    def build_messages(self, item, image_root=None):
        content = []
        if image_root:
            for i in range(len(item['images'])):
                item['images'][i] = os.path.join(image_root, item['images'][i])

        for i in range(len(item["images"])): 
            if os.path.exists(item['images'][i]):
                base64_image = self._encode_image(item['images'][i])
            else:
                base64_image = item['images'][i]
            content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })

        content.append({"type": "text", "text": item["problem"]})

        return [
            {
                "role": "user",
                "content": content
            }
        ]
    def contains_chinese(text):
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    def process_item(self, item, image_root, output_file, total_counter, lock):

        max_retries = 10
        attempt = 0
        result = None

        while attempt < max_retries:
            try:
                attempt += 1

                raw_messages = self.build_messages(item, image_root)
                headers = {"Content-Type": "application/json; charset=utf-8"}

                payload = {
                    "model": "QwenVL",
                    "messages": raw_messages,
                    # "do_sample": False,
                    "max_tokens": 2048,
                }


                response = self.session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=300 + attempt*5  
                )
                response.raise_for_status()

                output = response.json()["choices"][0]["message"]["content"]
                

                item['model_output'] = output
                item['success'] = True
                result = item

                break  

            except Exception as e:
                if attempt == max_retries:
                    print(f"请求失败（已达最大重试次数）: {str(e)}")

                    item['model_output'] = None
                    item['success'] = False
                    item['error'] = str(e)
                    item['attempt'] = attempt
                    result = item
                else:
                    sleep_time = min(2 ** attempt, 10)
                    time.sleep(sleep_time)

        return result, result.get("success", False) if result else False


def evaluate_batch(batch_data, api_url, image_root=None):
    total_result = []
    client = VLMessageClient(api_url)

    from tqdm import tqdm
    index = 0
    with tqdm(total=len(batch_data), desc="vLLM inference") as pbar:
        for item in batch_data:
            if 'idx' not in item:
                item['idx'] = str(index)
                index += 1
            try:
                result, _ = client.process_item(
                    item=item,
                    image_root=image_root,
                    output_file='./results.json',
                    total_counter=None,  
                    lock=None           
                )
                total_result.append(result)
            except Exception as e:
                print(f"Error: {str(e)}")
            finally:
                pbar.update(1)
                processed_info = f"{len(total_result)}/{len(batch_data)}"
                pbar.set_postfix({
                    "processed": processed_info
                })

    if len(total_result) > 0:
        total_result.sort(key=lambda x: int(x['idx']))

    return total_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://localhost:8080")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--output_path", default="./results.json")
    args = parser.parse_args()

    image_path = ""
    problem = ""

    batch_data = [
            {
                "images": [
                    image_path
                ],
                "problem": problem,
            },
        ]

    evaluate_batch(batch_data, args.api_url, image_root=args.image_root)

if __name__ == "__main__":
    main()
