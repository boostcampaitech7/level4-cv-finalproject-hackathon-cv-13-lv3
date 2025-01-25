# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, MessagesPreprocessor, load_dataset, register_dataset

import json
import random

import sys
sys.path.append('/data/skh/evaluator/audiolm-trainer')

class Q2APreprocessor(MessagesPreprocessor):

    def __init__(self, prompt_path):
        self.prompt_dict = {}
        self.prefix = "/data/dataset"
        #{'role': 'system', 'content': 'You are a helpful assistant.'}, \
        prompt_template = "[{'role': 'user', 'content': '<audio><PROMPT>'},\
                            {'role': 'assistant', 'content': '<TEXT>'}]"
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                # ex) prompt_template.format("Transcribe the following audio: <SpeechHere>")
                filted_prompts = [raw_prompt.replace("<Speech><SpeechHere></Speech>", "") for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.replace("<PROMPT>", raw_prompt) for raw_prompt in filted_prompts]
        super().__init__()

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # prepare prompts
        row = row['annotation']
        prompt=""

        if self.prompt_dict:
            # 단일 Task인 경우, 해당 Task의 Prompt 목록에서 무작위로 하나 선택
            prompt = random.choice(self.prompt_dict[row["task"]])

        audio_path = (self.prefix + '/' + row["path"]).replace("//", "/")
        message_content = {
            "text": row["text"],
            "audio": prompt.replace("<TEXT>", "") # 프롬프트에서 <TEXT> 부분 제거 후 audio 필드에 저장
        }

        messages = [
            {"role": "user", "content": message_content} # user의 content에 text와 audio를 담습니다.
            ]
        
        data = {
            "messages": messages, # 수정된 부분입니다. messages 리스트를 포함합니다.
            "audios": [audio_path] # audios는 리스트 형태로 유지합니다.
        }
        
        json_data = json.dumps(data, ensure_ascii=False) # JSON 문자열로 변환 (ensure_ascii=False는 한글 깨짐 방지)
        
        return super().preprocess(json.loads(json_data)) # json 문자열을 다시 dict로 변환하여 넘겨줍니다.


register_dataset(
    DatasetMeta(
        ms_dataset_id='cv13/stage1_valid_01',
        dataset_path='/data/dataset/stage1_sample1.json',
        preprocess_func=Q2APreprocessor(prompt_path='/data/skh/trainer/prompts/train_prompt.json'),
    ))

if __name__ == '__main__':

    train, valid = load_dataset(['cv13/stage1_valid_01'])
    print(f'dataset: {train}')
    print(f'dataset[0]: {train[0]}')


#CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 swift sft --torch_dtype 'bfloat16' \
# --use_liger 'True' --model 'Qwen/Qwen2.5-0.5B-Instruct' --custom_register_path /data/skh/dataset_q2a_swift.py \
# --dataset 'cv13/stage1_valid_01' --max_length '1024' --lora_rank '32' --lora_alpha '8' --init_weights 'True' \
# --learning_rate '1e-4' --attn_impl 'flash_attn' --gradient_accumulation_steps '16' --eval_steps '500' \
# --output_dir /data/skh/ --template default

#CUDA_VISIBLE_DEVICES=0 swift sft --torch_dtype 'bfloat16' \
# --use_liger 'True' --model 'Qwen/Qwen2.5-0.5B-Instruct' --custom_register_path /data/skh/dataset_q2a_swift.py \
# --dataset 'cv13/stage1_valid_01' --max_length '1024' --lora_rank '32' --lora_alpha '8' --init_weights 'True' \
# --learning_rate '1e-4' --attn_impl 'flash_attn' --gradient_accumulation_steps '16' --eval_steps '500' \
# --output_dir /data/skh/ --template default