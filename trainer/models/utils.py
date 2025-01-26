# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import StoppingCriteria
from peft import TaskType, LoraConfig, AdaLoraConfig, AdaptionPromptConfig

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    
 
def set_lora_config(cfg):
    lora_type = cfg.lora_type
    rank = cfg.rank
    alpha = cfg.alpha
    dropout = cfg.dropout
    targets = cfg.target_modules
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 언어 모델링 태스크
        inference_mode=False,  # 학습 모드로 설정
        r=rank,  # LoRA 랭크 차원 (8-32 권장)
        lora_alpha=alpha,  # 스케일링 팩터, 보통 r의 2배
        lora_dropout=dropout,  # 드롭아웃 확률 (0.1 권장)
        target_modules=targets,  # LoRA를 적용할 모듈들
    )
    
    if lora_type not in ["lora", "adalora", "llama-adapter"]:
        return lora_config
    
    if lora_type == "lora":
        return lora_config
    elif lora_type == "adalora":
        return AdaLoraConfig(
            peft_type=lora_type,
            task_type=TaskType.CAUSAL_LM,  # 언어 모델링 태스크
            inference_mode=False,  # 학습 모드로 설정
            r=rank,  # 초기 LoRA 랭크
            lora_alpha=alpha,  # 스케일링 팩터
            lora_dropout=dropout,  # 드롭아웃 확률
            target_modules=targets,  # AdaLoRA를 적용할 모듈들
            target_r=cfg.target_r if cfg.target_r else 8,  # 목표 랭크 (최종적으로 줄일 랭크 크기)
            init_r=cfg.init_r if cfg.init_r else 12,  # 초기 랭크 크기
            beta1=cfg.beta1 if cfg.beta1 else 0.85,  # AdaLoRA 업데이트의 첫 번째 모멘텀
            beta2=cfg.beta2 if cfg.beta2 else 0.85,  # AdaLoRA 업데이트의 두 번째 모멘텀
        )
    elif lora_type == "llama-adapter":
        return AdaptionPromptConfig(
            peft_type=lora_type,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            adapter_len=cfg.adapter_len if cfg.adapter_len else 32,  # 어댑터 토큰 수 (8-32 권장)
            adapter_layers=cfg.adapter_layers if cfg.adapter_layers else 30,  # 어댑터를 적용할 레이어 수
            r=rank,  # 어댑터 차원 (8-64 권장)
            lora_alpha=alpha,  # 보통 r의 2배
            lora_dropout=dropout,  # 0.1 권장
            target_modules=targets,
        )