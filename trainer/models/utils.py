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
from .quantization_configs import setup_awq_model, setup_gptq_model, setup_qlora_model, setup_hybrid_quant_model, setup_eetq_model, setup_hqq_model
from peft import prepare_model_for_kbit_training

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    

# 공통 후처리 및 최적화 함수
def post_quantization_optimization(model, is_train=True):
    """
    학습/추론에 따른 최적화 설정 적용
    """
    if is_train:
        # 학습 시에만 필요한 설정
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        # 추론 시에만 필요한 설정
        model.config.use_cache = True
    
    # 공통 최적화 설정
    torch.cuda.empty_cache()
    
    # Flash Attention 2.0 사용 (가능한 경우)
    if hasattr(model.config, "use_flash_attention_2"):
        model.config.use_flash_attention_2 = True
            
    # 컴파일 최적화 (메모리 누수 방지)
    if torch.cuda.is_available():
        model = torch.compile(
            model, 
            mode="reduce-overhead"
        )
    return model

# 실제 사용 예시
def setup_quantized_model(model_path, token, quant_method="awq", is_train=True):
    if quant_method == "qlora":
        model = setup_qlora_model(model_path, token)
    elif quant_method == "awq":
        model = setup_awq_model(model_path, token, "c4-new")
    elif quant_method == "gptq":
        model = setup_gptq_model(model_path, token, "c4-new")
    elif quant_method == "hybrid":
        model = setup_hybrid_quant_model(model_path, token)
    elif quant_method == "eetq":
        model = setup_eetq_model(model_path, token)
    elif quant_method == "hqq":
        model = setup_hqq_model(model_path, token)
    else:
        raise ValueError(f"Unsupported quantization method: {quant_method}")

    if is_train:
        # LoRA 학습을 위한 모델 준비
        model = prepare_model_for_kbit_training(model)
    
    # 학습/추론 모드에 따른 최적화 적용
    model = post_quantization_optimization(model, is_train=is_train)
    
    return model
