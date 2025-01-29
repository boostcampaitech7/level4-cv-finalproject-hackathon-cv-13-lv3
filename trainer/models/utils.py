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
from transformers import StoppingCriteria,  BitsAndBytesConfig, GPTQConfig, AutoModelForCausalLM, AwqConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    
# AWQ 방식
def setup_awq_model(model_path, token):
    quant_config = AwqConfig(
        bits=4
    )
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token,
        device_map="auto",
        quantization_config=quant_config
    )
    
    
    return model

# GPTQ 방식
def setup_gptq_model(model_path, token):
    # GPTQ 설정
    gptq_config = GPTQConfig(
        bits=4,  # quantization bits
        group_size=128,  # group size for quantization
        desc_act=True  # whether to use activation descriptors
    )
    
    # GPTQ 모델 초기화 및 양자화
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token,
        quantization_config=gptq_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    return model


def setup_qlora_model(model_path, token):
    # QLoRA 양자화 설정
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 기본 모델 로드 및 4bit 양자화 적용
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    # LoRA 학습을 위한 모델 준비
    model = prepare_model_for_kbit_training(model)
    
    return model

# SmoothQuant 방식

# LLM.int8() 방식
def setup_llm_int8_model(model_path, token):
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token,
        device_map="auto",
        quantization_config=quant_config
    )
    
    return model

# 공통 후처리 및 최적화 함수
def post_quantization_optimization(model):
    # 메모리 최적화
    model.config.use_cache = False  # 캐시 비활성화로 메모리 사용량 감소
    
    # (선택사항) Flash Attention 활성화
    if hasattr(model, 'enable_flash_attention'):
        model.enable_flash_attention()
    
    # (선택사항) 모델 융합 최적화
    if torch.cuda.is_available():
        model = torch.compile(model, mode="reduce-overhead")
    
    return model

# 실제 사용 예시
def initialize_quantized_model(model_path, token, quant_method="awq"):
    if quant_method == "awq":
        model = setup_awq_model(model_path, token)
    elif quant_method == "gptq":
        model = setup_gptq_model(model_path, token)
    elif quant_method == "qlora":
        model = setup_qlora_model(model_path, token)
    else:
        raise ValueError(f"Unsupported quantization method: {quant_method}")
    
    # 후처리 최적화 적용
    model = post_quantization_optimization(model)
    
    return model