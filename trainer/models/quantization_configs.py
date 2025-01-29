import torch
from transformers import StoppingCriteria,  BitsAndBytesConfig, GPTQConfig, AutoModelForCausalLM, AwqConfig
from peft import prepare_model_for_kbit_training

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