import torch
from transformers import StoppingCriteria,  BitsAndBytesConfig, GPTQConfig, AutoModelForCausalLM, AwqConfig, EetqConfig, HqqConfig

# AWQ, GTPQ 방식은 LoRA 방식이 지원되는지 확실하지 않음 - 확인 필요
# 만약 안된다면 LoRA adapter만 따로 추출한 뒤 원래 LLM에 병합시켜 한 모델로 만든 뒤
# 저장해 추론할 때 양자화

# AWQ 방식 - 추론 속도 최적화에 적합 (실시간 추론)
def setup_awq_model(model_path, token, datasets):
    quant_config = AwqConfig(
        bits=4,
        dataset=datasets
    )
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token,
        quantization_config=quant_config
    )
    
    return model

# GPTQ 방식 - 높은 압축률, Post-Training 양자화 사용 (안정적 대규모 배치)
def setup_gptq_model(model_path, token, datasets):
    # GPTQ 설정
    gptq_config = GPTQConfig(
        bits=4,  
        group_size=128,
        desc_act=True,
        dataset=datasets,
        disable_exllama=False,  # exllama 커널 비활성화
        use_cuda_fp16=True     # FP16 연산 사용
    )
    
    # GPTQ 모델 초기화 및 양자화
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token,
        quantization_config=gptq_config,
        torch_dtype=torch.float16,
        device_map="auto",  # 자동 디바이스 매핑
    )
    
    return model

# Fine-Tuning
def setup_qlora_model(model_path, token):
    # QLoRA 양자화 설정
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # bfloat16 대신 float16 사용
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=True,
        bnb_4bit_use_cpu_offload=False  # GPU 메모리가 충분한 경우
    )
    
    # 기본 모델 로드 및 4bit 양자화 적용
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token,
        quantization_config=quant_config,
        low_cpu_mem_usage=True
    )
    
    return model

# 중요도에 따라 레이어별로 다른 비트 적용 (예: Attention 레이어 → 8-bit, FFN 레이어 → 4-bit) 
def setup_hybrid_quant_model(model_path, token):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        mixed_precision=True,  # 중요 레이어는 8-bit 유지
        target_modules=["attn", "ffn"]  # 레이어 타겟팅
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, token=token, quantization_config=quant_config)
    return model

# EETQ 방식 (LLM.int() 방식 보다 빠른 8-bit 양자화로 알려짐)
def setup_eetq_model(model_path, token):
    quant_config = EetqConfig("int8")
    model = AutoModelForCausalLM.from_pretrained(model_path, token=token, quantization_config=quant_config)
    return model

# HQQ 방식 - 2, 3-bit 양자화에 최적화
def setup_hqq_model(model_path, token):
    quant_config = HqqConfig(nbits=2, group_size=64)
    model = AutoModelForCausalLM.from_pretrained(model_path, token=token, quantization_config=quant_config)
    return model
