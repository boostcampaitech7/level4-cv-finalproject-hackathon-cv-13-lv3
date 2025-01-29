import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from awq import AutoAWQForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class QuantizationCombinations:
    @staticmethod
    def fast_inference_setup(model_path, device="cuda"):
        """
        빠른 추론을 위한 조합: AWQ 4-bit + Outlier 처리
        """
        # AWQ 설정
        awq_config = {
            "zero_point": True,  # 영점 조정으로 정확도 향상
            "q_group_size": 128,  # 그룹 크기
            "w_bit": 4,  # 4비트 양자화
            "outlier_threshold": 3.0,  # 이상치 임계값
        }

        # AWQ 모델 초기화
        model = AutoAWQForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            safetensors=True,
            device_map="auto"
        )

        # 이상치 처리를 위한 설정
        outlier_config = {
            "threshold": 6.0,
            "sample_size": 128
        }

        # AWQ 양자화 적용
        model.quantize(
            awq_config,
            batch_size=1,
            export_path="awq_quantized_model"
        )

        # Flash Attention 활성화 (가능한 경우)
        if hasattr(model, "enable_flash_attention"):
            model.enable_flash_attention()

        # torch.compile 적용
        if torch.cuda.is_available():
            model = torch.compile(model, mode="reduce-overhead")

        return model

    @staticmethod
    def finetuning_setup(model_path, device="cuda"):
        """
        파인튜닝을 위한 조합: QLoRA (4-bit 양자화 + LoRA)
        """
        # QLoRA 양자화 설정
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_has_fp16_weight=True
        )

        # 모델 로드 및 양자화 적용
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # kbit 학습을 위한 모델 준비
        model = prepare_model_for_kbit_training(model)

        # LoRA 설정
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # LoRA 적용
        model = get_peft_model(model, lora_config)
        
        # 학습 가능한 파라미터 출력
        model.print_trainable_parameters()

        return model

    @staticmethod
    def stable_inference_setup(model_path, device="cuda"):
        """
        안정적인 추론을 위한 조합: LLM.int8() + SmoothQuant
        """
        # LLM.int8 설정
        int8_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True,
            llm_int8_skip_modules=None
        )

        # SmoothQuant 설정
        smooth_config = {
            "smoothing_factor": 0.5,
            "max_input_value": 20.0,
            "activation_scale": "per_token"
        }

        # 모델 로드 및 양자화 적용
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=int8_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # SmoothQuant 적용을 위한 후처리
        def apply_smooth_quant(model, config):
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # 활성화 값 스케일링
                    if hasattr(module, "weight"):
                        weight_scale = torch.max(torch.abs(module.weight.data)) * config["smoothing_factor"]
                        module.weight.data = torch.clamp(
                            module.weight.data,
                            min=-weight_scale,
                            max=weight_scale
                        )

        apply_smooth_quant(model, smooth_config)

        return model

    @staticmethod
    def optimize_model(model, optimization_level="max"):
        """
        모델 최적화 및 성능 향상을 위한 공통 후처리
        """
        # 메모리 최적화
        model.config.use_cache = False
        
        # 배치 처리 최적화
        if hasattr(model, "enable_batch_efficient_inference"):
            model.enable_batch_efficient_inference()
        
        # Flash Attention 활성화 (가능한 경우)
        if hasattr(model, "enable_flash_attention"):
            model.enable_flash_attention()
        
        if optimization_level == "max" and torch.cuda.is_available():
            model = torch.compile(model, mode="reduce-overhead")
        
        return model

def setup_quantized_model(model_path, setup_type="fast", optimization_level="max"):
    """
    사용자 친화적인 설정 함수
    """
    if setup_type == "fast":
        model = QuantizationCombinations.fast_inference_setup(model_path)
    elif setup_type == "finetune":
        model = QuantizationCombinations.finetuning_setup(model_path)
    elif setup_type == "stable":
        model = QuantizationCombinations.stable_inference_setup(model_path)
    else:
        raise ValueError(f"Unknown setup type: {setup_type}")

    # 공통 최적화 적용
    model = QuantizationCombinations.optimize_model(model, optimization_level)
    
    return model

# 사용 예시
def example_usage():
    model_path = "your/model/path"
    
    # 1. 빠른 추론 설정
    fast_model = setup_quantized_model(model_path, setup_type="fast")
    
    # 2. 파인튜닝 설정
    finetune_model = setup_quantized_model(model_path, setup_type="finetune")
    
    # 3. 안정적 추론 설정
    stable_model = setup_quantized_model(model_path, setup_type="stable")