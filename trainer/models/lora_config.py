from peft import TaskType, LoraConfig, AdaLoraConfig, AdaptionPromptConfig, IA3Config

def set_lora_config(cfg):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 언어 모델링 태스크
        inference_mode=False,  # 학습 모드로 설정
        r=cfg.rank,  # LoRA 랭크 차원 (8-32 권장)
        alpha=cfg.alpha,  # 스케일링 팩터, 보통 r의 2배
        dropout=cfg.dropout,  # 드롭아웃 확률 (0.1 권장)
        target_modules=cfg.target_modules,  # LoRA를 적용할 모듈들
    )
    
def set_adalora_config(cfg):
    return AdaLoraConfig(
        peft_type="adalora",
        task_type=TaskType.CAUSAL_LM,  # 언어 모델링 태스크
            inference_mode=False,  # 학습 모드로 설정
            r=cfg.rank,  # 초기 LoRA 랭크
            alpha=cfg.alpha,  # 스케일링 팩터
            dropout=cfg.dropout,  # 드롭아웃 확률
            target_modules=cfg.target_modules,  # AdaLoRA를 적용할 모듈들
            target_r=cfg.target_r if cfg.target_r else 8,  # 목표 랭크 (최종적으로 줄일 랭크 크기)
            init_r=cfg.init_r if cfg.init_r else 12,  # 초기 랭크 크기
            beta1=cfg.beta1 if cfg.beta1 else 0.85,  # AdaLoRA 업데이트의 첫 번째 모멘텀
            beta2=cfg.beta2 if cfg.beta2 else 0.85,  # AdaLoRA 업데이트의 두 번째 모멘텀
        )

def set_adaption_prompt_config(cfg):
    return AdaptionPromptConfig(
            peft_type="llama-adapter",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            adapter_len=cfg.adapter_len if cfg.adapter_len else 32,  # 어댑터 토큰 수 (8-32 권장)
            adapter_layers=cfg.adapter_layers if cfg.adapter_layers else 30,  # 어댑터를 적용할 레이어 수
            r=cfg.rank,  # 어댑터 차원 (8-64 권장)
            alpha=cfg.alpha,  # 보통 r의 2배
            dropout=cfg.dropout,  # 0.1 권장
            target_modules=cfg.target_modules,
        )
    
def set_ia3_config(cfg):
    return IA3Config(
        peft_type="ia3",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.rank,
        alpha=cfg.alpha,
        dropout=cfg.dropout,
        target_modules=cfg.target_modules,
    )