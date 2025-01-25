import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import time
from typing import Tuple
import numpy as np
from torch import nn
from transformers import BertConfig

from trainer.models.attention_torch import (
    FlashWhisperAttention, FlashLlamaAttention, 
    FlashBertAttention, FlashMultiheadAttention
)
from trainer.models.modeling_whisper import WhisperAttention
from trainer.models.modeling_llama import LlamaAttention, LlamaConfig
from trainer.models.Qformer import BertAttention
from trainer.models.beats.backbone import MultiheadAttention

MODEL_CONFIGS = {
    "whisper": {
        "hidden_size": 1024,
        "num_attention_heads": 16,
    },
    "llama": {
        "hidden_size": 768,
        "num_attention_heads": 12,
    },
    # ... 다른 모델들의 설정
}

def create_attention_module(model_type, config):
    if model_type == "whisper":
        return FlashWhisperAttention(
            embed_dim=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            # ...
        )
    # ... 다른 모델들의 생성 로직

def benchmark_attention(
    original_module: nn.Module,
    flash_module: nn.Module,
    batch_size: int = 4,
    seq_length: int = 512,
    num_heads: int = 8,
    hidden_size: int = 512,
    num_runs: int = 100,
) -> Tuple[float, float, float]:
    """
    주어진 attention 모듈들의 성능을 비교합니다.
    
    Args:
        original_module: 원본 attention 모듈
        flash_module: Flash attention 모듈
        batch_size: 배치 크기
        seq_length: 시퀀스 길이
        num_heads: attention heads 수
        hidden_size: hidden state 크기
        num_runs: 실행 횟수
    
    Returns:
        Tuple[float, float, float]: (평균 속도 차이(ms), 최대 오차, 평균 오차)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_module = original_module.to(device)
    flash_module = flash_module.to(device)

    # 입력 데이터 생성
    hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(device)
    # 여기서 실제로 8192x1024가 되는 이유를 찾아야 함
    
    # 모델 타입에 따라 다른 입력 처리
    if isinstance(original_module, MultiheadAttention):
        # MultiheadAttention용 입력 준비 (T x B x C 형태)
        query = hidden_states.transpose(0, 1)  # [seq_length, batch_size, hidden_size]
        key = query
        value = query
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        additional_inputs = {"key": key, "value": value}
    else:
        # 다른 모델들용 입력 준비
        if isinstance(original_module, LlamaAttention):
            attention_mask = torch.ones(batch_size, 1, seq_length, seq_length).to(device)
            position_ids = torch.arange(seq_length).expand(batch_size, -1).to(device)
            additional_inputs = {"position_ids": position_ids}
        elif isinstance(original_module, WhisperAttention):
            attention_mask = torch.ones(batch_size, 1, seq_length, seq_length).to(device)
            additional_inputs = {}
        elif isinstance(original_module, BertAttention):
            attention_mask = torch.ones(batch_size, seq_length).to(device)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(hidden_states.dtype).min
            attention_mask = extended_attention_mask
            additional_inputs = {}
        else:
            attention_mask = torch.ones(batch_size, seq_length).to(device)
            additional_inputs = {}

    # 워밍업과 벤치마크 실행
    for _ in range(10):
        with torch.no_grad():
            if isinstance(original_module, MultiheadAttention):
                _ = original_module(query, key, value)[0]
                _ = flash_module(query, key, value)[0]
            else:
                _ = original_module(hidden_states, attention_mask, **additional_inputs)[0]
                _ = flash_module(hidden_states, attention_mask, **additional_inputs)[0]

    # 속도 측정
    original_times = []
    flash_times = []
    output_diffs = []
    
    for _ in range(num_runs):
        torch.cuda.synchronize()
        
        # 원본 모듈 실행 시간 측정
        start_time = time.perf_counter()
        with torch.no_grad():
            if isinstance(original_module, MultiheadAttention):
                original_output = original_module(query, **additional_inputs)[0]  # key와 value는 additional_inputs로 전달
            else:
                original_output = original_module(hidden_states, attention_mask, **additional_inputs)[0]
        torch.cuda.synchronize()
        original_times.append(time.perf_counter() - start_time)
        
        # Flash 모듈 실행 시간 측정
        start_time = time.perf_counter()
        with torch.no_grad():
            if isinstance(flash_module, FlashMultiheadAttention):
                flash_output = flash_module(query, **additional_inputs)[0]  # key와 value는 additional_inputs로 전달
            else:
                flash_output = flash_module(hidden_states, attention_mask, **additional_inputs)[0]
        torch.cuda.synchronize()
        flash_times.append(time.perf_counter() - start_time)
        
        # 출력 차이 계산
        output_diff = torch.abs(original_output - flash_output).max().item()
        output_diffs.append(output_diff)

    # 결과 계산
    original_avg_time = np.mean(original_times) * 1000  # ms로 변환
    flash_avg_time = np.mean(flash_times) * 1000
    speed_diff = original_avg_time - flash_avg_time
    max_diff = max(output_diffs)
    avg_diff = np.mean(output_diffs)

    return speed_diff, max_diff, avg_diff

def run_benchmarks():
    """
    모든 attention 구현에 대해 벤치마크를 실행합니다.
    """
    # 테스트 설정
    configs = {
        "small": {"batch_size": 4, "seq_length": 512, "num_heads": 8, "hidden_size": 512},
        "medium": {
            "batch_size": 8, 
            "seq_length": 1024,
            "num_heads": 16,
            "hidden_size": 1024,
        },
        "large": {
            "batch_size": 8,  # 줄임
            "seq_length": 1024,  # 줄임
            "num_heads": 16,
            "hidden_size": 1024,
        },
    }

    print("\nAttention 구현 벤치마크 결과:")
    print("=" * 80)

    for config_name, config in configs.items():
        print(f"\n설정: {config_name}")
        print("-" * 80)
        print(f"{'모델':20} {'속도 향상(ms)':>15} {'최대 오차':>15} {'평균 오차':>15}")
        print("-" * 80)

        hidden_size = config["hidden_size"]
        num_heads = config["num_heads"]

        # 각 설정에 맞는 config 생성
        bert_config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            layer_norm_eps=1e-12
        )

        llama_config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )

        # Whisper Attention 생성 및 초기화
        whisper = WhisperAttention(hidden_size, num_heads)
        flash_whisper = FlashWhisperAttention(hidden_size, num_heads)
        
        # 가중치 초기화를 동일하게
        with torch.no_grad():
            flash_whisper.q_proj.weight.copy_(whisper.q_proj.weight)
            flash_whisper.k_proj.weight.copy_(whisper.k_proj.weight)
            flash_whisper.v_proj.weight.copy_(whisper.v_proj.weight)
            flash_whisper.out_proj.weight.copy_(whisper.out_proj.weight)

        # 모델 페어 설정
        model_pairs = [
            (whisper, flash_whisper),
            (LlamaAttention(llama_config), FlashLlamaAttention(llama_config)),
            (BertAttention(bert_config), FlashBertAttention(bert_config)),
            (MultiheadAttention(hidden_size, num_heads), FlashMultiheadAttention(hidden_size, num_heads)),
        ]

        for original, flash in model_pairs:
            speed_diff, max_diff, avg_diff = benchmark_attention(
                original, flash, **config
            )
            model_name = type(original).__name__.replace("Attention", "")
            print(f"{model_name:20} {speed_diff:>15.3f} {max_diff:>15.3e} {avg_diff:>15.3e}")

if __name__ == "__main__":
    run_benchmarks() 