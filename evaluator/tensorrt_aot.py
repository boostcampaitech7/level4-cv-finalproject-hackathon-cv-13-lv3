# Standard library imports
import argparse
import json
import random
import sys
import os
import time
from pathlib import Path
import multiprocessing as mp

# Third-party imports
import torch
import torch_tensorrt
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from accelerate import Accelerator
from torch_tensorrt.dynamo import refit_module_weights
from torch.export import export

# Add custom module path
sys.path.append(str(Path(__file__).parent / "audiolm-trainer"))

# Custom modules
from salmonn_utils import SALMONNTestDataset, load_preprocessor, load_model
from config import Config
from utils import get_accelerator_dataloader
from train import setup_seeds
from metrics import compute_wer, compute_spider

class LLMWrapper(nn.Module):
    def __init__(self, model):
        super(LLMWrapper, self).__init__()
        self.model = model
        
    def forward(
        self,
        inputs_embeds = None,
        attention_mask = None,
        input_ids = None,
        position_ids = None,
        past_key_values = None,
        query_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            query_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict=False
        )
        return outputs
    
    @torch.jit.export
    def generate(
        self,
        inputs_embeds,
        attention_mask,
        pad_token_id=None,
        max_new_tokens=200,
        num_beams=4,
        do_sample=True,
        min_length=1,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
    ):
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id[0],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )

    @property
    def base_model(self):
        return self.model

def parse_args():
    parser = argparse.ArgumentParser(description='SALMONN Evaluation Script')
    parser.add_argument(
        "--cfg-path", 
        type=str, 
        help='path to configuration file', 
        default='salmonn_eval_config.yaml'
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override settings in the config"
    )

    args = parser.parse_args()
    return args

def save_speech_encoder_trt(speech_encoder, config):
    # spectogram shape (8, 128, 3000), dtype=torch.float32
    sample_input = torch.randn(config.batch_size_eval, 128, 3000, dtype=torch.float16, device=torch.device(config.tensorrt_device))
    
    # 워밍업
    with torch.no_grad():
        for _ in range(3):
            speech_encoder(sample_input)
    
    # 컴파일용 입력 정의
    compile_inputs = [torch_tensorrt.Input(
        shape=[config.batch_size_eval, 128, 3000], 
        dtype=torch.float16,
        device=torch.device(config.tensorrt_device),
        format=torch.contiguous_format
    )]
    
    speech_encoder = torch_tensorrt.compile(
        speech_encoder,
        ir="dynamo",
        inputs=compile_inputs,
        enabled_precisions={torch.float16},  # float32만 사용
        optimization_level=config.optimization_level,
        use_explicit_typing=True,
        truncate_long_and_double=True,
        strict_types=True,
        device=torch.device(config.tensorrt_device),
        debug=True,
    )
    
    # 테스트
    with torch.no_grad():
        for _ in range(3):
            speech_encoder(sample_input)
    
    # 저장할 때는 실제 텐서 사용
    save_inputs = [sample_input]
    torch_tensorrt.save(speech_encoder, f"./trt_models/speech_trt_batch{config.batch_size_eval}.ep", inputs=save_inputs)
    
    del speech_encoder
    torch.cuda.empty_cache()

def save_llm_trt(llm, config):
    torch.cuda.empty_cache()
    if hasattr(llm, "merge_and_unload"):
        llm = llm.merge_and_unload()

    llm.config.use_cache = False

    # Wrap the model
    llm = LLMWrapper(llm)
    llm.eval().cuda()

    seq_len = 111
    hidden_size = 3072

    compile_inputs = [
        torch_tensorrt.Input(
            shape=[config.batch_size_eval, seq_len, hidden_size],  # inputs_embeds 형태
            dtype=torch.float16,
            device=torch.device(config.tensorrt_device)
        ),
        torch_tensorrt.Input(
            shape=[config.batch_size_eval, seq_len],
            dtype=torch.bool,
            device=torch.device(config.tensorrt_device),
            optional=True
        )
    ]

    # TensorRT 컴파일 옵션 수정
    llm = torch_tensorrt.compile(
        llm,
        ir="dynamo",
        inputs=compile_inputs,
        enabled_precisions={torch.float16},  # float16만 사용하도록 강제
        optimization_level=config.optimization_level,
        use_explicit_typing=True,
        truncate_long_and_double=True,
        strict_types=True,
        device=torch.device(config.tensorrt_device),
        debug=True,
    )

    # 실제 저장용 입력 생성
    inputs = [
        torch.randn(config.batch_size_eval, seq_len, hidden_size, dtype=torch.float16, device=torch.device(config.tensorrt_device)),
        torch.ones(config.batch_size_eval, seq_len, dtype=torch.bool, device=torch.device(config.tensorrt_device))
    ]

    # 기본 저장 방식 사용
    torch_tensorrt.save(
        llm, 
        f"./trt_models/llm_trt_batch{config.batch_size_eval}.ts", 
        inputs=inputs,
        output_format="torchscript"
    )

    del llm
    torch.cuda.empty_cache()

def save_aot_models(speech_encoder, llm, config):
    try:
        save_speech_encoder_trt(speech_encoder, config)
        save_llm_trt(llm, config)
    except Exception as e:
        print(f"Failed to save TensorRT models: {e}")
        raise

def main(args):
    # Config 객체 생성 전에 CUDA 디바이스 설정
    cfg = Config(args)
    torch.cuda.set_device(int(cfg.config.run.tensorrt_device.split(':')[1]))
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    setup_seeds(cfg.config.run)
    
    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model
    salmonn_preprocessor.eval()

    torch_tensorrt.runtime.set_multi_device_safe_mode(True)
    torch_tensorrt.runtime.set_cudagraphs_mode(True)

    torch.cuda.empty_cache()
        
    # # Create directory
    os.makedirs("trt_models", exist_ok=True)
    
    save_aot_models(
        salmonn_preprocessor.speech_encoder, 
        salmonn_preprocessor.llama_model, 
        cfg.config.run
    )
 
if __name__ == '__main__':
    args = parse_args()
    random.seed(42)
    main(args)