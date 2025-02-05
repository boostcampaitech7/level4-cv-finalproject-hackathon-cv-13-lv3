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
# from utils import export_llm
# from torch_tensorrt.ptq import DataLoaderCalibrator, CalibrationAlgo

# Add custom module path
sys.path.append(str(Path(__file__).parent / "audiolm-trainer"))

# Custom modules
from salmonn_utils import SALMONNTestDataset, load_preprocessor, load_model
from config import Config
from utils import get_accelerator_dataloader
from train import setup_seeds
from metrics import compute_wer, compute_spider

device = "cuda:0"
batch_size = 8

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        
    def forward(
        self,
        inputs_embeds,
        attention_mask,
        max_new_tokens=200,
        num_beams=4,
        do_sample=True,
        min_length=1,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
    ):
        # module 속성 체크하여 config 접근
        config = self.model.module.config if hasattr(self.model, "module") else self.model.config
        
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )

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

def save_speech_encoder_trt(speech_encoder):
    # spectogram shape (8, 128, 3000), dtype=torch.float32
    sample_input = torch.randn(batch_size, 128, 3000, dtype=torch.float16, device=torch.device(device)).contiguous()
    
    speech_encoder.eval()
    
    # 워밍업
    with torch.no_grad():
        for _ in range(3):
            speech_encoder(sample_input)
    
    # 컴파일용 입력 정의
    compile_inputs = [torch_tensorrt.Input(
        shape=[batch_size, 128, 3000], 
        dtype=torch.float16,
        device=torch.device(device),
        format=torch.contiguous_format
    )]
    
    speech_encoder = torch_tensorrt.compile(
        speech_encoder,
        ir="dynamo",
        inputs=compile_inputs,
        enabled_precisions={torch.float16},  # float32만 사용
        optimization_level=5,
        strict_types=True,
        debug=True,
        # workspace_size=4 * 1 << 30,
        # precision_mode="mixed",  # mixed precision 모드 사용
    )
    
    # 테스트
    with torch.no_grad():
        for _ in range(3):
            speech_encoder(sample_input)
    
    # 저장할 때는 실제 텐서 사용
    save_inputs = [sample_input]
    torch_tensorrt.save(speech_encoder, "./trt_models/speech_trt.ep", inputs=save_inputs)
    
    del speech_encoder
    torch.cuda.empty_cache()
    
def save_beats_trt(beats):
    beats_wrapper = BeatsWrapper(beats)
    
    inputs = [
        torch.randn(8, 1, 320, 320, dtype=torch.float32, device=torch.device(device)),
        # torch.randn(8, 268800, dtype=torch.bool, device=torch.device(device)),
        torch_tensorrt.Input(
            min_shape=[1, 0],
            opt_shape=[8, 200000],
            max_shape=[16, 300000],
            dtype=torch.bool,
            device=torch.device(device)
        )
    ]
    
    enabled_precisions = {torch.float16, torch.float32, torch.bool}
    
    encoder = torch_tensorrt.compile(
        beats_wrapper,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        use_explicit_typing=True,
        immutable_weights=True,
        make_refittable=False,
        optimization_level=3,
        truncate_long_and_double=True,
        heuristic_mode=False,
        strict_type_constraints=True,
        device=torch.device(device),
    )
    torch_tensorrt.save(encoder, "./trt_models/beats_encoder_trt.ep", inputs=inputs)
    
    del encoder
    torch.cuda.empty_cache()

def save_speech_Qformer_trt(speech_Qformer):
    
    bert = speech_Qformer.bert
    # batch_size = 16일떄 1408, batch_size = 1일때 88
    compile_inputs = [
        torch_tensorrt.Input(
            shape=[88 * batch_size, 1, 768],
            dtype=torch.float32,
            device=torch.device(device)
        ),
        torch_tensorrt.Input(
            shape=[88 * batch_size, 17, 2048],
            dtype=torch.float32,
            device=torch.device(device)
        ),
        torch_tensorrt.Input(
            shape=[88 * batch_size, 17],
            dtype=torch.bool,
            device=torch.device(device)
        )
    ]
    
    enabled_precisions = {torch.float16, torch.float32, torch.bool}
    
    bert = torch_tensorrt.compile(
        bert,
        ir="dynamo",
        inputs=compile_inputs,
        enabled_precisions=enabled_precisions,
        use_explicit_typing=True,
        immutable_weights=True,
        make_refittable=False,
        optimization_level=3,
        truncate_long_and_double=True,
        heuristic_mode=False,
        strict_type_constraints=True,
        device=torch.device(device)
    )   
    
    
    # 즉, batch_size * 88이 첫 번째 입력 크기
    sample_inputs = [
        # (batch_size * 88, 1, 768)
        torch.randn(88 * batch_size, 1, 768, dtype=torch.float32, device=torch.device(device)), # qeury tokens
        # (batch_size * 88, 17, 2048)
        torch.randn(88 * batch_size, 17, 2048, dtype=torch.float32, device=torch.device(device)), # speech embeds
        # (batch_size * 88, 17)
        torch.ones(88 * batch_size, 17, dtype=torch.bool, device=torch.device(device)),   # speech atts mask - 0 또는 1의 값만 가짐
    ]
    torch_tensorrt.save(bert, "./trt_models/qformer_trt.ep", inputs=sample_inputs)
    
    del bert
    torch.cuda.empty_cache()

def save_llm_trt(llm):
    # 필요한 모듈들을 함수 시작 부분에서 import
    import torch
    import torch_tensorrt
    import pickle
    import torch.serialization
    
    torch.cuda.empty_cache()
    if hasattr(llm, "merge_and_unload"):
        llm = llm.merge_and_unload()
    
    llm.config.use_cache = False
    
    # Wrap the model
    llm = LLMWrapper(llm)
    llm.eval().cuda()
    
    seq_len = 111
    hidden_size = 3072
    
    # pickle 프로토콜 버전 설정
    torch.serialization.DEFAULT_PROTOCOL = 5
    pickle.DEFAULT_PROTOCOL = 5 
    
    compile_inputs = [
        torch_tensorrt.Input(
            shape=[batch_size, seq_len, hidden_size],  # inputs_embeds 형태
            dtype=torch.float16,
            device=torch.device(device),
            format=torch.contiguous_format
        ),
        torch_tensorrt.Input(
            shape=[batch_size, seq_len],
            dtype=torch.bool,
            device=torch.device(device),
            format=torch.contiguous_format,
            optional=True
        )
    ]
    
    # TensorRT 컴파일 옵션 수정
    llm = torch_tensorrt.compile(
        llm,
        ir="dynamo",
        inputs=compile_inputs,
        use_python_runtime=False,
        enabled_precisions={torch.float16},
        force_fp32_layers=["LayerNorm"],
        use_explicit_typing=True,
        optimization_level=3,
        device=torch.device(device),
        debug=True,
        truncate_long_and_double=True,
        strict_types=True,
    )
    
    # 실제 저장용 입력 생성
    inputs = [
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=torch.device(device)),
        torch.ones(batch_size, seq_len, dtype=torch.bool, device=torch.device(device))
    ]
    
    # 기본 저장 방식 사용
    torch_tensorrt.save(
        llm, 
        "./trt_models/llm_trt.ts", 
        inputs=inputs,
        output_format="torchscript"
    )
    
    del llm
    torch.cuda.empty_cache()
    
def load_aot_models():
    speech_encoder = torch_tensorrt.load("./trt_models/speech_trt.ep")
    beats = torch_tensorrt.load("./trt_models/beats_trt.ep")
    speech_Qformer = torch_tensorrt.load("./trt_models/qformer_trt.ep")
    llm = torch_tensorrt.load("./trt_models/llm_trt.ep")
    
    return speech_encoder, beats, speech_Qformer, llm

def compile_group_1(speech_encoder, beats, speech_Qformer):
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    # Create cache directory
    os.makedirs("trt_models", exist_ok=True)
    
    try:
        # 1. speech_encoder
        speech_encoder = speech_encoder.eval().cuda(0)
        save_speech_encoder_trt(speech_encoder)
        del speech_encoder
        torch.cuda.empty_cache()
        
        # # 2. beats (순차적으로 컴파일)
        # beats = beats.cuda(0)
        # save_beats_trt(beats)
        # del beats
        # torch.cuda.empty_cache()
        
        # # 3. speech_Qformer
        # speech_Qformer = speech_Qformer.cuda(0)
        # save_speech_Qformer_trt(speech_Qformer)
        # del speech_Qformer
        # torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error in compile_group_1: {e}")
        raise e

def compile_group_2(llm):
    # torch.cuda.set_device(1)
    torch.cuda.empty_cache()
        
    # Create directory
    os.makedirs("trt_models", exist_ok=True)
    
    try:
        # llm = llm.eval().cuda(1)
        llm = llm.eval().cuda()
        save_llm_trt(llm)
    except Exception as e:
        print(f"Error in compile_group_2: {e}")
        raise e

def main():
    # Config 객체 생성 전에 CUDA 디바이스 설정
    torch.cuda.set_device(int(device.split(':')[1]))
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    args = parse_args()
    cfg = Config(args)
    
    setup_seeds(cfg.config.run)
    
    os.makedirs("trt_models", exist_ok=True)
    
    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model

    torch_tensorrt.runtime.set_multi_device_safe_mode(True)
    torch_tensorrt.runtime.set_cudagraphs_mode(True)

    # torch.cuda.empty_cache()
        
    # # Create directory
    # os.makedirs("trt_models", exist_ok=True)
    
    speech_encoder = salmonn_preprocessor.speech_encoder.eval().cuda()
    save_speech_encoder_trt(speech_encoder)
    del speech_encoder
    torch.cuda.empty_cache()
    
    # beats = salmonn_preprocessor.beats.eval().cuda()
    # save_beats_trt(beats)
    # del beats
    # torch.cuda.empty_cache()
    
    speech_Qformer = salmonn_preprocessor.speech_Qformer.eval().cuda()
    save_speech_Qformer_trt(speech_Qformer)
    del speech_Qformer
    torch.cuda.empty_cache()
    
    compile_group_2(salmonn_preprocessor.llama_model)
    
    # 모델을 detach하고 requires_grad를 False로 설정
    # speech_encoder = salmonn_preprocessor.speech_encoder.detach().requires_grad_(False)
    # beats = salmonn_preprocessor.beats.detach().requires_grad_(False) if salmonn_preprocessor.beats is not None else None
    # speech_Qformer = salmonn_preprocessor.speech_Qformer.detach().requires_grad_(False) if salmonn_preprocessor.speech_Qformer is not None else None
    # llama_model = salmonn_preprocessor.llama_model.detach().requires_grad_(False)

    # p1 = mp.Process(target=compile_group_1, args=(speech_encoder, beats, speech_Qformer))
    # p2 = mp.Process(target=compile_group_2, args=(llama_model,))
    
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()
    # save_speech_encoder_trt(salmonn_preprocessor.speech_encoder)
    # save_beats_trt(salmonn_preprocessor.beats)
    # save_speech_Qformer_trt(salmonn_preprocessor.speech_Qformer)
    # save_llm_trt(salmonn_preprocessor.llama_model)

 
if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    random.seed(42)
    main()