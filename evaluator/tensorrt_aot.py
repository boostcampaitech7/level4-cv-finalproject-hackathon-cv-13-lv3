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

class BeatsWrapper(nn.Module):
    def __init__(self, beats):
        super(BeatsWrapper, self).__init__()
        self.beats = beats
        
    def forward(self, inputs, padding_mask):
        return self.beats.extract_features(inputs, padding_mask, feature_only=True)

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
    # parser.add_argument(
    #     "--mode", 
    #     type=str, 
    #     default="valid_aac",
    #     choices=['submission_asr', 'submission_aac', 'valid_asr', 'valid_aac'],
    #     help="Evaluation mode"
    # )

    args = parser.parse_args()
    # args.task = args.mode.split("_")[1]
    # args.make_submission = args.mode.startswith("submission")
    
    return args

def get_dataset(dataset_cfg, run_cfg, task):
    testset = SALMONNTestDataset(
        dataset_cfg.prefix, dataset_cfg.test_ann_path, dataset_cfg.whisper_path, task
    )
    test_loader = get_accelerator_dataloader(testset, run_cfg, is_train=False)
    return test_loader

def get_calibrator(dataloader):
    return DataLoaderCalibrator(
        dataloader,
        cache_file="./tmp/trt_cache/calibration.cache",
        algo_type=CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device="cuda"
    )

def replace_test_ann_path(cfg, task):
    if "test_ann_path" not in cfg.config.datasets.keys():
        if task == "asr":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_asr
        elif task == "aac":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_aac
    return cfg

def save_speech_encoder_trt(speech_encoder):
    # 모델을 float32로 변환
    # speech_encoder = speech_encoder.float()
    
    # spectogram shape (8, 128, 3000), dtype=torch.float32
    sample_input = torch.randn(16, 128, 3000, dtype=torch.float16, device=torch.device("cuda:0")).contiguous()
    
    # 모델을 eval 모드로 설정
    speech_encoder.eval()
    
    # 워밍업 단계에서 더 많은 반복 수행
    with torch.no_grad():
        for _ in range(3):  # 여러 번 워밍업
            speech_encoder(sample_input)
    
    enabled_precisions = {torch.float16}
    
    inputs = [sample_input]
    speech_encoder = torch_tensorrt.compile(
        speech_encoder,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        # use_explicit_typing=True,
        # immutable_weights=True,
        # make_refittable=False,
        # optimization_level=5,
        # truncate_long_and_double=True,
        # heuristic_mode=False,
        # strict_type_constraints=False,
        device=torch.device("cuda:0"),
        # cuda_graph_batch_size=8,
        # strict_type_constraints=True,
        # device=torch.device("cuda:0"),
        # debug=False,               # 디버그 모드 비활성화
        # trace_only=False,         # 전체 변환 수행
        # preserve_parameters=True,  # 파라미터 보존
        # min_block_size=1          # 최소 블록 크기 설정
    )
    
    # 변환된 모델도 여러 번 테스트
    with torch.no_grad():
        for _ in range(3):
            speech_encoder(sample_input)
    
    # 저장 시에도 동일한 inputs 전달
    torch_tensorrt.save(speech_encoder, "./trt_models/speech_trt.ep", inputs=inputs)
    
    del speech_encoder
    torch.cuda.empty_cache()
    
def save_beats_trt(beats):
    beats_wrapper = BeatsWrapper(beats)
    
    inputs = [
        torch.randn(8, 1, 320, 320, dtype=torch.float32, device=torch.device("cuda:0")),
        # torch.randn(8, 268800, dtype=torch.bool, device=torch.device("cuda:0")),
        torch_tensorrt.Input(
            min_shape=[1, 0],
            opt_shape=[8, 200000],
            max_shape=[16, 300000],
            dtype=torch.bool,
            device=torch.device("cuda:0")
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
        optimization_level=1,
        truncate_long_and_double=True,
        heuristic_mode=False,
        strict_type_constraints=True,
        device=torch.device("cuda:0"),
    )
    torch_tensorrt.save(encoder, "./trt_models/beats_encoder_trt.ep", inputs=inputs)
    
    del encoder
    torch.cuda.empty_cache()

def save_speech_Qformer_trt(speech_Qformer):
    
    bert = speech_Qformer.bert
    # batch_size = 16일떄 1408, batch_size = 1일때 88
    # 즉, batch_size * 88이 첫 번째 입력 크기
    inputs = [
        # (batch_size * 88, 1, 768)
        torch.randn(88, 1, 768, dtype=torch.float32, device=torch.device("cuda:0")), # qeury tokens
        # (batch_size * 88, 17, 2048)
        torch.randn(88, 17, 2048, dtype=torch.float32, device=torch.device("cuda:0")), # speech embeds
        # (batch_size * 88, 17)
        torch.randint(0, 2, (88, 17), dtype=torch.float16, device=torch.device("cuda:0")),   # speech atts mask - 0 또는 1의 값만 가짐
    ]
    
    
    
    enabled_precisions = {torch.float16, torch.float32, torch.float16}
    
    bert = torch_tensorrt.compile(
        bert,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        use_explicit_typing=True,
        immutable_weights=True,
        make_refittable=False,
        optimization_level=1,
        truncate_long_and_double=True,
        heuristic_mode=False,
        strict_type_constraints=True,
        device=torch.device("cuda:0")
    )   
    torch_tensorrt.save(bert, "./trt_models/qformer_trt.ep", inputs=inputs)
    
    del bert
    torch.cuda.empty_cache()

def save_llm_trt(llm):
    if hasattr(llm, "merge_and_unload"):
        llm = llm.merge_and_unload()
    
    llm.config.use_cache = False
    # llm.half()
    
    # Wrap the model
    llm.eval().cuda()
    
    torch.cuda.empty_cache()
    
    batch_size = 8  # 로그에서 확인된 실제 배치 크기
    seq_len = 111    # 로그에서 확인된 실제 시퀀스 길이
    hidden_size = 3072  # 로그에서 확인된 임베딩 크기
    
    inputs = [
        None,
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=torch.device("cuda:0")), # input_embeds
        torch.randint(0, llm.config.vocab_size, (batch_size, seq_len), dtype=torch.int64, device=torch.device("cuda:0")), # attn_mask
    ]
    # calibrator = torch_tensorrt.ptq.Calibrator(...) # 양자화 callibrator
    
    llm = torch_tensorrt.compile(
        llm,
        ir="dynamo",
        inputs=inputs,
        use_python_runtime=False,
        enabled_precisions={torch.float16, torch.int64},
        force_fp32_layers=["LayerNorm"],  # LayerNorm은 FP32로 강제
        # dynamic_batching = True,# use_dynamic_shape=True, # 
        use_explicit_typing=True,
        immutable_weights=True,
        make_refittable=False,
        optimization_level=1,
        truncate_long_and_double=True,
        heuristic_mode=False,
        device=torch.device("cuda:0"),
    )
    torch_tensorrt.save(llm, "./trt_models/llm_trt.ep", inputs=inputs)
    
    del llm
    torch.cuda.empty_cache()
    
def load_aot_models():
    speech_encoder = torch_tensorrt.load("./trt_models/speech_trt.ep")
    beats = torch_tensorrt.load("./trt_models/beats_trt.ep")
    speech_Qformer = torch_tensorrt.load("./trt_models/qformer_trt.ep")
    llm = torch_tensorrt.load("./trt_models/llm_trt.ep")
    
    return speech_encoder, beats, speech_Qformer, llm

# 모델을 다시 tensorrt로 compile하지 않고 weights만 업데이트 (load_aot_models 이후에 사용)
def refit_models(model, updated_model):
    pass
    # llm = torch_tensorrt.load("llm_trt.ep")
    # new_llm = torch.export.export(model.llama_model, inputs)
    
    
    # new_trt_llm = refit_module_weights(llm, new_llm, args_input=inputs, in_place=True)
    
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
    args = parse_args()
    cfg = Config(args)
    # cfg = replace_test_ann_path(cfg, args.task)
    
    setup_seeds(cfg.config.run)
    
    os.makedirs("trt_models", exist_ok=True)
    
    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model

    # torch_tensorrt.runtime.set_cudagraphs_mode(True)
    torch_tensorrt.runtime.set_multi_device_safe_mode(True)

    # torch.cuda.empty_cache()
        
    # # Create directory
    # os.makedirs("trt_models", exist_ok=True)
    
    # speech_encoder = salmonn_preprocessor.speech_encoder.eval().cuda()
    # save_speech_encoder_trt(speech_encoder)
    # del speech_encoder
    # torch.cuda.empty_cache()
    
    # beats = salmonn_preprocessor.beats.eval().cuda()
    # save_beats_trt(beats)
    # del beats
    # torch.cuda.empty_cache()
    
    speech_Qformer = salmonn_preprocessor.speech_Qformer.eval().cuda()
    save_speech_Qformer_trt(speech_Qformer)
    del speech_Qformer
    torch.cuda.empty_cache()
    
    # compile_group_2(salmonn_preprocessor.llama_model)
    
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
    
    # Load data
    # dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task)
    
 
if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    random.seed(42)
    main()