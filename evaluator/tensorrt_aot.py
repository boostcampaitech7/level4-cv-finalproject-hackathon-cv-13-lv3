# Standard library imports
import argparse
import json
import random
import sys
import os
import time
from pathlib import Path

# Third-party imports
import torch
import torch_tensorrt
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from torch_tensorrt.dynamo import refit_module_weights
from torch_tensorrt.ptq import DataLoaderCalibrator, CalibrationAlgo

# Add custom module path
sys.path.append(str(Path(__file__).parent / "audiolm-trainer"))

# Custom modules
from salmonn_utils import SALMONNTestDataset, load_preprocessor, load_model
from config import Config
from utils import get_accelerator_dataloader
from train import setup_seeds
from metrics import compute_wer, compute_spider

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
    parser.add_argument(
        "--mode", 
        type=str, 
        default="valid_aac",
        choices=['submission_asr', 'submission_aac', 'valid_asr', 'valid_aac'],
        help="Evaluation mode"
    )
    parser.add_argument(
        "--timer",
        action="store_true",
        help="Enable timer"
    )

    args = parser.parse_args()
    args.task = args.mode.split("_")[1]
    args.make_submission = args.mode.startswith("submission")
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
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 80, 1500],  # [batch_size, mel_bins, sequence_length]
            opt_shape=[32, 80, 1500],  # Whisper의 기본 입력 크기
            max_shape=[64, 80, 1500],  # 모델이 처리할 수 있는 최대 크기
            dtype=torch.half,
        )
    ]
    enabled_precisions = {torch.float16, torch.half}
    
    speech_encoder = torch_tensorrt.compile(
        speech_encoder,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        use_explicit_typing=True,
        immutable_weights=False,
        reuse_cached_engines=True,
        cache_built_engines=True
    )
    torch_tensorrt.save(speech_encoder, "speech_trt.ep", inputs=inputs)

def save_beats_trt(beats):
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 1, 320, 320],  # [batch_size, channels, height, width]
            opt_shape=[32, 1, 320, 320],  # BEATs의 기본 입력 크기
            max_shape=[64, 1, 320, 320],  # patch_embedding: Conv2d(1, 512, kernel_size=(16, 16))
            dtype=torch.half,            # 320x320 -> (20x20) patches
        )
    ]
    
    enabled_precisions = {torch.float16, torch.half}
    
    beats = torch_tensorrt.compile(
        beats,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        use_explicit_typing=True,
        immutable_weights=False,
        reuse_cached_engines=True,
        cache_built_engines=True
    )
    torch_tensorrt.save(beats, "beats_trt.ep", inputs=inputs)

def save_speech_Qformer_trt(speech_Qformer):
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 32, 768],  # [batch_size, num_query_tokens, hidden_size]
            opt_shape=[32, 32, 768],  # BertConfig.hidden_size = 768
            max_shape=[64, 32, 768],  # num_query_tokens = 32 (from init_speech_Qformer)
            dtype=torch.half,
        ),
        torch_tensorrt.Input(
            min_shape=[1, 94, 1280],  # [batch_size, sequence_length, encoder_hidden_size]
            opt_shape=[32, 94, 1280],  # Whisper encoder output: 1280 dim
            max_shape=[64, 94, 1280],  # seq_len = 1500/16 ≈ 94 (after conv layers)
            dtype=torch.half,
        )
    ]
    
    enabled_precisions = {torch.float16, torch.half}
    
    speech_Qformer = torch_tensorrt.compile(
        speech_Qformer,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        use_explicit_typing=True,
        immutable_weights=False,
        reuse_cached_engines=True,
        cache_built_engines=True
    )   
    torch_tensorrt.save(speech_Qformer, "qformer_trt.ep", inputs=inputs)

def save_llm_trt(llm):
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 1500],  # [batch_size, sequence_length]
            opt_shape=[32, 1500],  # LLaMA의 입력 시퀀스 길이
            max_shape=[64, 1500],  # Whisper의 출력 길이와 맞춤
            dtype=torch.int32,
            device='cuda'
        )
    ]
    
    enabled_precisions = {torch.float16, torch.int8}
    calibrator = torch_tensorrt.ptq.Calibrator(...) # 양자화 callibrator
    
    llm = torch_tensorrt.compile(
        llm,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        calibrator=calibrator,
        use_explicit_typing=True,
        immutable_weights=False,
        reuse_cached_engines=True,
        cache_built_engines=True,
        max_workspace_size=4 * (1 << 30),
        device='cuda'
    )
    torch_tensorrt.save(llm, "llm_trt.ep", inputs=inputs)

def load_aot_models():
    speech_encoder = torch_tensorrt.load("speech_trt.ep")
    beats = torch_tensorrt.load("beats_trt.ep")
    speech_Qformer = torch_tensorrt.load("qformer_trt.ep")
    llm = torch_tensorrt.load("llm_trt.ep")
    
    return speech_encoder, beats, speech_Qformer, llm

# 모델을 다시 tensorrt로 compile하지 않고 weights만 업데이트 (load_aot_models 이후에 사용)
def refit_models(model, updated_model):
    pass
    llm = torch_tensorrt.load("llm_trt.ep")
    new_llm = torch.export.export(model.llama_model, inputs)
    
    
    new_trt_llm = refit_module_weights(llm, new_llm, args_input=inputs, in_place=True)
    
def main():
    args = parse_args()
    cfg = Config(args)
    cfg = replace_test_ann_path(cfg, args.task)
    
    setup_seeds(cfg.config.run)
    
    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model
    
    # Set models to eval mode
    salmonn_preprocessor.eval().cuda()

    save_speech_encoder_trt(salmonn_preprocessor.speech_encoder)
    save_beats_trt(salmonn_preprocessor.beats)
    save_speech_Qformer_trt(salmonn_preprocessor.speech_Qformer)
    save_llm_trt(salmonn_preprocessor.llama_model)
    
    
    # Load data
    dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task)
    
 
if __name__ == '__main__':
    random.seed(42)
    main()