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
from tqdm import tqdm
from accelerate import Accelerator
from torch_tensorrt.dynamo import refit_module_weights
from torch.export import export
# from torch_tensorrt.ptq import DataLoaderCalibrator, CalibrationAlgo

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
    inputs = [
        # torch_tensorrt.Input(
        #     min_shape=[1, 128, 3000],      # [batch_size, mel_bins, max_time]
        #     opt_shape=[8, 128, 3000],     # Whisper의 기본 입력 크기
        #     max_shape=[16, 128, 3000],     # 최대 시퀀스 길이
        #     dtype=torch.float16,
        #     device=torch.device("cuda:0")
        # )
        torch_tensorrt.Input(
            shape=[8, 128, 3000],
            dtype=torch.float32,
            device=torch.device("cuda:0")
        )
    ]
    
    enabled_precisions = {torch.float16, torch.float32}
    
    speech_encoder = torch_tensorrt.compile(
        speech_encoder,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        use_explicit_typing=True,
        immutable_weights=False,
        make_refittable=True,
        optimization_level=1,
        truncate_long_and_double=True,
        heuristic_mode=False,
        device=torch.device("cuda:0")
    )
    torch_tensorrt.save(speech_encoder, "./trt_models/speech_trt.ep", inputs=inputs)

    del speech_encoder
    torch.cuda.empty_cache()
    
def save_beats_trt(beats):
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 1, 320, 320],  # [batch_size, channels, height, width]
            opt_shape=[16, 1, 320, 320],  # BEATs의 기본 입력 크기
            max_shape=[32, 1, 320, 320],  # patch_embedding: Conv2d(1, 512, kernel_size=(16, 16))
            dtype=torch.half,            # 320x320 -> (20x20) patches
        )
    ]
    
    enabled_precisions = {torch.float16}
    
    beats = torch_tensorrt.compile(
        beats,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        use_explicit_typing=True,
        immutable_weights=False,
    )
    torch_tensorrt.save(beats, "./trt_models/beats_trt.ep", inputs=inputs)
    
    del beats
    torch.cuda.empty_cache()

def save_speech_Qformer_trt(speech_Qformer):
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 32, 768],  # [batch_size, num_query_tokens, hidden_size]
            opt_shape=[16, 32, 768],  # BertConfig.hidden_size = 768
            max_shape=[32, 32, 768],  # num_query_tokens = 32 (from init_speech_Qformer)
            dtype=torch.half,
        ),
        torch_tensorrt.Input(
            min_shape=[1, 94, 1280],  # [batch_size, sequence_length, encoder_hidden_size]
            opt_shape=[16, 94, 1280],  # Whisper encoder output: 1280 dim
            max_shape=[32, 94, 1280],  # seq_len = 1500/16 ≈ 94 (after conv layers)
            dtype=torch.half,
        )
    ]
    
    enabled_precisions = {torch.float16}
    
    speech_Qformer = torch_tensorrt.compile(
        speech_Qformer,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        use_explicit_typing=True,
        immutable_weights=False,
    )   
    torch_tensorrt.save(speech_Qformer, "./trt_models/qformer_trt.ep", inputs=inputs)
    
    del speech_Qformer
    torch.cuda.empty_cache()

def save_llm_trt(llm):
    if hasattr(llm, "merge_and_unload"):
        llm = llm.merge_and_unload()
    
    llm.config.use_cache = False
    llm.half()
    torch.cuda.empty_cache()
    
    batch_size = 8  # 로그에서 확인된 실제 배치 크기
    seq_len = 111    # 로그에서 확인된 실제 시퀀스 길이
    hidden_size = 3072  # 로그에서 확인된 임베딩 크기
    
    inputs = [
        # inputs_embeds
        torch_tensorrt.Input(
            shape=[batch_size, seq_len, hidden_size],
            dtype=torch.float16,  # 로그에서 확인된 dtype
            device=torch.device("cuda:0"),
            name="inputs_embeds"
        ),
        torch_tensorrt.Input(
            shape=[batch_size, seq_len],
            dtype=torch.int64,  # 로그에서 확인된 dtype
            device=torch.device("cuda:0"),
            name="attention_mask"
        ),
    ]
    # calibrator = torch_tensorrt.ptq.Calibrator(...) # 양자화 callibrator
    
    llm = torch_tensorrt.compile(
        llm,
        ir="dynamo",
        inputs=inputs,
        use_python_runtime=False,
        enabled_precisions={torch.float16},
        # enabled_precisions_to_force={torch.float16, torch.float32},
        force_fp32_layers=["LayerNorm"],  # LayerNorm은 FP32로 강제
        dynamic_batching = True,# use_dynamic_shape=True, # 
        # use_explicit_typing=False,
        # immutable_weights=False,
        # make_refittable=True,
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

    # torch.cuda.empty_cache()
        
    # # Create directory
    # os.makedirs("trt_models", exist_ok=True)
    
    # speech_encoder = salmonn_preprocessor.speech_encoder.eval().cuda()
    # save_speech_encoder_trt(speech_encoder)
    # del speech_encoder
    # torch.cuda.empty_cache()
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
    
    # Load data
    # dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task)
    
 
if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    random.seed(42)
    main()