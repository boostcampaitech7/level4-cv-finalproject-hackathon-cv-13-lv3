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

def replace_test_ann_path(cfg, task):
    if "test_ann_path" not in cfg.config.datasets.keys():
        if task == "asr":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_asr
        elif task == "aac":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_aac
    return cfg

def load_aot_models(config):
    
    # 모델 로드 전에 디렉토리 확인
    if not os.path.exists("./trt_models"):
        raise FileNotFoundError("TensorRT models directory not found. Please run tensorrt_aot.py first.")
        
    try:
        speech_path = f"./trt_models/speech_trt_batch{config.batch_size_eval}.ep"
        llm_path = f"./trt_models/llm_trt_batch{config.batch_size_eval}.ts"
        
        speech_encoder = torch_tensorrt.load(speech_path).module()
        llm = torch.jit.load(llm_path).cuda()
        
        return speech_encoder, llm
    except Exception as e:
        print(f"Error loading TensorRT models: {e}")
        raise e

def main():
    args = parse_args()
    cfg = Config(args)
    cfg = replace_test_ann_path(cfg, args.task)
    
    setup_seeds(cfg.config.run)
    
    # Accelerator 초기화
    accelerator = Accelerator()
    
    # Runtime 설정
    torch_tensorrt.runtime.set_multi_device_safe_mode(True)
    # torch_tensorrt.runtime.set_cudagraphs_mode(True)  # 런타임 CUDA 그래프만 비활성화 (Single GPU 모드에서 사용?)
    
    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model
    salmonn_preprocessor.eval()
    
    if cfg.config.run.tensorrt:
        speech_encoder, trt_llm = load_aot_models(cfg.config.run)
            
        # 새로운 모델 할당
        salmonn_preprocessor.speech_encoder = speech_encoder

        # salmonn_preprocessor.llama_model.forward = trt_llm.forward
        
        del trt_llm, speech_encoder
        torch.cuda.empty_cache()
        
    # Load data
    dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task)
    
    # Prepare with accelerator
    encode_speech, prompt_wrap, tokenizer, embed_tokens, llama_model, dataloader = accelerator.prepare(
        salmonn_preprocessor.encode_speech, 
        salmonn_preprocessor.prompt_wrap, 
        tokenizer, 
        salmonn_preprocessor.embed_tokens, 
        salmonn_preprocessor.llama_model, 
        dataloader
    )
    
    with open("audiolm-trainer/prompts/test_prompt.json", "r") as f:
        test_prompt = json.load(f)

    # Evaluation loop with torch.no_grad()
    with torch.no_grad():
        local_testset_ids = []
        local_hyps = []
        local_refs = []
        
        # autocast를 loop 시작 전에 한 번만 적용
        with torch.inference_mode(), accelerator.autocast():
            for samples in tqdm(dataloader, disable=not accelerator.is_local_main_process):
                # Preprocess
                batch_size = samples["spectrogram"].shape[0]
                spectrogram = samples["spectrogram"].half()
                raw_wav = samples.get("raw_wav", None)
                audio_padding_mask = samples.get("padding_mask", None)
            
                # Encode speech
                speech_embeds, speech_atts = encode_speech(
                    spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
                )
            
                # Add prompt embeds + audio embed 
                prompts = [test_prompt[task] for task in samples['task']]
                templated_prompts = [cfg.config.model.prompt_template.format(prompt) for prompt in prompts]

                speech_embeds, speech_atts = prompt_wrap(
                    speech_embeds, speech_atts, templated_prompts, multi_prompt=True
                )

                bos = torch.ones(
                    [batch_size, 1],
                    dtype=torch.int32,
                    device=speech_embeds.device,
                ) * tokenizer.bos_token_id

                bos_embeds = embed_tokens(bos)
                atts_bos = speech_atts[:, :1]
            
                embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
                attns = torch.cat([atts_bos, speech_atts], dim=1)

                generate_cfg = cfg.config.generate
            
                outputs = llama_model.generate(
                    inputs_embeds=embeds,
                    pad_token_id=llama_model.config.eos_token_id[0],
                    max_new_tokens=generate_cfg.get("max_new_tokens", 200),
                    num_beams=generate_cfg.get("num_beams", 4),
                    do_sample=generate_cfg.get("do_sample", True),
                    min_length=generate_cfg.get("min_length", 1),
                    temperature=generate_cfg.get("temperature", 0.7),
                    top_p=generate_cfg.get("top_p", 0.9),
                    repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
                    length_penalty=generate_cfg.get("length_penalty", 1.0),
                    attention_mask=attns,
                )

                # 로컬에 결과 저장 (gather 없이)
                decoded_results = tokenizer.batch_decode(outputs)
                local_hyps.extend([result.split(generate_cfg.end_sym)[0].lower() for result in decoded_results])
                local_testset_ids.extend(samples["testset_id"])
                
                if not args.make_submission:
                    local_refs.extend(samples["text"])
        
        # 모든 프로세스의 처리가 끝난 후 한번에 gather
        accelerator.wait_for_everyone()
        
        # 디버깅을 위한 출력
        if accelerator.is_local_main_process:
            print(f"Local process collected: {len(local_testset_ids)} samples")
        
        # 전체 결과를 한번에 gather
        all_testset_ids = accelerator.gather_for_metrics(local_testset_ids)
        all_hyps = accelerator.gather_for_metrics(local_hyps)
        if not args.make_submission:
            all_refs = accelerator.gather_for_metrics(local_refs)
        
        # 디버깅을 위한 출력
        if accelerator.is_local_main_process:
            print(f"After gathering: {len(all_testset_ids)} samples")
            print(f"Hyps length: {len(all_hyps)}")
            
            # 결과 저장
            result_df = pd.DataFrame({
                "testset_id": all_testset_ids,
                "text": all_hyps
            })
            result_df.drop_duplicates(subset=['testset_id'], keep='first', inplace=True)
            result_df.sort_values(by="testset_id", inplace=True)
            
            def save_result(result_dir, mode):
                os.makedirs(result_dir, exist_ok=True)
                final_file = os.path.join(result_dir, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{mode}.csv")
                result_df.to_csv(final_file, index=False)
            
            if args.make_submission:
                save_result("submission_results", "submission")
            else:
                if args.task == 'asr':
                    compute_wer(all_hyps, all_refs)
                elif args.task == 'aac':
                    compute_spider(all_hyps, all_refs)
                save_result("valid_results", "valid")

    # 프로세스 그룹 정리
    accelerator.end_training()

if __name__ == '__main__':
    random.seed(42)
    main()

