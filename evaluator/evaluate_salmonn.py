import argparse
import json
import random
import sys
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import time
# Add custom module path
sys.path.append(str(Path(__file__).parent / "audiolm-trainer"))

# Custom modules
from salmonn_utils import SALMONNTestDataset, load_preprocessor, load_model
from config import Config
from utils import get_dataloader, prepare_sample
from metrics import compute_wer, compute_spider
from inference_timer import InferenceTimer

from dist_utils import get_rank, init_distributed_mode, is_dist_avail_and_initialized, get_world_size, is_main_process

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def save_result(testset_ids, hyps, result_dir, mode):
    os.makedirs(result_dir, exist_ok=True)
    
    # Save rank-specific results with sorting
    result_file = os.path.join(result_dir, f"{mode}_rank{get_rank()}.csv")
    rank_df = pd.DataFrame({
        "testset_id": testset_ids, 
        "text": hyps
    })
    # Sort by testset_id to ensure consistent ordering
    rank_df = rank_df.sort_values('testset_id')
    rank_df.to_csv(result_file, index=False)

    if is_dist_avail_and_initialized():
        dist.barrier()

    if is_main_process():
        merged_df = pd.DataFrame()
        for rank in range(get_world_size()):
            rank_file = os.path.join(result_dir, f"{mode}_rank{rank}.csv")
            rank_df = pd.read_csv(rank_file)
            merged_df = pd.concat([merged_df, rank_df], ignore_index=True)
        
        # Remove duplicates and ensure ordering
        merged_df = merged_df.drop_duplicates(subset=['testset_id'], keep='first')
        merged_df = merged_df.sort_values('testset_id')
        
        final_file = os.path.join(result_dir, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{mode}.csv")
        merged_df.to_csv(final_file, index=False)
        
        # Cleanup
        for rank in range(get_world_size()):
            rank_file = os.path.join(result_dir, f"{mode}_rank{rank}.csv")
            os.remove(rank_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path", 
        type=str, 
        help='path to configuration file', 
        default='salmonn_eval_config.yaml'
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # --- Deprecated options ---
    parser.add_argument("--task", type=str, default=None, 
                    help="(Deprecate) Task to evaluate. Use --mode instead. This option will be removed in a future version.", 
                    choices=['asr', 'aac'])

    parser.add_argument("--skip_scoring", action='store_false', default=True, 
                    help="(Deprecate) If True, skip scoring after inference. Use --mode instead. This option will be removed in a future version.")
    # --- Deprecated options end ---
    # --- New options ---
    parser.add_argument("--mode", type=str, default="submission_asr", 
                    help="Mode to evaluate. Supports submission and validation modes for ASR and AAC tasks.", 
                    choices=['submission_asr', 'submission_aac', 'valid_asr', 'valid_aac'])
    parser.add_argument("--timer", action='store_true', default=False,
                        help="If True, measure the time taken for inference.")
    # --- New options end ---

    args = parser.parse_args()

    if args.mode is None:
        # --- For Previous Version ---
        if args.task is None:
            raise ValueError("Either --task or --mode must be provided")
        args.mode = convert_task_to_mode(args.task, args.skip_scoring)

    # --- Override Previous Version Args ---
    args.task = args.mode.split("_")[1]
    args.make_submission = args.mode.split("_")[0] == "submission"

    return args

def convert_task_to_mode(task, skip_scoring):
    if skip_scoring:
        if task == 'asr':
            return 'submission_asr'
        elif task == 'aac':
            return 'submission_aac'
    else:
        if task == 'asr':       
            return 'valid_asr'
        elif task == 'aac':
            return 'valid_aac'
    
    raise ValueError(f"Invalid task: {task} | {skip_scoring}")

def get_dataset(dataset_cfg, run_cfg, task, use_distributed):
    testset = SALMONNTestDataset(
        dataset_cfg.prefix, dataset_cfg.test_ann_path, dataset_cfg.whisper_path, task
    )
    test_loader = get_dataloader(testset, run_cfg, is_train=False, use_distributed=use_distributed)
    return test_loader

def replace_test_ann_path(cfg):
    if "test_ann_path" not in cfg.config.datasets.keys():
        if args.task == "asr":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_asr
        elif args.task == "aac":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_aac
    return cfg

def register_hooks(preprocessor, llm):
    timer = InferenceTimer()
    
    # Register hooks for all models
    models_to_profile = {
        'preprocessor': preprocessor,
        'llm': llm
    }
    
    for model_name, model in models_to_profile.items():
        timer.register_module_hooks(model, prefix=model_name)
    
    return timer

def main(args):

    cfg = Config(args)
    cfg = replace_test_ann_path(cfg)
    init_distributed_mode(cfg.config.run) # 분산학습 환경 설정

    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)

    if cfg.config.run.use_distributed == True:
        llama_model = DDP(llama_model, device_ids=[cfg.config.run.gpu]).module

    salmonn_preprocessor.llama_model = llama_model

        
    # Load data
    dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task, cfg.config.run.use_distributed)

    with open("audiolm-trainer/prompts/test_prompt.json", "r") as f:
        test_prompt = json.load(f)

    if args.timer:
        timer = register_hooks(salmonn_preprocessor, llama_model)

    # Evaluation
    testset_ids, hyps, refs = [], [], []
    idx = 0
    for samples in tqdm(dataloader):
        idx += 1
        if idx > 10:
            break
        testset_id = samples["testset_id"]
        testset_ids.extend(testset_id)

        # Preprocess
        samples = prepare_sample(samples, cuda_enabled=torch.cuda.is_available())
        batch_size = samples["spectrogram"].shape[0]
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)
        speech_embeds, speech_atts = salmonn_preprocessor.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # Add prompt embeds + audio embed 
        prompts = [test_prompt[task] for task in samples['task']]
        templated_prompts = [cfg.config.model.prompt_template.format(prompt) for prompt in prompts]

        speech_embeds, speech_atts = salmonn_preprocessor.prompt_wrap(speech_embeds, speech_atts, templated_prompts, multi_prompt=True)
        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * tokenizer.bos_token_id

        bos_embeds = llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        generate_cfg = cfg.config.generate

        # Generation
        outputs = llama_model.model.generate(
            inputs_embeds=embeds,
            pad_token_id=llama_model.config.eos_token_id[0],
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )

        results = tokenizer.batch_decode(outputs)
        hyp = [result.split(generate_cfg.end_sym)[0].lower() for result in results]
        hyps.extend(hyp)

        if not args.make_submission:
            ref = samples["text"]
            refs.extend(ref)

    if is_dist_avail_and_initialized():
            dist.barrier()       

    # In main code:
    if args.make_submission:
        save_result(testset_ids, hyps, "submission_results", args.mode)
    else:
        if args.task == 'asr':
            compute_wer(hyps, refs)
        elif args.task == 'aac':
            compute_spider(hyps, refs)
        save_result(testset_ids, hyps, "valid_results", args.mode)
    
    if args.timer:
        timer.save_measurement(
            output_dir="inference_times", 
            models={
                'preprocessor': salmonn_preprocessor,  # preprocessor 전체를 전달
                'llm': llama_model  # llm 모델 전달
            }
        )

if __name__ == '__main__':
    args = parse_args()

    random.seed(42)
    main(args)