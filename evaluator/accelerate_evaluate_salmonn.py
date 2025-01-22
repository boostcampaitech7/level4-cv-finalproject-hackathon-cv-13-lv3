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

from accelerate import Accelerator

# Add custom module path
sys.path.append(str(Path(__file__).parent / "audiolm-trainer"))

# Custom modules
from salmonn_utils import SALMONNTestDataset, load_preprocessor, load_model
from config import Config
from utils import get_dataloader, prepare_sample
from metrics import compute_wer, compute_spider
from inference_timer import InferenceTimer

def save_result(testset_ids, hyps, result_dir, mode):
    os.makedirs(result_dir, exist_ok=True)
    
    accelerator = Accelerator()
    
    # Save rank-specific results with sorting
    result_file = os.path.join(result_dir, f"{mode}_rank{accelerator.process_index}.csv")
    rank_df = pd.DataFrame({
        "testset_id": testset_ids, 
        "text": hyps
    })
    # Sort by testset_id to ensure consistent ordering
    rank_df = rank_df.sort_values('testset_id')
    rank_df.to_csv(result_file, index=False)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged_df = pd.DataFrame()
        for rank in range(accelerator.num_processes):
            rank_file = os.path.join(result_dir, f"{mode}_rank{rank}.csv")
            rank_df = pd.read_csv(rank_file)
            merged_df = pd.concat([merged_df, rank_df], ignore_index=True)
        
        # Remove duplicates and ensure ordering
        merged_df = merged_df.drop_duplicates(subset=['testset_id'], keep='first')
        merged_df = merged_df.sort_values('testset_id')
        
        final_file = os.path.join(result_dir, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{mode}.csv")
        merged_df.to_csv(final_file, index=False)
        
        # Cleanup
        for rank in range(accelerator.num_processes):
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
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file",
    )
    parser.add_argument("--mode", type=str, default="submission_asr", 
                    help="Mode to evaluate", 
                    choices=['submission_asr', 'submission_aac', 'valid_asr', 'valid_aac'])
    parser.add_argument("--timer", action='store_true', default=False,
                        help="If True, measure the time taken for inference.")

    args = parser.parse_args()
    args.task = args.mode.split("_")[1]
    args.make_submission = args.mode.split("_")[0] == "submission"

    return args

def get_dataset(dataset_cfg, run_cfg, task):
    testset = SALMONNTestDataset(
        dataset_cfg.prefix, dataset_cfg.test_ann_path, dataset_cfg.whisper_path, task
    )
    test_loader = get_dataloader(testset, run_cfg, is_train=False)
    return test_loader

def replace_test_ann_path(cfg, task):
    if "test_ann_path" not in cfg.config.datasets.keys():
        if task == "asr":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_asr
        elif task == "aac":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_aac
    return cfg

def register_hooks(preprocessor, llm):
    timer = InferenceTimer()
    models_to_profile = {
        'preprocessor': preprocessor,
        'llm': llm
    }
    for model_name, model in models_to_profile.items():
        timer.register_module_hooks(model, prefix=model_name)
    return timer

def main():
    args = parse_args()
    random.seed(42)

    # Initialize accelerator
    accelerator = Accelerator()

    cfg = Config(args)
    cfg = replace_test_ann_path(cfg, args.task)

    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)

    # Prepare models with accelerator
    salmonn_preprocessor, llama_model = accelerator.prepare(salmonn_preprocessor, llama_model)
    salmonn_preprocessor.llama_model = llama_model
        
    # Load data
    dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task)
    dataloader = accelerator.prepare(dataloader)

    with open("audiolm-trainer/prompts/test_prompt.json", "r") as f:
        test_prompt = json.load(f)

    if args.timer:
        timer = register_hooks(salmonn_preprocessor, llama_model)

    # Evaluation
    testset_ids, hyps, refs = [], [], []
    for samples in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        testset_id = samples["testset_id"]
        testset_ids.extend(testset_id)

        # Preprocess
        samples = prepare_sample(samples, cuda_enabled=accelerator.device.type == "cuda")
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
        with accelerator.autocast():
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

    accelerator.wait_for_everyone()

    # Gather results from all processes
    if accelerator.num_processes > 1:
        testset_ids = accelerator.gather(testset_ids).tolist()
        hyps = accelerator.gather(hyps).tolist()
        if not args.make_submission:
            refs = accelerator.gather(refs).tolist()

    # Save results
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
                'preprocessor': salmonn_preprocessor,
                'llm': llama_model
            }
        )

if __name__ == '__main__':
    main() 