import argparse
import json
import random
import sys
import torch
import pandas as pd
import numpy as np
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
from evaluate_salmonn import get_dataset

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
        "in xxx=yyy format will be merged into config file",
        default=None
    )
    parser.add_argument("--mode", type=str, default="valid_aac", 
                    help="Mode to evaluate. Supports submission and validation modes for ASR and AAC tasks.", 
                    choices=['valid_asr', 'valid_aac'])
    
    args = parser.parse_args()
    
    if args.mode is None:
        raise ValueError("Either --mode must be provided")

    # Extract task from mode
    args.task = args.mode.split("_")[1]

    return args

def replace_test_ann_path(cfg):
    if "test_ann_path" not in cfg.config.datasets.keys():
        if args.task == "asr":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_asr
        elif args.task == "aac":
            cfg.config.datasets.test_ann_path = cfg.config.datasets.test_ann_path_aac
    return cfg

def register_hooks(layer_times, salmonn_preprocessor, llama_model, tokenizer):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    def hook_fn_start(module, input):
        starter.record()
        
    def hook_fn_end(name):
        def hook(module, input, output):
            if name not in layer_times:
                layer_times[name] = []
            ender.record()
            torch.cuda.synchronize()
            cur_time = starter.elapsed_time(ender)
            layer_times[name].append(cur_time)
        return hook
    
    # Register hooks for Whisper encoder
    for name, module in salmonn_preprocessor.speech_encoder.named_modules():
        if name != "layers":
            module.register_forward_pre_hook(hook_fn_start)
            module.register_forward_hook(hook_fn_end(f"speech_encoder_{name}"))
    
    for name, module in salmonn_preprocessor.speech_encoder.layers.named_modules():
        module.register_forward_pre_hook(hook_fn_start)
        module.register_forward_hook(hook_fn_end(f"speech_encoder_layers_{name}"))
    
    # Register hooks for layer norms and projections
    salmonn_preprocessor.ln_speech.register_forward_pre_hook(hook_fn_start)
    salmonn_preprocessor.ln_speech.register_forward_hook(hook_fn_end("ln_speech"))
    
    # Register hooks for BEATs
    for name, module in salmonn_preprocessor.beats.named_modules():
        if name not in ["encoder", "encoder.layers"]:
            module.register_forward_pre_hook(hook_fn_start)
            module.register_forward_hook(hook_fn_end(f"beats_{name}"))
    
    # Register hooks for BEATs encoder pos_conv
    for name, module in salmonn_preprocessor.beats.encoder.pos_conv.named_modules():
        if name:  # Skip empty name
            module.register_forward_pre_hook(hook_fn_start)
            module.register_forward_hook(hook_fn_end(f"beats_encoder_pos_conv_{name}"))
    
    # Register hooks for BEATs encoder layers
    for name, module in salmonn_preprocessor.beats.encoder.layers.named_modules():
        module.register_forward_pre_hook(hook_fn_start)
        module.register_forward_hook(hook_fn_end(f"beats_encoder_layers_{name}"))
    
    salmonn_preprocessor.ln_audio.register_forward_pre_hook(hook_fn_start)
    salmonn_preprocessor.ln_audio.register_forward_hook(hook_fn_end("ln_audio"))
    
    # Register hooks for Q-Former
    # Embeddings
    for name, module in salmonn_preprocessor.speech_Qformer.bert.embeddings.named_modules():
        if module is not None and name:  # Skip None components and empty name
            module.register_forward_pre_hook(hook_fn_start)
            module.register_forward_hook(hook_fn_end(f"qformer_embeddings_{name}"))
    
    # Encoder layers with detailed components
    for name, module in salmonn_preprocessor.speech_Qformer.bert.encoder.layer.named_modules():
        module.register_forward_pre_hook(hook_fn_start)
        module.register_forward_hook(hook_fn_end(f"qformer_encoder_layer_{name}"))
    
    salmonn_preprocessor.speech_llama_proj.register_forward_pre_hook(hook_fn_start)
    salmonn_preprocessor.speech_llama_proj.register_forward_hook(hook_fn_end("speech_llama_proj"))
    
    # Register hooks for LLaMA
    # Embedding
    llama_model.model.model.embed_tokens.register_forward_pre_hook(hook_fn_start)
    llama_model.model.model.embed_tokens.register_forward_hook(hook_fn_end("llama_embed_tokens"))
    
    # Layers
    for name, module in llama_model.model.model.layers.named_modules():
        module.register_forward_pre_hook(hook_fn_start)
        module.register_forward_hook(hook_fn_end(f"llama_layers_{name}"))
    
    # Final norm
    llama_model.model.model.norm.register_forward_pre_hook(hook_fn_start)
    llama_model.model.model.norm.register_forward_hook(hook_fn_end("llama_final_norm"))
    
    # LM head
    llama_model.model.lm_head.register_forward_pre_hook(hook_fn_start)
    llama_model.model.lm_head.register_forward_hook(hook_fn_end("llama_lm_head"))
    
    # Rotary embeddings
    llama_model.model.model.rotary_emb.register_forward_pre_hook(hook_fn_start)
    llama_model.model.model.rotary_emb.register_forward_hook(hook_fn_end("llama_rotary_emb"))
    
    # TODO: Tokenizer 구조확인하고 hook 등록하기 decoder를 사용 중

def save_layer_times(layer_times, output_dir="inference_times"):
    """Save layer timing information to JSON and CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate statistics
    timing_stats = {}
    for name, times in layer_times.items():
        timing_stats[name] = {
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "total_time": float(np.sum(times)),
            "calls": len(times)
        }
    
    # Group by component
    component_groups = {
        "speech_encoder": {},
        "beats": {},
        "qformer": {},
        "llama": {},
        "layer_norms": {},
        "projections": {}
    }
    
    for name, stats in timing_stats.items():
        if name.startswith("speech_encoder"):
            component_groups["speech_encoder"][name] = stats
        elif name.startswith("beats"):
            component_groups["beats"][name] = stats
        elif name.startswith("qformer"):
            component_groups["qformer"][name] = stats
        elif name.startswith("model"):
            component_groups["llama"][name] = stats
        elif name in ["ln_speech", "ln_audio"]:
            component_groups["layer_norms"][name] = stats
        elif name == "speech_llama_proj":
            component_groups["projections"][name] = stats
    
    # Save detailed JSON
    json_path = os.path.join(output_dir, "layer_times_detailed.json")
    with open(json_path, "w") as f:
        json.dump({"individual_layers": timing_stats, "component_groups": component_groups}, f, indent=2)
    
    # Print summary
    print("\nTiming Analysis Summary:")
    for component, layers in component_groups.items():
        if layers:
            total_time = sum(stats["total_time"] for stats in layers.values())
            print(f"\n{component} total time: {total_time:.2f}ms")
            print("Top 5 most time-consuming layers:")
            sorted_layers = sorted(layers.items(), key=lambda x: x[1]["total_time"], reverse=True)[:5]
            for name, stats in sorted_layers:
                print(f"  {name}: {stats['total_time']:.2f}ms ({stats['mean_time']:.2f}ms ± {stats['std_time']:.2f}ms per call)")

def main(args):
    cfg = Config(args)
    cfg = replace_test_ann_path(cfg)
    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model

    # Load data
    dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task)

    with open("audiolm-trainer/prompts/test_prompt.json", "r") as f:
        test_prompt = json.load(f)

    layer_times = {}
    register_hooks(layer_times, salmonn_preprocessor, llama_model, tokenizer)
    
    # Evaluation
    testset_ids, hyps, refs = [], [], []
    for samples in tqdm(dataloader):
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
            pad_token_id=llama_model.config.eos_token_id,
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
    
    # Save timing results
    save_layer_times(layer_times)
    
    # For more detailed view of model parameters
    print("\n=== Detailed WhisperModel Parameters ===")
    total_params = sum(p.numel() for p in salmonn_preprocessor.speech_encoder.parameters())
    trainable_params = sum(p.numel() for p in salmonn_preprocessor.speech_encoder.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n=== Detailed BEATs Parameters ===")
    total_params = sum(p.numel() for p in salmonn_preprocessor.beats.parameters())
    trainable_params = sum(p.numel() for p in salmonn_preprocessor.beats.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n=== Detailed Q-Former Parameters ===")
    total_params = sum(p.numel() for p in salmonn_preprocessor.speech_Qformer.parameters())
    trainable_params = sum(p.numel() for p in salmonn_preprocessor.speech_Qformer.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n=== Detailed LLM Model Parameters ===")
    total_params = sum(p.numel() for p in llama_model.parameters())
    trainable_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

if __name__ == '__main__':
    args = parse_args()
    random.seed(42)
    
    main(args)