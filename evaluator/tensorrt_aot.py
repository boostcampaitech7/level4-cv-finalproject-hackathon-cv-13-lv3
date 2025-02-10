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
from salmonn_utils import load_preprocessor, load_model
from dataset import SALMONNDataset
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
    
class CalibrationDataloader(torch.utils.data.DataLoader):
    def __init__(self, salmonn, dataloader, cfg):
        self.dataloader = iter(dataloader)
        self.salmonn = salmonn
        self.cfg = cfg
        self.max_samples = 30
        self.device = torch.device(cfg.config.run.tensorrt_device)
        self.batch_size = cfg.config.run.batch_size_eval
        self.persistent_workers = True
        
        # test_prompt 로드
        with open("audiolm-trainer/prompts/test_prompt.json", "r") as f:
            self.test_prompt = json.load(f)
            
        # 미리 데이터 준비
        with torch.no_grad():
            for _ in range(self.max_samples):
                try:
                    samples = next(self.dataloader)
                    processed = self._process_sample(samples)
                    self.samples.append(processed)
                except StopIteration:
                    break
                
    def _process_sample(self, samples):
        # evaluate_efficiency_salmonn.py의 model_inference 함수와 동일한 전처리
        batch_size = samples["spectrogram"].shape[0]
        # float16으로 변환하고 디바이스로 이동
        spectrogram = samples["spectrogram"].to(dtype=torch.float16, device=self.device)
        raw_wav = samples.get("raw_wav", None)
        if raw_wav is not None:
            raw_wav = raw_wav.to(self.device)
        audio_padding_mask = samples.get("padding_mask", None)
        if audio_padding_mask is not None:
            audio_padding_mask = audio_padding_mask.to(self.device)
        
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            speech_embeds, speech_atts = self.salmonn.encode_speech(
                spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask
            )
            
        # Prompt wrapping
        prompts = [self.test_prompt[task] for task in samples["task"]]
        templated_prompts = [
            self.cfg.config.model.prompt_template.format(prompt) for prompt in prompts
        ]
        
        speech_embeds, speech_atts = self.salmonn.prompt_wrap(
            speech_embeds, speech_atts, templated_prompts, multi_prompt=True
        )

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.salmonn.llama_tokenizer.bos_token_id
        
        bos_embeds = self.salmonn.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        speech_embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        speech_atts = torch.cat([atts_bos, speech_atts], dim=1)
        
        return [speech_embeds, speech_atts]
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

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

def get_dataset(dataset_cfg, run_cfg):
    dataset = SALMONNDataset(
        dataset_cfg.prefix, 
        dataset_cfg.valid_ann_path, 
        dataset_cfg.whisper_path
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=run_cfg.batch_size_eval,
        num_workers=run_cfg.num_workers,
        pin_memory=True,
        collate_fn=dataset.collater,
        drop_last=False,
        shuffle=False,
    )
    return loader

def save_speech_encoder_trt(speech_encoder, config):
    # spectogram shape (8, 128, 3000), dtype=torch.float32
    sample_input = torch.randn(config.batch_size_eval, 128, 3000, dtype=torch.float16, device=torch.device(config.tensorrt_device))
    
    # 워밍업
    with torch.no_grad():
        for _ in range(3):
            speech_encoder(sample_input)
    
    # 컴파일용 입력 정의
    compile_inputs = [
        torch_tensorrt.Input(
            shape=[config.batch_size_eval, 128, 3000], 
            dtype=torch.float16,
            device=torch.device(config.tensorrt_device),
            format=torch.contiguous_format
    )]
    
    speech_encoder = torch_tensorrt.compile(
        speech_encoder,
        ir="dynamo",
        inputs=compile_inputs,
        enabled_precisions={torch.float16, torch.float32},  # float32도 사용
        optimization_level=config.optimization_level,
        use_explicit_typing=True,
        strict_types=True,
        device=torch.device(config.tensorrt_device),
        debug=False, # 프로덕션 환경에서는 False로 설정
        # 추가 옵션
        truncate_long_and_double=True,
        require_full_compilation=True,
        # use_fast_partitioner=False # 더 빠른 추론이 가능해지나 compile이 더 오래걸림
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
    # validate_model_kwargs 메소드를 무효화
    llm._validate_model_kwargs = lambda x: None
    
    # Wrap the model
    llm = LLMWrapper(llm)
    llm = llm.eval().cuda()

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
        enabled_precisions={torch.float16, torch.float32}, # float32도 사용
        optimization_level=config.optimization_level,
        use_explicit_typing=True,
        strict_types=True,
        device=torch.device(config.tensorrt_device),
        debug=False, # 프로덕션 환경에서는 False로 설정
        # 추가 옵션
        truncate_long_and_double=True,
        # use_fast_partitioner=False # 더 빠른 추론이 가능해지나 compile이 더 오래걸림
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


def save_ptq_llm(salmonn, dataloader, config):
    torch.cuda.empty_cache()
    batch_size = config.config.run.batch_size_eval
    tensorrt_device = config.config.run.tensorrt_device
    llm = salmonn.llama_model
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
            shape=[batch_size, seq_len, hidden_size],
            dtype=torch.float16,
            device=torch.device(tensorrt_device)
        ),
        torch_tensorrt.Input(
            shape=[batch_size, seq_len],
            dtype=torch.bool,
            device=torch.device(tensorrt_device),
            optional=True
        )
    ]
    
    # 캘리브레이션 데이터셋 생성
    calibration_dataloader = CalibrationDataloader(salmonn, dataloader, config)
    
    calibrator = torch_tensorrt.ts.ptq.DataLoaderCalibrator(
        calibration_dataloader,
        cache_file="./calibration.cache",
        use_cache=False,
        algo_type=torch_tensorrt.ts.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device(tensorrt_device)
    )
    
    # TensorRT 컴파일 설정에 양자화 관련 파라미터 추가
    llm = torch_tensorrt.compile(
        llm,
        ir="dynamo",
        inputs=compile_inputs,
        enabled_precisions={torch.int8, torch.float16, torch.float32},  # int8 추가
        optimization_level=config.optimization_level,
        use_explicit_typing=True,
        strict_types=True,
        device=torch.device(tensorrt_device),
        debug=False, # 프로덕션 환경에서는 False로 설정
        # 추가 옵션
        calibrator=calibrator,
        truncate_long_and_double=True,
        # use_fast_partitioner=False # 더 빠른 추론이 가능해지나 compile이 더 오래걸림
    )

    # 실제 저장용 입력 생성
    inputs = [
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=torch.device(tensorrt_device)),
        torch.ones(batch_size, seq_len, dtype=torch.bool, device=torch.device(tensorrt_device))
    ]

    torch_tensorrt.save(
        llm, 
        f"./trt_models/llm_ptq_trt_batch{batch_size}.ts", 
        inputs=inputs,
        output_format="torchscript"
    )

    del llm
    torch.cuda.empty_cache()

def save_aot_models(speech_encoder, llm, config):
    try:
        # save_speech_encoder_trt(speech_encoder, config)
        save_llm_trt(llm, config)
    except Exception as e:
        print(f"Failed to save TensorRT models: {e}")
        raise

def main(args):
    # Config 객체 생성 전에 CUDA 디바이스 설정
    cfg = Config(args)
    torch.cuda.set_device(int(cfg.config.run.tensorrt_device.split(':')[1]))
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    torch_tensorrt.runtime.set_cudagraphs_mode(True)

    setup_seeds(cfg.config.run)
    
    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model
    salmonn_preprocessor.eval()
    
    # Load data - 한 번만 생성
    dataset = get_dataset(cfg.config.datasets, cfg.config.run)
    
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
    
    # save_ptq_llm(
    #     salmonn_preprocessor,
    #     dataset,  # 여기서 calibration_dataloader 사용
    #     cfg
    # )
 
if __name__ == '__main__':
    args = parse_args()
    random.seed(42)
    main(args)
