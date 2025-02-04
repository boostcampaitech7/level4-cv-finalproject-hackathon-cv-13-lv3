# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

from utils import *
from config import Config
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset import SALMONNDataset
from runner import Runner
from accelerate_runner import AccelerateRunner

from models.Qformer import BertLayer

def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--dryrun", action='store_true', help='if True, use dummy model and skip forward/backward')
    parser.add_argument("--accelerate", action='store_true', help='if True, use accelerate')
    
    return parser.parse_args()


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now() # ex) 202403151205 현재 시각

    # load config
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets
    use_token_pruning = model_config.get("use_token_pruning", True)
    token_keep_rate = model_config.get("token_keep_rate", 0.7)
    print("Token pruning configuration: use_token_pruning={}, token_keep_rate={}".format(use_token_pruning, token_keep_rate))

    if args.accelerate:
        setup_seeds(run_config) # 랜덤 시드 설정
    else:
        init_distributed_mode(run_config) # 분산학습 환경 설정
        setup_seeds(run_config) # 랜덤 시드 설정
        setup_logger() # set after init_distributed_mode() to only log on master. 메인 GPU에서만 출력

        global_rank = int(os.environ["RANK"]) # 현재 GPU ID
        if global_rank == 0: # 메인 GPU에서만 실행
            wandb.login()
            wandb.init(project="audio_lm", name=run_config.exp_name)

    # print config
    cfg.pretty_print()

    # build datasets
    datasets = {
        "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
        "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
        "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
    }

    # build model
    if not args.dryrun:
        model = load_model(model_config)
    else: # load small dummy language model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)

    # build runner
    if args.accelerate:
        runner = AccelerateRunner(cfg, model, datasets, job_id, args.dryrun)
    else:
        runner = Runner(cfg, model, datasets, job_id, args.dryrun)

    # 목적: float32 행렬 곱셈 연산의 정밀도와 속도 사이의 트레이드오프를 조절합니다.

    # 옵션:
    # 'highest': 최고 정밀도 (기본값, 속도 ↓ / 정확도 ↑).
    #'high': Tensor Core를 활용한 중간 정밀도 (속도 ↑ / 정확도 약간 ↓).
    #'medium': 낮은 정밀도 (속도 ↑↑ / 정확도 ↓).
    torch.set_float32_matmul_precision('high')

    torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.benchmark = True  # 입력 크기가 고정된 경우
    # torch.backends.cudnn.deterministic = False  # 재현성보다 속도 우선
    # torch.backends.cuda.memory_efficient = True  # 메모리 효율적 SDPA 활성화

    # train
    runner.train()
    
    # Log parameters to Google Sheets
    Gsheet_param(cfg.config)

if __name__ == "__main__":
    main()    
