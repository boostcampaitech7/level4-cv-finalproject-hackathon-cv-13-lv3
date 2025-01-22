import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.backends.cuda as cuda
import wandb

from utils import *
from config import Config
from models import load_model
from dataset import SALMONNDataset
from accelerate_runner import AccelerateRunner


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

    return parser.parse_args()


def setup_seeds(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # Generate unique job ID using current timestamp
    job_id = now()

    # load config
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    setup_seeds(run_config)
    setup_logger()

    # Initialize wandb
    wandb.login()
    wandb.init(project="audio_lm", name=run_config.exp_name)

    # Print config
    cfg.pretty_print()

    # Build datasets
    datasets = {
        "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
        "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
        "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
    }

    # Build model
    if not args.dryrun:
        model = load_model(model_config)
    else:  # Load small dummy language model for testing
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)

    # Build runner and start training
    runner = AccelerateRunner(cfg, model, datasets, job_id, args.dryrun)
    runner.train()


if __name__ == "__main__":
    main() 