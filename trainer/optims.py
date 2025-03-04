# This script is from https://github.com/salesforce/LAVIS/blob/main/lavis/common/optims.py

import math
import logging

import torch
from apollo_torch import APOLLOAdamW


class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )


class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        if total_cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=total_cur_step,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch * self.iters_per_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def get_adamw(model, config):
    num_parameters = 0
    # weight decay 미적용 파라미터, weight decay 적용 파라미터 (weight decay = L2 정규화)
    p_wd, p_non_wd = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        # weight decay 미적용 파라미터 (bias, ln = layer norm, bn = batch norm)
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            # weight decay 적용 파라미터
            p_wd.append(p)

        num_parameters += p.data.nelement()

    logging.info("number of trainable parameters: %d" % num_parameters)
    optim_params = [
        {
            "params": p_wd,
            "weight_decay": float(config.weight_decay),
        },
        {"params": p_non_wd, "weight_decay": 0},
    ]
    beta2 = config.get("beta2", 0.999)

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(config.init_lr),
        weight_decay=float(config.weight_decay),
        betas=(0.9, beta2),
    )

    return optimizer    

def get_apollo(model, config):
    num_parameters = 0
    # weight decay 미적용 파라미터, weight decay 적용 파라미터 (weight decay = L2 정규화)

    p_wd_lowrank = []  # 저랭크 적용 + weight decay
    p_wd_nonlowrank = []  # 일반 적용 + weight decay
    p_non_wd = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        # weight decay 미적용 파라미터 (bias, ln = layer norm, bn = batch norm)
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            # 2. weight decay 적용 파라미터 중 저랭크 대상 분리
            # 예시: 'weight'를 포함하고 2D 이상인 경우 (예: Linear, Embedding)
            if 'weight' in n and p.ndim > 1:
                p_wd_lowrank.append(p)
            else:
                p_wd_nonlowrank.append(p)

        num_parameters += p.data.nelement()

    logging.info("number of trainable parameters: %d" % num_parameters)
    # 3. APOLLO를 위한 param_groups 설정
    param_groups = [
        # 일반 매개변수 (weight decay 적용)
        {
            "params": p_wd_nonlowrank,
            "weight_decay": float(config.weight_decay),
        },
        # 저랭크 매개변수 (APOLLO 설정 + weight decay)
        {
            "params": p_wd_lowrank,
            "weight_decay": float(config.weight_decay),
            "rank": int(config.rank),  # 저랭크 랭크
            "proj": "random",  # 투영 방식
            "scale_type": config.scale_type,  # 스케일링 방식
            "scale": int(config.scale),  # 스케일링 크기
            "update_proj_gap": int(config.update_proj_gap),  # 투영 업데이트 주기
            "proj_type": "std",  # 투영 타입
        },
        # weight decay 미적용 매개변수
        {
            "params": p_non_wd,
            "weight_decay": 0
        }
    ]
    beta2 = config.get("beta2", 0.999)

    # 4. APOLLOAdamW로 최적화기 생성
    optimizer = APOLLOAdamW(
        param_groups,
        lr=float(config.init_lr),
        betas=(0.9, beta2),
    )

    return optimizer

def get_optimizer(model, config):
    use_apollo = config.get("use_apollo", False)
    if use_apollo:
        optimizer = get_apollo(model, config)
    else:
        optimizer = get_adamw(model, config)
    return optimizer