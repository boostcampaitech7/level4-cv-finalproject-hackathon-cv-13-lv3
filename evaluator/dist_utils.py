"""
Adapted from salesforce@LAVIS. Below is the original copyright:
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import datetime
import functools

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    # os.environ에 RANK, WORLD_SIZE가 있으면 분산학습 환경이라고 판단, os.environ은 os 환경 변수에 접근 가능
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])        # 현재 프로세스 ID
        args.world_size = int(os.environ["WORLD_SIZE"])  # 총 GPU 개수
        args.gpu = int(os.environ["LOCAL_RANK"])   # 현재 GPU 번호
    elif "SLURM_PROCID" in os.environ:
        # SLURM 환경에서 사용되는 환경 변수 (대규모 컴퓨터 클러스터에서 분산 학습할 때 사용)
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count() # 사용 가능한 GPU 수로 나눈 나머지를 GPU ID로 설정
    else:
        print("Not using distributed mode")
        args.use_distributed = False
        return

    args.use_distributed = True

    torch.cuda.set_device(args.gpu) # gpu 설정
    # 분산 학습 백엔드 설정 (nccl: NVIDIA NCCL, NVIDIA가 개발한 GPU간 통신 라이브러리)
    # 각 GPU에서 gradient를 계산하고 NCCL이 이를 모아서 평균을 내 모든 GPU에 전달
    args.dist_backend = "nccl" 

    # 분산 학습 초기화 방법 설정
    # 각 GPU에서 동일한 초기화 방법을 사용하여 동일한 결과를 얻을 수 있도록 함
    print(
        "| distributed init (rank {}, world {}): {}".format(
            args.rank, args.world_size, args.dist_url
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend, # nccl
        # 분산 학습에서 각 GPU가 서로를 찾고 통신하는 방법
        # 지금은 env://로 환경 변수를 통한 초기화 방법
        init_method=args.dist_url, 
        world_size=args.world_size, # 총 GPU 개수
        rank=args.rank, # 현재 프로세스 ID
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    # 모든 GPU가 동일한 시점에 동일한 작업을 수행하도록 동기화
    torch.distributed.barrier() # 모든 GPU가 이 지점에 도달할 때까지 대기
    
    # 분산 학습 환경에서 각 GPU에서 같은 내용을 출력하지 않고 메인 프로세스에서만 출력하도록 함
    # setup_for_distributed(args.rank == 0) # 지금은 setup_logger()에서 처리하므로 주석 처리


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:  # non-distributed training
        rank = 0
        world_size = 1
    return rank, world_size


def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0: # 메인 프로세스에서만 실행
            return func(*args, **kwargs)

    return wrapper