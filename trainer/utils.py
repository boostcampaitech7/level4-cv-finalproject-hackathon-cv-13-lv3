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

import logging
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler
import soundfile as sf
import numpy as np
import librosa

from dist_utils import is_main_process, get_world_size, get_rank

import numpy as np

import torch

import os

import gspread
from gspread.exceptions import WorksheetNotFound
from gspread_formatting import *
from dotenv import dotenv_values

def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")


def setup_logger():
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

def setup_acceletate_logger(accelerate, log_level_warning):
    log_level = logging.INFO if accelerate.is_local_main_process and not log_level_warning else logging.WARN
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    
def get_dataloader(dataset, config, is_train=True, use_distributed=True):
    # 분산 학습을 사용하면
    if use_distributed:
        # DistributedSampler: 데이터를 분산 학습 환경에 맞게 섞기
        sampler = DistributedSampler(
            dataset,
            shuffle=is_train,          # 매 epoch마다 데이터 섞기
            num_replicas=get_world_size(),  # 총 GPU 수
            rank=get_rank()            # 현재 GPU ID
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train if is_train else config.batch_size_eval,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=sampler,  # 여기서 sampler 사용
        shuffle=sampler is None and is_train,  # sampler가 없을 때만 shuffle
        collate_fn=dataset.collater,
        drop_last=is_train,
    )

    if is_train:
        # IterLoader: 데이터를 무한 반복하는 이터레이터 생성
        # 일반적인 데이터 로더는 데이터를 다 쓰면 종료됨
        # IterLoader는 데이터를 다 쓰면 종료되지 않고 원하는 iteration만큼 무한 반복
        loader = IterLoader(loader, use_distributed=use_distributed)

    return loader


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
        
    1 iteration = batch_size개의 데이터를 처리
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader) # 데이터 로더를 iterator로 변환
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader) # 데이터 로더에서 다음 데이터 가져오기
        except StopIteration: # 데이터 로더에서 데이터를 모두 사용했다면
            self._epoch += 1 # 에폭 증가
            # 분산 학습을 사용하면
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                # 각 GPU가 다른 데이터를 사용하도록 설정
                self._dataloader.sampler.set_epoch(self._epoch) # 분산 학습 시 매 에폭마다 shuffle이 제대로 작동하기 위함
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            
            # 다시 처음부터 데이터를 사용하도록 설정
            self.iter_loader = iter(self._dataloader) # 데이터 로더를 iterator로 변환
            data = next(self.iter_loader) # 데이터 로더에서 다음 데이터 가져오기

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


def prepare_one_sample(wav_path, wav_processor, cuda_enabled=True):
    audio, sr = sf.read(wav_path)
    if len(audio.shape) == 2: # stereo to mono
        audio = audio[:, 0]
    if len(audio) < sr: # pad audio to at least 1s
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)

    if sr != wav_processor.sampling_rate: # TODO. use more efficient implementation            
        audio = librosa.resample(audio, orig_sr=sr, target_sr=wav_processor.sampling_rate)
        sr = wav_processor.sampling_rate

    audio = audio[: sr * 30] # truncate audio to at most 30s

    spectrogram = wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]

    samples = {
        "spectrogram": spectrogram,
        "raw_wav": torch.from_numpy(audio).unsqueeze(0),
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
    }
    if cuda_enabled:
        samples = move_to_cuda(samples)

    return samples

def Gsheet_param(cfg):
    # env 파일 불러오기
    env_path = "/data/env/.env"
    env = dotenv_values(env_path)

    # 서비스 연결
    gc = gspread.service_account(env['JSON_PATH'])

    # url에 따른 spread sheet 열기
    doc = gc.open_by_url(env['URL'])

    # 저장할 변수 dict 선언
    param_dict = dict()

    # User 명
    param_dict['user'] = os.path.abspath(__file__).split("/")[2]

    # for idx, (key, value) in enumerate(cfg.items()):
    #     if idx < 4:

    #         pass
    #     else :
    param_dict['backbone'] = cfg.model.llama_path
    param_dict['Encoder_Speech'] = cfg.model.whisper_path
    param_dict['Encoder_Audio'] = cfg.model.beats_path
    param_dict['lora_rank'] = cfg.model.lora_rank
    param_dict['lora_alpha'] = cfg.model.lora_alpha
    param_dict['lora_dropout'] = cfg.model.lora_dropout
    param_dict['output_dir'] = cfg.run.output_dir
    param_dict['exp_name'] = cfg.run.exp_name
    param_dict['batch_size_train'] = cfg.run.exp_name
    param_dict['max_epoch'] = cfg.run.optims.max_epoch
    param_dict['warmup_steps'] = cfg.run.optims.warmup_steps
    param_dict['warmup_start_lr'] = cfg.run.optims.warmup_start_lr
    param_dict['init_lr'] = cfg.run.optims.init_lr
    param_dict['min_lr'] = cfg.run.optims.min_lr


    # sheet에 추가하기 위해서 값들을 list로 저장
    params = [param_dict[k] for k in param_dict]

    # sheet가 없는 경우 Head Row를 구성하기 위해서 Col 명을 list로 저장
    cols = [k.capitalize() for k in param_dict]
    
    try:
        # 워크시트가 있는지 확인
        worksheet = doc.worksheet(cfg.project_name)
    except WorksheetNotFound:
        # 워크시트가 없으면 새로 생성
        worksheet = doc.add_worksheet(title=cfg.project_name, rows="1000", cols="30")
        # Col 명 추가
        worksheet.append_rows([cols])

        # Header Cell 서식 
        header_formatter = CellFormat(
            backgroundColor=Color(0.9, 0.9, 0.9),
            textFormat=TextFormat(bold=True, fontSize=12),
            horizontalAlignment='CENTER',
        )
        
        # Header의 서식을 적용할 범위
        header_range = f"A1:{chr(ord('A') + len(cols) - 1)}1"

        # Header 서식 적용
        format_cell_range(worksheet, header_range, header_formatter)

        # Header Cell의 넓이 조정
        for idx, header in enumerate(cols):
            column_letter = chr(ord('A') + idx)
            width = max(len(header)*10+20,80)
            set_column_width(worksheet, column_letter, width)

        print(f"'{cfg.project_name}' 워크시트가 생성되었습니다.")

    # 실험 인자를 작성한 worksheet
    worksheet = doc.worksheet(cfg.project_name)

    # 실험 인자 worksheet에 추가
    worksheet.append_rows([params])

    # 현재 작성하는 실험 인자들 Cell의 서식
    # 노란색으로 하이라이트
    row_formatter = CellFormat(
        backgroundColor=Color(1, 1, 0),
        textFormat=TextFormat(fontSize=12),
        horizontalAlignment="CENTER"
    )

    # 이전 작성 실험인자들 배경색 원상복구
    rollback_formatter = CellFormat(
        backgroundColor=Color(1.0, 1.0, 1.0)
    )
    
    # 마지막 줄에만 하이라이팅이 들어가야 하므로 마지막 row 저장
    last_row = len(worksheet.get_all_values())
    row_range = f"A{last_row}:{chr(ord('A') + len(cols) - 1)}{last_row}"
    rollback_range = f"A{last_row - 1}:{chr(ord('A') + len(cols) - 1)}{last_row - 1}"
    
    # 헤더셀의 서식이 초기화되는 것을 방지하기 위한 조건문
    if last_row - 1 != 1:
        format_cell_range(worksheet, rollback_range, rollback_formatter)
    
    format_cell_range(worksheet, row_range, row_formatter)

def get_accelerator_dataloader(dataset, config, is_train=True):
    """
    Accelerator에서 사용할 데이터로더를 생성합니다.
    Args:
        dataset: 데이터셋 객체
        config: 설정 객체
        is_train: 학습용 데이터로더인지 여부
    Returns:
        DataLoader: PyTorch DataLoader 객체
    """
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size_train if is_train else config.batch_size_eval,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=dataset.collater,
        drop_last=is_train,
    )

    if is_train:
        loader = IterLoader(loader, use_distributed=False)
    return loader
