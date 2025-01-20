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


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")


def setup_logger():
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARN,
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