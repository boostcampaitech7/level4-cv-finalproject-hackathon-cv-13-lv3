import numpy as np
from torch.utils.data import BatchSampler
from accelerate import Accelerator

import json

class BucketingBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, length_info_path='/data/dataset/path_length_pair.json', default_length=30.0, shuffle=True, distributed=False):
        """
        Args:
            dataset: SALMONNDataset 객체
            length_info: {'PATH': LENGTH} 형태의 딕셔너리. LENGTH는 초 단위.
            batch_size: 한 배치당 샘플 수.
            default_length: length_info에 해당 경로가 없을 때 사용할 기본 길이(초 단위).
            shuffle: 버킷 내에서 섞을지 여부.
            distributed: 분산 학습 환경이면 True (torch.distributed가 초기화되었다고 가정)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        
        with open(length_info_path, "r") as f:
            self.length_info = json.load(f)

        self.accelerator = Accelerator()     
        # 각 sample의 길이를 length_info에서 가져오고, 없으면 default_length 사용
        self.lengths = [
            self.length_info.get(ann["path"], default_length) 
            for ann in dataset.annotation
        ]
        
        # 전체 인덱스를 오디오 길이 기준으로 오름차순 정렬
        self.indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        
        # 배치 단위로 묶은 결과를 미리 생성
        self.batches = self._create_batches()
        
    def _create_batches(self):
        # 버킷 단위로 묶어, 각 버킷 내에서만 섞은 후 전체 인덱스를 재구성하는 방식
        if self.shuffle:
            # 버킷 사이즈는 배치 사이즈의 몇 배로 설정 (예: 10배)
            bucket_size = self.batch_size * 10
            buckets = [self.indices[i:i+bucket_size] for i in range(0, len(self.indices), bucket_size)]
            for bucket in buckets:
                np.random.shuffle(bucket)
            indices = [idx for bucket in buckets for idx in bucket]
        else:
            indices = self.indices

        # 분산 학습일 경우 각 GPU(프로세스)마다 인덱스를 분할
        if self.distributed:
            # torch.distributed가 초기화되었음을 전제로 world_size와 rank를 가져옵니다.
            world_size = self.accelerator.state.num_processes
            rank = self.accelerator.state.process_index
            indices = indices[rank::world_size]

        # 최종적으로 배치 사이즈 단위로 자르기
        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
