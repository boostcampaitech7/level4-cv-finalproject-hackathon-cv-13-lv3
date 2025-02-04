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

import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor
import librosa

from transformers import ClapModel, ClapProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clap_model = ClapModel.from_pretrained("laion/larger_clap_music").to(device)
clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

def extract_clap_embedding(audio, sr, target_sr=48000):
    """
    오디오 데이터를 받아서 CLAP 임베딩을 추출함.
    :param audio: numpy array, 원시 오디오 데이터
    :param sr: int, 현재 샘플링 레이트
    :param target_sr: CLAP 모델이 요구하는 샘플링 레이트 (보통 48000)
    :return: torch.Tensor, CLAP 임베딩
    """
    # 필요하면 샘플링 레이트 조정 (CLAP는 보통 48kHz를 사용)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    inputs = clap_processor(audios=audio, return_tensors="pt", sampling_rate=sr)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_features = clap_model.get_audio_features(**inputs)
    return audio_features.squeeze(0)  # (feature_dim,)

class SALMONNDataset(Dataset):
    def __init__(self, prefix, ann_path, whisper_path):
        super().__init__()
        
        # 데이터 경로 설정
        self.prefix = prefix

        # json 파일 로드
        data = json.load(open(ann_path, "r"))["annotation"]
        annotation_wo_GigaSpeech = [item for item in data if 'GigaSpeech' not in item['path']]
        self.annotation = annotation_wo_GigaSpeech
        
        # Whisper 모델 로드 (특히 음성 데이터를 처리하는 모델)
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        # 배치 내 샘플들을 하나로 모음
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        # 오디오 데이터 로드
        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)
        
        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        # 만약 CLAP 임베딩이 배치별로 필요하다면 여기서도 결합할 수 있음.
        # 예: clap_embeddings = torch.stack([s["clap_embedding"] for s in samples], dim=0)

        return {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
            # "clap_embedding": clap_embeddings,  # 필요한 경우 추가
        }

    def __getitem__(self, index):
        ann = self.annotation[index]
        audio_path = os.path.join(self.prefix, ann["path"]).replace("//", "/")
        try:
            audio, sr = sf.read(audio_path)
        except Exception as e:
            print(f"Failed to load {audio_path} ({e}). Using 0-th sample instead.")
            audio, sr = sf.read(os.path.join(self.prefix, self.annotation[0]["path"]))
        
        if len(audio.shape) == 2:  # stereo -> mono
            audio = audio[:, 0]

        # 1초 미만이면 패딩
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        
        # Whisper 모델 샘플링 레이트와 다르면 재샘플링
        if sr != self.wav_processor.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.wav_processor.sampling_rate)
            sr = self.wav_processor.sampling_rate

        # 최대 30초 자르기
        audio = audio[: sr * 30]

        # Whisper 전용 spectrogram 추출
        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        text = ann["text"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        # 여기서 CLAP 임베딩 추출 (CLAP는 보통 48kHz 사용하므로, 필요하면 재샘플링)
        clap_embedding = extract_clap_embedding(audio, sr, target_sr=48000)

        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": text,
            "task": task,
            "Q": Q,
            "id": ann["path"],
            "clap_embedding": clap_embedding,  # CLAP 임베딩 추가
        }