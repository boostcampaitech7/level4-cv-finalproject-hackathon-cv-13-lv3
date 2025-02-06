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

from silero_vad import load_silero_vad, get_speech_timestamps

class SALMONNDataset(Dataset):
    def __init__(self, prefix, ann_path, whisper_path, save_vad=False):
        super().__init__()
        # 데이터 경로 설정
        self.prefix = prefix
        # json 파일 로드
        self.annotation = json.load(open(ann_path, "r"))["annotation"]
        # Whisper 모델 로드 (특히 음성 데이터를 처리하는 모델)
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.vad_model = load_silero_vad()
        self.save_vad = save_vad
        self.save_vad_cnt = 0

    def extract_speech_by_vad(self, audio, original_sr, path, space_sec=0.0):
        """
        원본 audio와 원본 sampling rate를 받아,
        1. 8kHz 버전을 생성하여 VAD를 적용 (타임스탬프는 초 단위)
        2. 원본 오디오를 whisper_target_sr로 리샘플링한 후, VAD 타임스탬프에 해당하는 구간을 추출
        """
        # 1. VAD용으로 8kHz 버전 생성
        target_vad_sr = 16000
        audio_for_vad = librosa.resample(audio, orig_sr=original_sr, target_sr=target_vad_sr)
        
        # VAD 적용: 타임스탬프를 초 단위로 반환하도록 설정합니다.
        speech_segments = get_speech_timestamps(
            audio_for_vad, 
            self.vad_model, 
            sampling_rate=target_vad_sr,
            return_seconds=False  # 타임스탬프를 초 단위로 반환
        )
        remove = False
        # 만약 음성 구간이 없으면 원본 전체를 사용
        if len(speech_segments) == 0:
            with open("no_speech.log", "a") as log_file:
                log_file.write(path + "\n")
            speech_segments = [{'start': 0.0, 'end': len(audio_for_vad) / target_vad_sr}]
            remove = True
        
        whisper_target_sr = self.wav_processor.sampling_rate

        if original_sr != whisper_target_sr:
            # 2. 원본 오디오를 Whisper의 입력 샘플링 레이트로 리샘플링 (예: 16000, 32000 등)
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=whisper_target_sr)
        
        # VAD에서 얻은 초 단위 타임스탬프를 Whisper 샘플링 레이트의 인덱스로 변환하여 해당 구간만 추출
        speech_audio_segments = []
        for seg in speech_segments:
            start_idx = int((seg['start'] / target_vad_sr) * whisper_target_sr)
            end_idx   = int((seg['end']   / target_vad_sr) * whisper_target_sr)
            speech_audio_segments.append(audio[start_idx:end_idx])
            
        # 여러 음성 구간이 있을 경우 이어붙임
        if len(speech_audio_segments) > 1 and space_sec > 0:
            silence_samples = int(space_sec * whisper_target_sr)
            silence = np.zeros(silence_samples, dtype=audio.dtype)
            spaced_segments = []
            for i, segment in enumerate(speech_audio_segments):
                spaced_segments.append(segment)
                # 마지막 구간이 아니라면 silence 삽입
                if i < len(speech_audio_segments) - 1:
                    spaced_segments.append(silence)
            speech_audio = np.concatenate(spaced_segments, axis=0)
        else:
            speech_audio = np.concatenate(speech_audio_segments, axis=0)

        if self.save_vad is True and self.save_vad_cnt < 100 and remove is False:
            import random, os
            if random.randint(0, 2) == 1:
                sf.write(os.path.join("output", os.path.basename(path)), speech_audio, samplerate=whisper_target_sr)
                self.save_vad_cnt = self.save_vad_cnt + 1

        # vad_string = f"VAD Filter Applied : {len(audio)} -> {len(speech_audio)}"
        # print(vad_string)

        return speech_audio


    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        # 배치 내 샘플들을 하나로 모음
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        # 오디오 데이터 로드
        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        # 오디오 길이 계산
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        # 오디오 패딩
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        # 패딩 마스크 생성 (어디까지가 실제 오디오인지 표시)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)
        
        # 텍스트, 작업 유형, 질문 추출
        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        return {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]
        audio_path = (self.prefix + '/' + ann["path"]).replace("//", "/")
        try:
            # audio = 오디오 데이터, sr = 샘플링 레이트 (1초당 샘플 수, 샘플 = 오디오 데이터 단위)
            audio, sr = sf.read(audio_path)
        except:
            print(f"Failed to load {audio_path}. Load 0-th sample for now")
            audio, sr = sf.read(self.prefix + '/' + self.annotation[0]["path"])
        
        if len(audio.shape) == 2: # stereo to mono
            audio = audio[:, 0] # 한 채널만 선택해서 사용

        # 확장된 오디오 데이터 처리
        #if "expand_wav" in ann:
            # p = 확장된 오디오 데이터 경로
            #for p in ann["expand_wav"]:
                #expand_audio, _ = sf.read(self.prefix + '/' + p)
                #if len(expand_audio.shape) == 2:
                    #expand_audio = expand_audio[:, 0]
                # sil = 무음 데이터
                #sil = np.zeros(int(sr/10), dtype=float)
                # 원본 오디오와 확장 오디오 사이에 짧은 무음 데이터를 넣어서 구분
                #audio = np.concatenate((audio, sil, expand_audio), axis=0)
        
        # 오디오 데이터가 1초 미만이면 무음 데이터를 추가해서 1초 이상으로 만듦
        if len(audio) < sr: # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float) # 무음 데이터 생성
            audio = np.concatenate((audio, sil), axis=0) # 원본 오디오와 무음 데이터 연결

        task = ann.get("task", "asr") # 작업 유형 추출
        if task == "asr":
            # silero-vad를 사용하여 음성(말하는 부분)만 남기기
            audio = self.extract_speech_by_vad(audio, sr, audio_path)
        elif sr != self.wav_processor.sampling_rate: # TODO. use more efficient implementation            
            # Whisper 모델의 sr에 맞게 샘플링 (librosa.resample = 고품질이지만 느림)
            # scipy.signal.resample = 빠르지만 품질이 떨어짐, 다른 방법으로 교체 가능
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.wav_processor.sampling_rate)

        sr = self.wav_processor.sampling_rate

        # 오디오 데이터 30초로 자르기
        audio = audio[: sr * 30] # truncate audio to at most 30s
    

        # 오디오 데이터를 특징 벡터로 변환
        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        text = ann["text"] # 텍스트 추출
        # get(key, default)
        Q = ann.get("Q", "") # 질문 추출

        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": text,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }