import torchaudio
import json
import os
import torch
import numpy as np
import librosa

def get_paths(root_dir='/data/dataset'):
    """
    - 예시: 특정 JSON(annotation)에서 path들을 뽑아오는 함수
    - 데이터셋마다 directory 구조가 다를 수 있으므로 적절히 수정
    """
    annotation = None
    paths = {}

    # 예시 JSON 2개를 합쳐서 annotation 리스트를 얻는다
    with open('/data/dataset/new_jsons/stage1_train_indented.json', 'r') as f:
        annotation = json.load(f)['annotation']
    with open('/data/dataset/new_jsons/stage2_train_indented.json', 'r') as f:
        annotation += json.load(f)['annotation']

    # path 정보를 dataset 별로 분류
    for annot in annotation:
        path = os.path.join(annot['path'])   # 'GigaSpeech/XXX.wav' 같은 형태
        dataset = path.split('/')[0]         # 'GigaSpeech' 등
        if dataset not in paths:
            paths[dataset] = set()
        paths[dataset].add(path)
    
    # set -> list 변환
    for k, v in paths.items():
        paths[k] = list(v)

    return paths


def audio_resample(waveform: torch.Tensor, 
                   sample_rate: int, 
                   target_samplerate: int = 16000):
    """
    - 현재 sample_rate가 target_samplerate와 다르면 resample
    - GPU 사용량 줄이기 위해 주로 CPU에서 수행
    """
    if sample_rate != target_samplerate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_samplerate)
        waveform = resampler(waveform)
        sample_rate = target_samplerate
    return waveform, sample_rate


#################################
# 각 데이터셋별 로드 함수 예시
#################################
def func1(audio_path):
    """
    Clotho 전용 로드 로직 (필요시 커스텀)
    """
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

def func2(audio_path):
    """
    audiocaps 전용 로드 로직
    """
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

def func3(audio_path):
    """
    GigaSpeech 전용 로드 로직
    (실제론 large 파일에서 세그먼트 추출할 수도 있음)
    """
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

def func4(audio_path):
    """
    LibriSpeech 전용 로드 로직
    """
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

def func5(audio_path):
    """
    MusicNet 전용 로드 로직
    """
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

def func6(audio_path):
    """
    WavCaps 전용 로드 로직
    """
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr


def load_audio(path, 
               root_dir='/data/dataset', 
               target_samplerate=16000):
    """
    - path 예시: "GigaSpeech/audio/xxx.wav"
    - 앞부분(split('/')[0]) = "GigaSpeech" / "LibriSpeech" / ...
    - 데이터셋마다 다른 funcX를 호출해 로드
    - 이후 resample
    """
    dataset = path.split('/')[0]  
    audio_path = os.path.join(root_dir, path)
    
    # dataset에 따라 다른 로딩 로직
    if dataset == 'Clotho':
        waveform, sample_rate = func1(audio_path)
    elif dataset == 'audiocaps':
        waveform, sample_rate = func2(audio_path)
    elif dataset == 'GigaSpeech':
        waveform, sample_rate = func3(audio_path)
    elif dataset == 'LibriSpeech':
        waveform, sample_rate = func4(audio_path)
    elif dataset == 'MusicNet':
        waveform, sample_rate = func5(audio_path)
    elif dataset == 'WavCaps':
        waveform, sample_rate = func6(audio_path)
    else:
        # default
        waveform, sample_rate = torchaudio.load(audio_path)

    # 리샘플(16kHz)
    waveform, sample_rate = audio_resample(waveform, sample_rate, target_samplerate=target_samplerate)
    
    # 채널 수가 여러 개인 경우 모노로 합치기
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform, sample_rate


#################################
# 추가 유틸 함수 (ASR, 공통)
#################################
def normalize_waveform(waveform: torch.Tensor, max_peak: float = 0.9):
    """
    - peak-based normalization
    - 클리핑 방지용
    """
    peak = waveform.abs().max()
    if peak > 0:
        ratio = max_peak / peak
        waveform = waveform * ratio
    return waveform


def remove_silence(waveform: torch.Tensor, 
                   sr: int, 
                   top_db: int = 20) -> torch.Tensor:
    """
    - librosa.effects.split() 기반 무음 제거
    - CPU에서 처리 권장
    """
    waveform_np = waveform.squeeze().numpy()
    intervals = librosa.effects.split(waveform_np, top_db=top_db)
    if len(intervals) == 0:
        # 전부 무음이면 원본 반환 (혹은 빈 텐서 반환)
        return waveform

    trimmed_segments = []
    for start, end in intervals:
        trimmed_segments.append(waveform_np[start:end])
    
    trimmed_np = np.concatenate(trimmed_segments)
    trimmed_tensor = torch.from_numpy(trimmed_np).unsqueeze(0)
    return trimmed_tensor


def pad_or_trim(waveform: torch.Tensor, max_length: int):
    """
    - 오디오 길이가 max_length(샘플 수)를 초과하면 자르고,
      모자라면 zero-padding
    """
    length = waveform.shape[-1]
    if length > max_length:
        waveform = waveform[..., :max_length]
    else:
        pad_size = max_length - length
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    return waveform