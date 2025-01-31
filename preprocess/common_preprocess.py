import torchaudio
import json
import os
import torch
import numpy as np
import librosa

###############################
# 1. 전역 파라미터 (경량화 관련)
###############################
MAX_CHUNK_SEC = 30  # 30초 단위로 나누어 처리 (원하는 대로 조절)
USE_OFFLINE_FEATURES = False  # True면 오프라인 특징 추출 사용
USE_HALF = False              # True면 waveform을 half-precision으로 변환
CHUNK_OVERLAP_SEC = 0         # 스트리밍 시 overlap 적용 가능

def get_paths(root_dir='/data/dataset'):
    """
    예시 JSON 2개(stage1, stage2)를 합쳐서 path 정보를 수집
    dataset 별로 path를 dict로 분류
    """
    annotation = []
    paths = {}

    # 파일 존재 여부 및 JSON 구조 확인
    stage1_path = os.path.join(root_dir, 'new_jsons', 'stage1_train_indented.json')
    stage2_path = os.path.join(root_dir, 'new_jsons', 'stage2_train_indented.json')

    if not os.path.exists(stage1_path):
        raise FileNotFoundError(f"JSON file not found: {stage1_path}")
    if not os.path.exists(stage2_path):
        raise FileNotFoundError(f"JSON file not found: {stage2_path}")

    with open(stage1_path, 'r') as f:
        annotation = json.load(f)['annotation']
    with open(stage2_path, 'r') as f:
        annotation += json.load(f)['annotation']

    for annot in annotation:
        # annot['path']: 예 'GigaSpeech/XXX.wav'
        path = os.path.join(annot['path'])
        dataset = path.split('/')[0]  # ex: 'GigaSpeech'
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
    - sample_rate != target_samplerate일 경우 Resample
    - GPU/CPU 모두에서 동작 가능
    - 필요시 TorchScript로 스크립팅하여 최적화 가능
    """
    if sample_rate != target_samplerate:
        # ex) resampler = torch.jit.script(torchaudio.transforms.Resample(...))
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, 
                                                   new_freq=target_samplerate)
        waveform = resampler(waveform)
        sample_rate = target_samplerate
    return waveform, sample_rate

#################################
# 각 데이터셋별 로드 함수 예시
#################################
def func1(audio_path):  # Clotho
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

def func2(audio_path):  # AudioCaps
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

def func3(audio_path):  # GigaSpeech
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

def func4(audio_path):  # LibriSpeech
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

def func5(audio_path):  # MusicNet
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr

def func6(audio_path):  # WavCaps
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr


def load_audio(path, root_dir='/data/dataset', target_samplerate=16000):
    """
    - dataset 종류에 따라 다른 함수를 통해 오디오 로드
    - 리샘플(16kHz), 모노 변환, half-precision 변환(옵션)
    """
    dataset = path.split('/')[0]
    audio_path = os.path.join(root_dir, path)
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # dataset별 로딩
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
    waveform, sample_rate = audio_resample(waveform, sample_rate, 
                                           target_samplerate=target_samplerate)
    
    # 다채널 -> 모노
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # half-precision (optional)
    # 여기서는 CPU 텐서인 경우 float16 변환이 가능하나,
    # 일부 librosa/VAD 연산은 float32가 필요할 수 있으므로
    # -> VAD 이후 half로 바꿔도 됨 (상황 맞춰서)
    if USE_HALF:
        waveform = waveform.half()

    return waveform, sample_rate


#################################
# 2. 추가 유틸 함수 (VAD/정규화/패딩 등)
#################################
def normalize_waveform(waveform: torch.Tensor, max_peak: float = 0.9):
    """
    peak-based normalization
    """
    peak = waveform.abs().max()
    if peak > 0:
        ratio = max_peak / peak
        waveform = waveform * ratio
    return waveform


def remove_silence(waveform: torch.Tensor, sr: int, top_db: int = 20) -> torch.Tensor:
    """
    librosa.effects.split() 기반 무음 제거
    - waveform을 float()로 변환 후 numpy 변환
    - intervals로 잘라서 concatenate
    - half precision은 VAD 이후 재적용
    """
    # 먼저 float 변환 (librosa 의 CPU 연산)
    waveform_np = waveform.squeeze().float().numpy()
    intervals = librosa.effects.split(y=waveform_np, top_db=top_db)
    if len(intervals) == 0:
        # 전부 무음이면 원본 반환(혹은 빈 텐서)
        return waveform

    trimmed_segments = []
    for start, end in intervals:
        trimmed_segments.append(waveform_np[start:end])
    
    trimmed_np = np.concatenate(trimmed_segments, axis=0)
    trimmed_tensor = torch.from_numpy(trimmed_np).unsqueeze(0)

    # half precision 다시 적용
    if USE_HALF:
        trimmed_tensor = trimmed_tensor.half()

    return trimmed_tensor


def pad_or_trim(waveform: torch.Tensor, max_length: int):
    """
    - waveform 길이가 max_length(샘플 수) 초과하면 자르고,
      모자라면 zero-pad
    """
    length = waveform.shape[-1]
    if length > max_length:
        waveform = waveform[..., :max_length]
    else:
        pad_size = max_length - length
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    return waveform


##############################
# 3. (선택) Chunk splitting
##############################
def chunk_waveform(waveform: torch.Tensor, sr: int, 
                   chunk_sec: int = MAX_CHUNK_SEC, 
                   overlap_sec: int = CHUNK_OVERLAP_SEC):
    """
    - 긴 waveform을 여러 chunk로 나누어 반환
    - overlap_sec > 0 이면 인접 chunk 간 overlap 적용
    """
    chunk_len = int(chunk_sec * sr)
    overlap_len = int(overlap_sec * sr)

    total_len = waveform.shape[-1]
    start = 0
    results = []

    while start < total_len:
        end = start + chunk_len
        if end > total_len:
            end = total_len
        
        c = waveform[..., start:end]
        results.append(c.clone())  # clone()해서 별도 텐서

        # 다음 start = end - overlap
        next_start = end - overlap_len
        if next_start <= start:
            # 오버랩이 너무 커서 무한 루프 방지
            break
        start = next_start

    return results


##############################
# 4. (선택) 오프라인 특징 추출
##############################
def extract_features_offline(waveform: torch.Tensor, sr: int):
    """
    - MelSpectrogram 등 추출 후 -> log 변환
    - USE_HALF이면 half 변환
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=80,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
    )
    mel = mel_transform(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

    if USE_HALF:
        mel_db = mel_db.half()

    return mel_db


##############################
# 5. 예시 파이프라인 함수
##############################
def preprocess_audio(path, 
                     root_dir='/data/dataset', 
                     target_sr=16000, 
                     max_length_samples=16000*30,
                     apply_vad=False,
                     top_db=20):
    """
    - load_audio()로 wave 로딩
    - (옵션) VAD 무음 제거
    - 정규화
    - chunk 분할 후 pad_or_trim
    """
    waveform, sr = load_audio(path, root_dir, target_sr)

    # VAD (옵션)
    if apply_vad:
        waveform = remove_silence(waveform, sr, top_db)

    # 정규화
    waveform = normalize_waveform(waveform, max_peak=0.9)

    # chunk
    chunks = chunk_waveform(waveform, sr, chunk_sec=30)
    processed_list = []
    for c in chunks:
        c = pad_or_trim(c, max_length_samples)
        processed_list.append(c)

    return processed_list

##############################
# usage example
##############################
if __name__ == "__main__":
    # 예: GigaSpeech 음성 파일 90초라 가정
    # chunk 30초씩 3개
    path_example = "GigaSpeech/audio/chapter01_0001.wav"
    
    # 전처리
    chunk_waveforms = preprocess_audio(
        path_example,
        root_dir="/data/dataset",
        target_sr=16000,
        max_length_samples=16000*30,
        apply_vad=False,  # 음악 캡셔닝이라면 False일 수도
        top_db=20
    )

    # 아래는 offline feature 예시
    for i, c_wav in enumerate(chunk_waveforms):
        if USE_OFFLINE_FEATURES:
            mel_feat = extract_features_offline(c_wav, 16000)
            print(f"Chunk {i}, mel shape = {mel_feat.shape}")
        else:
            print(f"Chunk {i}, shape = {c_wav.shape}")