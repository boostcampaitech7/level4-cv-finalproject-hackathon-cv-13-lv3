import os
import json
import logging
import torchaudio
import torch
import numpy as np
import librosa

###############################
# 로깅 설정
###############################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

###############################
# 1. 전역 파라미터 (경량화 관련)
###############################
MAX_CHUNK_SEC = 30    # 30초 단위로 나누어 처리 (원하는 대로 조절)
USE_OFFLINE_FEATURES = False  # True면 오프라인 특징 추출 사용
USE_HALF = False              # True면 waveform을 half-precision으로 변환
CHUNK_OVERLAP_SEC = 0         # 스트리밍 시 overlap 적용 가능

def get_paths(root_dir='/data/dataset'):
    """
    stage1_train_indented.json과 stage2_train_indented.json에서
    'annotation' 키를 합쳐서, dataset 별 path 리스트를 구성.

    Args:
        root_dir (str): JSON 파일이 위치한 상위 디렉토리.

    Returns:
        dict: {dataset_name: [path1, path2, ...], ...}
    """
    annotation = []
    paths = {}

    stage1_path = os.path.join(root_dir, 'new_jsons', 'stage1_train_indented.json')
    stage2_path = os.path.join(root_dir, 'new_jsons', 'stage2_train_indented.json')

    if not os.path.exists(stage1_path):
        logger.error(f"JSON file not found: {stage1_path}")
        raise FileNotFoundError(f"JSON file not found: {stage1_path}")
    if not os.path.exists(stage2_path):
        logger.error(f"JSON file not found: {stage2_path}")
        raise FileNotFoundError(f"JSON file not found: {stage2_path}")

    with open(stage1_path, 'r') as f:
        annotation = json.load(f)['annotation']
    with open(stage2_path, 'r') as f:
        annotation += json.load(f)['annotation']

    for annot in annotation:
        # ex) annot['path'] = 'GigaSpeech/xxx.wav'
        path = os.path.join(annot['path'])
        dataset = path.split('/')[0]   # e.g. 'GigaSpeech'
        if dataset not in paths:
            paths[dataset] = set()
        paths[dataset].add(path)
    
    # set -> list 변환
    for k, v in paths.items():
        paths[k] = list(v)

    logger.info(f"Collected dataset paths: {list(paths.keys())}")
    return paths

def audio_resample(waveform: torch.Tensor, sample_rate: int,
                   target_samplerate: int = 16000):
    """
    sample_rate != target_samplerate인 경우 Resample 실행 (CPU/GPU 모두 가능).
    필요시 TorchScript로 스크립팅하여 최적화 가능.

    Args:
        waveform (Tensor): (channels, time)
        sample_rate (int)
        target_samplerate (int)

    Returns:
        tuple: (waveform, new_sample_rate)
    """
    if sample_rate != target_samplerate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                   new_freq=target_samplerate)
        waveform = resampler(waveform)
        sample_rate = target_samplerate
    return waveform, sample_rate

#################################
# 2. 각 데이터셋별 로드 함수 예시
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
    dataset 종류에 따라 별도 funcX로 오디오 로드.
    이후 리샘플(16kHz), 모노 변환, (옵션) half-precision 변환.

    Returns:
        tuple: (waveform, sample_rate)
    """
    dataset = path.split('/')[0]
    audio_path = os.path.join(root_dir, path)
    
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return torch.zeros(1,1), target_samplerate

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
        waveform, sample_rate = torchaudio.load(audio_path)

    # 리샘플(16kHz)
    waveform, sample_rate = audio_resample(waveform, sample_rate,
                                           target_samplerate=target_samplerate)
    
    # 다채널 -> 모노
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # half-precision
    if USE_HALF:
        waveform = waveform.half()

    return waveform, sample_rate

#################################
# 3. 추가 유틸 함수 (VAD, 정규화, 패딩 등)
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
    - waveform float32 변환 후 numpy로 librosa 처리
    - intervals로 잘라서 concat
    - half precision은 VAD 이후 재적용
    """
    if waveform.numel() == 0:
        # 비어있으면 그냥 반환
        return waveform

    wave_np = waveform.squeeze().float().numpy()
    intervals = librosa.effects.split(y=wave_np, top_db=top_db)
    if len(intervals) == 0:
        # 전부 무음 -> 빈 텐서
        logger.info("All silent segment.")
        return torch.zeros(1,1)

    trimmed_segs = []
    for start, end in intervals:
        trimmed_segs.append(wave_np[start:end])
    
    trimmed_np = np.concatenate(trimmed_segs, axis=0)
    trimmed_tensor = torch.from_numpy(trimmed_np).unsqueeze(0)

    if USE_HALF:
        trimmed_tensor = trimmed_tensor.half()

    return trimmed_tensor

def pad_or_trim(waveform: torch.Tensor, max_length: int):
    """
    wave 길이가 max_length 넘으면 자르고, 모자라면 zero-pad
    """
    length = waveform.shape[-1]
    if length > max_length:
        waveform = waveform[..., :max_length]
    else:
        pad_size = max_length - length
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    return waveform

##############################
# 4. Chunk splitting (옵션)
##############################
def chunk_waveform(waveform: torch.Tensor, sr: int,
                   chunk_sec: int = MAX_CHUNK_SEC,
                   overlap_sec: int = CHUNK_OVERLAP_SEC):
    """
    긴 waveform -> 여러 chunk로 분할
    overlap_sec > 0 -> 인접 청크간 겹침
    반환: list of chunk (Tensor)
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
        
        c = waveform[..., start:end].clone()
        results.append(c)

        # 다음 start = end - overlap
        next_start = end - overlap_len
        if next_start <= start:
            # 무한루프 방지
            break
        start = next_start

    return results

##############################
# 5. 오프라인 특징 추출(옵션)
##############################
def extract_features_offline(waveform: torch.Tensor, sr: int):
    """
    - MelSpectrogram -> log 변환
    - USE_HALF이면 half 변환
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=80,
        n_fft=1024,
        hop_length=256,
        win_length=1024
    )
    mel = mel_transform(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

    if USE_HALF:
        mel_db = mel_db.half()

    return mel_db

##############################
# 6. 예시 파이프라인 함수
##############################
def preprocess_audio(path, 
                     root_dir='/data/dataset',
                     target_sr=16000,
                     max_length_samples=16000*30,
                     apply_vad=False,
                     top_db=20):
    """
    1) load_audio(path)
    2) remove_silence (옵션)
    3) normalize
    4) chunk_waveform -> pad_or_trim
    5) list of chunk 반환
    """
    waveform, sr = load_audio(path, root_dir, target_sr)

    if apply_vad:
        waveform = remove_silence(waveform, sr, top_db)

    # 정규화
    waveform = normalize_waveform(waveform, max_peak=0.9)

    # 청크 분할
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
    path_example = "data/dataset/GigaSpeech/0"
    # 30초씩 나누어 전처리
    chunk_waveforms = preprocess_audio(
        path_example,
        root_dir="/data/dataset",
        target_sr=16000,
        max_length_samples=16000*30,
        apply_vad=False,
        top_db=20
    )

    for i, cwav in enumerate(chunk_waveforms):
        length_sec = cwav.shape[-1] / 16000
        print(f"Chunk {i}: shape={cwav.shape}, duration={length_sec:.2f}s")

    if USE_OFFLINE_FEATURES and len(chunk_waveforms) > 0:
        feat = extract_features_offline(chunk_waveforms[0], 16000)
        print("Offline feature shape:", feat.shape)