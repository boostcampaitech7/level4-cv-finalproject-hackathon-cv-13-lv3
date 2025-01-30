import os
import json
import torchaudio
import torch
import librosa

def load_audio(file_path, target_sr=16000):

    waveform, sr = torchaudio.load(file_path)
    
    # 리샘플링
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    
    # (채널, time) -> 모노로 합치기
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform, sr

def normalize_waveform(waveform, max_peak=0.9):
    peak = waveform.abs().max()
    if peak > 0:
        ratio = max_peak / peak
        waveform = waveform * ratio
    return waveform

def remove_silence(waveform, sr, top_db=25):
    # librosa.effects.split()은 numpy array를 필요로 함
    waveform_np = waveform.squeeze().numpy()
    intervals = librosa.effects.split(y=waveform_np, top_db=top_db)

    trimmed = []
    for start, end in intervals:
        trimmed.append(waveform_np[start:end])
    
    if len(trimmed) == 0:
        # 전부 무음이면 빈 텐서 반환
        return torch.zeros(0)
    
    trimmed_waveform_np = np.concatenate(trimmed)
    trimmed_waveform = torch.from_numpy(trimmed_waveform_np).unsqueeze(0)
    return trimmed_waveform

def pad_or_trim(waveform, max_length):
    """
    - 최대 길이 초과 시 자르고, 부족하면 패딩
    - ASR 학습 시 10~20초 이상 오디오가 길면 batch 지연이 커짐
      -> latency를 위해 일정 길이 제한 권장
    """
    length = waveform.shape[-1]
    if length > max_length:
        waveform = waveform[..., :max_length]
    elif length < max_length:
        pad_size = max_length - length
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    return waveform

def extract_features_mfcc(waveform, sr=16000, n_mfcc=13):
    """
    - CPU에서 하는 간단 MFCC 추출 예시
    - 실제로는 MelSpectrogram -> dct 변환 등 내부적으로 torch script 가능
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft":400, "hop_length":160, "n_mels":23}
    )
    mfcc = mfcc_transform(waveform)
    return mfcc  # shape: [channel, n_mfcc, time]

class LightASRPreprocessor(torch.nn.Module):
    """
    - GigaSpeech, LibriSpeech를 위한 경량화 전처리 모듈
    - TorchScript 호환 가능하도록 Module 상속.
    """
    def __init__(self, config):
        super(LightASRPreprocessor, self).__init__()
        self.sample_rate = config.get("sample_rate", 16000)
        self.max_length = config.get("max_length", 16000 * 15)  # 15초 제한
        self.apply_vad = config.get("apply_vad", False)
        self.top_db = config.get("vad_top_db", 25)
        self.extract_feature = config.get("extract_feature", True)
        self.n_mfcc = config.get("n_mfcc", 13)
        self.return_waveform = config.get("return_waveform", False)

    def forward(self, file_path: str) -> torch.Tensor:
        """
        - 파일 경로 입력 -> 전처리 -> (MFCC) 피처 텐서 반환
        - TorchScript 컴파일 가능 (단, dynamic file IO는 조금 주의 필요)
        """
        waveform, sr = load_audio(file_path, self.sample_rate)
        
        # 무음 제거 (긴 세그먼트일 때만?)
        if self.apply_vad:
            waveform = remove_silence(waveform, sr, self.top_db)
            if waveform.shape[-1] == 0:
                # 무음만 있으면 빈 텐서 반환
                return torch.zeros(0)

        # 정규화
        waveform = normalize_waveform(waveform, max_peak=0.9)

        # 길이 제한
        waveform = pad_or_trim(waveform, self.max_length)

        # 특징 추출
        if self.extract_feature:
            features = extract_features_mfcc(waveform, sr, self.n_mfcc)
            return features  # shape: [1, n_mfcc, T]
        else:
            return waveform

if __name__ == "__main__":
    import numpy as np

    config = {
        "sample_rate": 16000,
        "max_length": 16000 * 15,  # 15초
        "apply_vad": True,
        "vad_top_db": 25,
        "extract_feature": True,
        "n_mfcc": 13,
        "return_waveform": False
    }

    preprocessor = LightASRPreprocessor(config)
    
    # TorchScript로 컴파일
    scripted_preprocessor = torch.jit.script(preprocessor)

    test_file = "/path/to/Librispeech/sample-0001.flac"

    with torch.no_grad():
        # 전처리 수행 (CPU)
        mfcc = scripted_preprocessor(test_file)
        print("MFCC shape:", mfcc.shape)
    
    # mfcc 텐서를 바로 모델(Salmonn) 입력으로 사용하거나,
    # disk에 저장해서 추후 GPU 추론 시 메모리 절약 가능.