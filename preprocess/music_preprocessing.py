############################
# music_preprocessing.py
############################

import torch
import torchaudio
from nnAudio import Spectrogram
import os

class MusicPreprocessor:
    """
    MusicPreprocessor:
    - 오디오 파일을 로드
    - nnAudio로 on-the-fly 변환 (STFT / Mel / CQT)
    - GPU 상에서 파이프라인 수행

    config 예시:
    config = {
        "sample_rate": 22050,
        "transform_type": "mel",  # "stft" / "mel" / "cqt"
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 128,
        "fmin": 50.0,
        "fmax": 8000.0,
        "bins_per_octave": 12,
        ...
    }
    """

    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config.get("sample_rate", 22050)
        self.transform_type = config.get("transform_type", "stft")

        # nnAudio 스펙트럼 레이어 초기화
        # transform_type에 따라 다른 레이어 불러오기
        if self.transform_type == "stft":
            # 예: STFT 레이어
            self.spectrogram_layer = Spectrogram.STFT(
                n_fft=config.get("n_fft", 1024),
                hop_length=config.get("hop_length", 512),
                freq_scale=config.get("freq_scale", "no"),  # 'no', 'log'
                trainable=config.get("trainable_stft", False),
                sr=self.sample_rate
            )
        elif self.transform_type == "mel":
            # 예: MelSpectrogram 레이어
            self.spectrogram_layer = Spectrogram.MelSpectrogram(
                sr=self.sample_rate,
                n_fft=config.get("n_fft", 1024),
                n_mels=config.get("n_mels", 128),
                hop_length=config.get("hop_length", 512),
                fmin=config.get("fmin", 0.0),
                fmax=config.get("fmax", None),
                trainable_mel=config.get("trainable_mel", False),
                trainable_STFT=config.get("trainable_stft", False)
            )
        elif self.transform_type == "cqt":
            # 예: CQT 레이어
            self.spectrogram_layer = Spectrogram.CQT(
                sr=self.sample_rate,
                hop_length=config.get("hop_length", 512),
                fmin=config.get("fmin", 32.70), # C1 = 32.70 Hz
                fmax=config.get("fmax", None),
                bins_per_octave=config.get("bins_per_octave", 12),
                trainable_cqt=config.get("trainable_cqt", False)
            )
        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")

    def load_audio(self, file_path: str) -> torch.Tensor:
        """
        torchaudio를 사용해 오디오 파일 로드 -> wave Tensor 반환
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        waveform, sr = torchaudio.load(file_path)
        # 리샘플링이 필요하면 적용
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate

        # 모노 변환 (채널 평균)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    def __call__(self, file_path: str) -> torch.Tensor:
        """
        - 오디오 로드
        - nnAudio 레이어로 변환 (GPU 상에서)
        - 스펙트럼 텐서 반환
        """
        waveform = self.load_audio(file_path)

        # (batch, time) 형태로 맞추기
        # nnAudio는 (batch, time) or (batch, 1, time) 등
        waveform = waveform.unsqueeze(0)  # => shape: [1, 1, time]
        
        # GPU로 이동 (필요시)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waveform = waveform.to(device)

        self.spectrogram_layer = self.spectrogram_layer.to(device)

        # 변환 수행
        # nnAudio 레이어에 waveform (batch, time) or (batch, 1, time)을 넣으면
        # (batch, freq_bins, time_bins) 형태 텐서 반환
        spectrogram = self.spectrogram_layer(waveform)

        return spectrogram


####################################
# 사용 예시
####################################
if __name__ == "__main__":
    config = {
        "sample_rate": 22050,
        "transform_type": "mel",  # "stft", "mel", or "cqt"
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 128,
        "fmin": 50.0,
        "fmax": 8000.0,
    }

    preprocessor = MusicPreprocessor(config)
    test_audio = "path/to/your_music_file.wav"

    with torch.no_grad():
        spec = preprocessor(test_audio)
        print("Spectrogram shape:", spec.shape)  # e.g. [1, freq_bins, time_bins]