import os
import argparse
import yaml
import logging

import torch
import torchaudio
from nnAudio import Spectrogram

##############################
# 로깅 설정
##############################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Music Preprocessing with nnAudio")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--audio-list", type=str, default=None, help="Text file with audio paths (one per line)")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save spectrogram .pt files")
    return parser.parse_args()

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class MusicPreprocessor:
    """
    - 오디오 파일 로드 -> nnAudio (STFT/Mel/CQT) 변환 -> Tensor 반환
    - GPU 사용 가능
    - config 예시:
      {
        "sample_rate": 22050,
        "transform_type": "mel",   # "stft" / "mel" / "cqt"
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 128,
        "fmin": 50.0,
        "fmax": 8000.0,
        "bins_per_octave": 12,
        "device": "cuda" or "cpu"
      }
    """
    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config.get("sample_rate", 22050)
        self.transform_type = config.get("transform_type", "stft")
        self.device_str = config.get("device", "cuda")
        
        # 실제 device 설정
        if self.device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Fallback to CPU.")
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)

        # transform_type에 따라 nnAudio 레이어 초기화
        if self.transform_type == "stft":
            self.spectrogram_layer = Spectrogram.STFT(
                n_fft=config.get("n_fft", 1024),
                hop_length=config.get("hop_length", 512),
                freq_scale=config.get("freq_scale", "no"),  
                trainable=config.get("trainable_stft", False),
                sr=self.sample_rate
            )
        elif self.transform_type == "mel":
            self.spectrogram_layer = Spectrogram.MelSpectrogram(
                sr=self.sample_rate,
                n_fft=config.get("n_fft", 1024),
                n_mels=config.get("n_mels", 128),
                hop_length=config.get("hop_length", 512),
                fmin=config.get("fmin", 0.0),
                fmax=config.get("fmax", None),
                trainable_mel=config.get("trainable_mel", False),
                trainable_STFT=config.get("trainable_STFT", False)
            )
        elif self.transform_type == "cqt":
            self.spectrogram_layer = Spectrogram.CQT(
                sr=self.sample_rate,
                hop_length=config.get("hop_length", 512),
                fmin=config.get("fmin", 32.70),
                fmax=config.get("fmax", None),
                bins_per_octave=config.get("bins_per_octave", 12),
                trainable_cqt=config.get("trainable_cqt", False)
            )
        else:
            raise ValueError(f"Unknown transform_type: {self.transform_type}")

        # spectrogram_layer를 device로 이동
        self.spectrogram_layer = self.spectrogram_layer.to(self.device)

    def load_audio(self, file_path: str) -> torch.Tensor:
        """
        오디오 파일 로드 -> wave Tensor 반환
        (sr != self.sample_rate 이면 리샘플)
        (여러 채널이면 모노 변환)
        """
        if not os.path.exists(file_path):
            logger.warning(f"Audio file not found: {file_path}")
            return torch.zeros(1,1, device=self.device)

        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform.to(self.device)

    def __call__(self, file_path: str) -> torch.Tensor:
        """
        - 오디오 로드
        - nnAudio 레이어 변환
        - 반환: (batch=1, freq_bins, time_bins)
        """
        waveform = self.load_audio(file_path)
        if waveform.numel() == 0:
            # empty -> return empty tensor
            return waveform

        # shape: [1, 1, time]
        waveform = waveform.unsqueeze(0)

        # spectrogram 변환
        spectrogram = self.spectrogram_layer(waveform)
        return spectrogram


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    logger.info(f"Loaded config: {config}")

    # MusicPreprocessor 초기화
    preprocessor = MusicPreprocessor(config)

    # audio-list
    file_list = []
    if args.audio_list and os.path.exists(args.audio_list):
        with open(args.audio_list, 'r') as f:
            file_list = [line.strip() for line in f if line.strip()]
    else:
        logger.warning("No valid audio-list provided. Using default 'sample_music.wav'.")
        file_list = ["sample_music.wav"]

    # output dir
    output_dir = args.output_dir or config.get("output_dir", "./music_outputs")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Start preprocessing {len(file_list)} files. Output={output_dir}")

    # 전처리
    for idx, filepath in enumerate(file_list):
        with torch.no_grad():
            spec = preprocessor(filepath)

        # shape 확인
        logger.info(f"[{idx+1}/{len(file_list)}] {filepath} -> shape={spec.shape}")

        # 저장
        base = os.path.splitext(os.path.basename(filepath))[0]
        save_name = os.path.join(output_dir, base + "_spec.pt")
        torch.save(spec.cpu(), save_name)
        logger.info(f"Saved: {save_name}")

    logger.info("All done.")


if __name__ == "__main__":
    main()