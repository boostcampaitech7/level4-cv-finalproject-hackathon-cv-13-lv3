import os
import json
import torchaudio
import torch
import librosa
import argparse
import yaml
import numpy as np
import logging

####################################
# 0. Logging 설정 (필요 시)
####################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

####################################
# 1. Argparse + YAML 설정
####################################
def parse_args():
    parser = argparse.ArgumentParser(description="ASR Preprocessing")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--audio-list", type=str, default=None, help="Path to text file containing audio paths (one per line)")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save extracted features (optional)")
    return parser.parse_args()

def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

####################################
# 2. 공통 함수들
####################################
def load_audio(file_path, target_sr=16000, device="cpu"):
    if not os.path.exists(file_path):
        logger.warning(f"Audio file not found: {file_path}")
        return None, 0

    waveform, sr = torchaudio.load(file_path)
    
    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    
    # Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Optional GPU
    waveform = waveform.to(device)
    return waveform, sr

def normalize_waveform(waveform, max_peak=0.9):
    peak = waveform.abs().max()
    if peak > 0:
        ratio = max_peak / peak
        waveform = waveform * ratio
    return waveform

def remove_silence(waveform, sr, top_db=25):
    # librosa.effects.split() needs CPU & numpy
    # -> waveform back to CPU & float
    wave_cpu = waveform.squeeze().cpu().float().numpy()
    intervals = librosa.effects.split(y=wave_cpu, top_db=top_db)

    if len(intervals) == 0:
        # all silent
        return torch.zeros(0, device=waveform.device)
    
    trimmed_segments = []
    for start, end in intervals:
        trimmed_segments.append(wave_cpu[start:end])
    
    trimmed_np = np.concatenate(trimmed_segments)
    trimmed_tensor = torch.from_numpy(trimmed_np).unsqueeze(0).to(waveform.device)
    return trimmed_tensor

def pad_or_trim(waveform, max_length):
    length = waveform.shape[-1]
    if length > max_length:
        waveform = waveform[..., :max_length]
    else:
        pad_size = max_length - length
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    return waveform

def extract_features_mfcc(waveform, sr=16000, n_mfcc=13, device="cpu"):
    # move transform to same device
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft":400, "hop_length":160, "n_mels":23}
    ).to(device)

    mfcc = mfcc_transform(waveform)
    return mfcc  # shape: [channel, n_mfcc, time]

####################################
# 3. LightASRPreprocessor 모듈
####################################
class LightASRPreprocessor(torch.nn.Module):
    """
    - GigaSpeech, LibriSpeech 등을 위한 경량화 전처리 모듈
    - TorchScript 호환 (dynamic file IO 주의)
    """
    def __init__(self, config, device='cpu'):
        super(LightASRPreprocessor, self).__init__()
        self.device = device
        self.sample_rate = config.get("sample_rate", 16000)
        self.max_length = config.get("max_length", 16000 * 15)  # default 15s
        self.apply_vad = config.get("apply_vad", False)
        self.top_db = config.get("vad_top_db", 25)
        self.extract_feature = config.get("extract_feature", True)
        self.n_mfcc = config.get("n_mfcc", 13)
        self.return_waveform = config.get("return_waveform", False)
        self.vad_min_sec = config.get("vad_min_sec", 3.0)  # optional: only apply VAD if > 3s

    def forward(self, file_path: str) -> torch.Tensor:
        """
        - file_path -> 전처리 -> (MFCC) 텐서 or waveform 반환
        """
        waveform, sr = load_audio(file_path, self.sample_rate, device=self.device)
        if waveform is None or waveform.numel() == 0:
            # failed or empty
            return torch.zeros(0, device=self.device)

        # (옵션) VAD
        if self.apply_vad:
            duration_sec = waveform.shape[-1] / sr
            # "긴 세그먼트" 기준 예시
            if duration_sec > self.vad_min_sec:
                waveform = remove_silence(waveform, sr, self.top_db)
                if waveform.shape[-1] == 0:
                    return torch.zeros(0, device=self.device)

        # 정규화
        waveform = normalize_waveform(waveform, max_peak=0.9)

        # 길이 제한
        waveform = pad_or_trim(waveform, self.max_length)

        # MFCC or 원본
        if self.extract_feature and not self.return_waveform:
            features = extract_features_mfcc(waveform, sr, self.n_mfcc, device=self.device)
            return features  # shape: [1, n_mfcc, T]
        else:
            # waveform 반환
            return waveform

####################################
# 4. main() 함수
####################################
def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    logger.info(f"Loaded config from {args.config}: {config}")

    # device 설정 (CPU or GPU)
    device_str = config.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("Requested CUDA but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # 전처리 객체 생성
    preprocessor = LightASRPreprocessor(config, device=device)
    logger.info("Initialized LightASRPreprocessor")

    # 파일 목록 불러오기
    #  1) config에 "audio_files" key가 있다면 그걸로 사용
    #  2) 또는 args.audio_list 로 텍스트파일 읽기
    file_list = []
    if "audio_files" in config:
        file_list = config["audio_files"]
        if not isinstance(file_list, list):
            logger.error("audio_files in config must be a list!")
            file_list = []
    elif args.audio_list is not None:
        if os.path.exists(args.audio_list):
            with open(args.audio_list, "r") as f:
                file_list = [line.strip() for line in f if line.strip()]
        else:
            logger.error(f"audio-list file not found: {args.audio_list}")
    else:
        # fallback -> single 'sample.wav'
        logger.warning("No audio list provided. Using default sample.wav.")
        file_list = ["sample.wav"]

    # 출력 디렉토리
    output_dir = args.output_dir if args.output_dir else config.get("output_dir", "./asr_outputs")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Processing {len(file_list)} files. Output dir={output_dir}")

    # 전처리 수행
    for idx, file_path in enumerate(file_list):
        with torch.no_grad():
            result = preprocessor(file_path)
        # shape 보고 저장(예시)
        if result.numel() == 0:
            logger.info(f"[{idx+1}/{len(file_list)}] {file_path} -> Empty result (possibly silent).")
            continue

        # 예시: 텐서를 .pt로 저장
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(output_dir, base_name + ".pt")
        torch.save(result.cpu(), save_path)
        logger.info(f"[{idx+1}/{len(file_list)}] Saved processed features: {save_path}")

    # TorchScript로 변환 & 저장 (옵션)
    scripted_preprocessor = torch.jit.script(preprocessor)
    script_path = os.path.join(output_dir, "light_asr_preprocessor.pt")
    scripted_preprocessor.save(script_path)
    logger.info(f"Saved TorchScript module to {script_path}")

if __name__ == "__main__":
    main()