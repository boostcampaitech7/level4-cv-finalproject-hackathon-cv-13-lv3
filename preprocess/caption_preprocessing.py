import os
import json
import torch
import librosa
import argparse
import yaml
import logging
from torch import nn

##################################
# (1) 로깅 설정
##################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

##################################
# (2) CLI + YAML
##################################
def parse_args():
    parser = argparse.ArgumentParser(description="Caption Preprocessing")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--audio-list", type=str, default=None, help="Text file containing audio file paths")
    parser.add_argument("--captions-json", type=str, default=None, help="JSON file containing {audio_path: caption} pairs (optional)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save processed results")
    return parser.parse_args()

def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

##################################
# (3) 공통 함수 (from common_preprocessing)
##################################
def load_audio(path, root_dir='/data/dataset', target_samplerate=16000):
    import torchaudio
    import numpy as np

    # 실제 구현: load_audio 함수
    audio_path = os.path.join(root_dir, path) if root_dir else path
    if not os.path.exists(audio_path):
        logging.warning(f"Audio file not found: {audio_path}")
        return torch.zeros(1,1), target_samplerate

    waveform, sr = torchaudio.load(audio_path)
    if sr != target_samplerate:
        resampler = torchaudio.transforms.Resample(sr, target_samplerate)
        waveform = resampler(waveform)
        sr = target_samplerate

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, sr

def normalize_waveform(waveform, max_peak=0.95):
    peak = waveform.abs().max()
    if peak > 0:
        ratio = max_peak / peak
        waveform = waveform * ratio
    return waveform

def remove_silence(waveform, sr, top_db=20):
    import numpy as np
    # CPU 기반 librosa
    waveform_np = waveform.squeeze().float().cpu().numpy()
    intervals = librosa.effects.split(y=waveform_np, top_db=top_db)
    if len(intervals) == 0:
        return torch.zeros(1,1)
    trimmed = []
    for start, end in intervals:
        trimmed.append(waveform_np[start:end])
    trimmed_np = np.concatenate(trimmed)
    trimmed_tensor = torch.from_numpy(trimmed_np).unsqueeze(0)
    return trimmed_tensor

def pad_or_trim(waveform, max_length):
    import torch
    length = waveform.shape[-1]
    if length > max_length:
        waveform = waveform[..., :max_length]
    else:
        pad_size = max_length - length
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    return waveform

# ##################################
# # (4) 억양 분류 + TTS (스텁)
# ##################################
# def classify_accent(waveform: torch.Tensor, sr: int) -> str:
#     mean_amp = float(waveform.mean().abs())
#     if mean_amp > 0.01:
#         return "US"
#     else:
#         return "IN"

# def synthesize_tts(text: str, accent: str="US") -> torch.Tensor:
#     sr = 16000
#     duration_sec = 2.0
#     length = int(sr * duration_sec)

#     import math
#     t = torch.linspace(0, duration_sec, steps=length)
#     freq = 220.0 if accent=="US" else 260.0
#     waveform = 0.1*torch.sin(2*math.pi*freq*t).unsqueeze(0)
#     return waveform

def clean_caption(caption: str) -> str:
    import re
    no_content_pattern = re.compile(r"(no sound|silent audio|배경음 없음)", re.IGNORECASE)
    caption = re.sub(no_content_pattern, "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption

##################################
# (5) CaptionPreprocessor
##################################
class CaptionPreprocessor(nn.Module):
    def __init__(self, config):
        super(CaptionPreprocessor, self).__init__()
        self.root_dir = config.get('root_dir', '/data/dataset')
        self.target_sr = config.get('target_sr', 16000)
        self.max_length = config.get('max_length', 16000 * 30)
        self.apply_vad = config.get('apply_vad', False)
        self.vad_top_db = config.get('vad_top_db', 20)
        self.normalize = config.get('normalize', True)
        self.min_length = config.get('min_length', 1600)
        self.clean_text_flag = config.get('clean_text', True)

        self.accent_classify = config.get('accent_classify', False)
        self.filter_nonus = config.get('filter_nonus', False)
        self.tts_augment = config.get('tts_augment', False)

    def forward(self, relative_path: str, caption: str = ""):
        waveform, sr = load_audio(
            path=relative_path,
            root_dir=self.root_dir,
            target_samplerate=self.target_sr
        )
        if waveform.shape[-1] == 0:
            # 파일이 없거나 로드 실패
            return torch.zeros(1,1), ""

        if self.apply_vad:
            waveform = remove_silence(waveform, sr, self.vad_top_db)

        if waveform.shape[-1] < self.min_length:
            return torch.zeros(1,1), ""

        if self.normalize:
            waveform = normalize_waveform(waveform, max_peak=0.95)

        waveform = pad_or_trim(waveform, self.max_length)

        # 억양 분류
        if self.accent_classify:
            accent = classify_accent(waveform, sr)
            if accent != "US":
                if self.filter_nonus:
                    return torch.zeros(1,1), ""
                if self.tts_augment:
                    tts_wave = synthesize_tts(caption, accent="US")
                    waveform = torch.cat([waveform, tts_wave], dim=-1)
                    waveform = pad_or_trim(waveform, self.max_length)

        # 텍스트 전처리
        new_caption = caption
        if self.clean_text_flag and caption:
            new_caption = clean_caption(caption)

        return waveform, new_caption

##################################
# (6) main(): 파일 loop, 전처리
##################################
def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {args.config}")

    # 옵션: device 설정, 여기서는 CPU assumed
    preprocessor = CaptionPreprocessor(config)

    # audio_list
    file_list = []
    if args.audio_list and os.path.exists(args.audio_list):
        with open(args.audio_list, 'r') as f:
            file_list = [line.strip() for line in f if line.strip()]
    else:
        logger.warning("No audio_list provided or file not found. Use default sample list.")
        file_list = ["WavCaps/sample_data/example.wav"]

    # captions json (optional)
    caption_dict = {}
    if args.captions_json and os.path.exists(args.captions_json):
        with open(args.captions_json, 'r') as jf:
            caption_dict = json.load(jf)
    logger.info(f"Loaded {len(caption_dict)} captions from {args.captions_json}")

    # output dir
    output_dir = args.output_dir or config.get("output_dir", "./caption_outputs")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Start processing {len(file_list)} files.")
    for idx, fpath in enumerate(file_list):
        raw_caption = caption_dict.get(fpath, "")  # 혹은 basename key
        wave, cap = preprocessor(fpath, raw_caption)

        # 결과 shape, caption 확인
        logger.info(f"[{idx+1}/{len(file_list)}] {fpath}: wave shape={wave.shape}, caption='{cap}'")

        # 필요하다면 저장
        base = os.path.splitext(os.path.basename(fpath))[0]
        wave_save = os.path.join(output_dir, base + "_wave.pt")
        torch.save(wave, wave_save)
        logger.info(f"Saved wave to {wave_save}")

        # caption은 별도 txt/json에 기록 가능
        # ...

    # TorchScript 저장 (옵션)
    scripted = torch.jit.script(preprocessor)
    script_path = os.path.join(output_dir, "caption_preprocessor.pt")
    scripted.save(script_path)
    logger.info(f"Saved TorchScript to {script_path}")


if __name__ == "__main__":
    main()