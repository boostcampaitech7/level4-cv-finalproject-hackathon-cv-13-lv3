import torch
from torch import nn
import re

from common_preprocessing import (
    load_audio,
    normalize_waveform,
    remove_silence,       
    pad_or_trim
)

def clean_caption(caption: str) -> str:
    no_content_pattern = re.compile(r"(no sound|silent audio|배경음 없음)", re.IGNORECASE)
    caption = re.sub(no_content_pattern, "", caption)

    # 공백 정리
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption

class CaptionPreprocessor(nn.Module):
    def __init__(self, config):
        super(CaptionPreprocessor, self).__init__()
        self.root_dir = config.get('root_dir', '/data/dataset')
        self.target_sr = config.get('target_sr', 16000)
        self.max_length = config.get('max_length', 16000 * 30)  # 30초 제한 예시
        self.apply_vad = config.get('apply_vad', False)
        self.vad_top_db = config.get('vad_top_db', 20)
        self.normalize = config.get('normalize', True)
        self.min_length = config.get('min_length', 1600) # 0.1초(1600)보다 짧으면 제외?

        # 텍스트 라벨 관련
        self.clean_text = config.get('clean_text', True)

    def forward(self, relative_path: str, caption: str = ""):
        # 1. 오디오 로드 (16kHz 모노 변환)
        waveform, sr = load_audio(path=relative_path,
                                  root_dir=self.root_dir,
                                  target_samplerate=self.target_sr)

        # 2. VAD로 무음 제거 (옵션)
        #    캡셔닝에서는 배경음을 살리는 경우가 많아 False가 기본.
        if self.apply_vad:
            waveform = remove_silence(waveform, sr, top_db=self.vad_top_db)

        # 3. 지나치게 짧은 경우(예: 0.1초 미만)는 제외 or 패딩
        if waveform.shape[-1] < self.min_length:
            # 여기서는 그냥 zero tensor 반환 or None 처리
            # 실제론 special handling(데이터셋에서 제외) 가능
            return torch.zeros(1, 1), ""

        # 4. Normalize
        if self.normalize:
            waveform = normalize_waveform(waveform, max_peak=0.95)

        # 5. 길이 제한(예: 30초)
        waveform = pad_or_trim(waveform, max_length=self.max_length)

        # 6. 텍스트 전처리 (캡션)
        if self.clean_text and caption:
            new_caption = clean_caption(caption)
        else:
            new_caption = caption

        return waveform, new_caption


if __name__ == "__main__":
    # 테스트 예시
    config = {
        "root_dir": "/data/dataset",
        "target_sr": 16000,
        "max_length": 16000 * 30,   # 30초
        "apply_vad": False,        # 기본: VAD 끔
        "vad_top_db": 20,
        "normalize": True,
        "min_length": 1600,        # 0.1초
        "clean_text": True
    }

    preprocessor = CaptionPreprocessor(config)

    # TorchScript 호환
    scripted_preprocessor = torch.jit.script(preprocessor)

    example_path = "WavCaps/sample_data/example.wav"
    example_caption = "This clip has no sound, it is silent audio."

    with torch.no_grad():
        wav_tensor, refined_caption = scripted_preprocessor(example_path, example_caption)

    print("Waveform shape:", wav_tensor.shape)
    print("Refined caption:", refined_caption)