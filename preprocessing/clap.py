#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
전처리 스크립트:
SALMONNDataset을 이용하여 오디오 파일을 읽고, Whisper를 통한 spectrogram과 함께
CLAP 임베딩을 미리 추출하여 .pt 파일로 저장함.

사용법: python preprocess_clap.py --prefix "/data/dataset" --ann_path "/data/dataset/annotation.json" --whisper_path "openai/whisper-tiny" --output_dir "/data/dataset/preprocessed_clap"
"""

import os
import argparse
import json
import torch
import soundfile as sf
import numpy as np
import librosa
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperFeatureExtractor, ClapModel, ClapProcessor

# 전역 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Whisper 모델 로드 (음성 전처리용)
def load_whisper_processor(whisper_path):
    return WhisperFeatureExtractor.from_pretrained(whisper_path)

# CLAP 모델 및 프로세서 로드
def load_clap_models():
    clap_model = ClapModel.from_pretrained("laion/larger_clap_music").to(device)
    clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
    return clap_model, clap_processor

def extract_clap_embedding(audio, sr, clap_model, clap_processor, target_sr=48000):
    """
    오디오 데이터를 받아 CLAP 임베딩을 추출함.
    - 필요시 샘플링 레이트(target_sr)로 재조정함.
    """
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    inputs = clap_processor(audios=audio, return_tensors="pt", sampling_rate=sr)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_features = clap_model.get_audio_features(**inputs)
    # squeeze해서 (feature_dim,) 형태로 반환
    return audio_features.squeeze(0)

def process_sample(ann, prefix, whisper_processor, clap_model, clap_processor):
    """
    하나의 샘플을 처리하여 전처리된 결과를 반환함.
    Whisper를 이용해 spectrogram을 추출하고, CLAP 임베딩을 미리 계산함.
    """
    audio_path = os.path.join(prefix, ann["path"]).replace("//", "/")
    try:
        audio, sr = sf.read(audio_path)
    except Exception as e:
        print(f"Failed to load {audio_path} ({e}). Using 0-th sample instead.")
        audio_path = os.path.join(prefix, ann["path"])  # fallback
        audio, sr = sf.read(audio_path)
    
    if len(audio.shape) == 2:  # stereo -> mono
        audio = audio[:, 0]

    # 1초 미만이면 패딩
    if len(audio) < sr:
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    
    # Whisper 모델의 sampling_rate에 맞게 resample
    if sr != whisper_processor.sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=whisper_processor.sampling_rate)
        sr = whisper_processor.sampling_rate

    # 최대 30초로 자르기
    audio = audio[: sr * 30]

    # Whisper spectrogram 추출
    spectrogram = whisper_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()

    # CLAP 임베딩 미리 추출 (CLAP는 보통 48kHz 사용)
    clap_embedding = extract_clap_embedding(audio, sr, clap_model, clap_processor, target_sr=48000)

    text = ann["text"]
    task = ann.get("task", "asr")
    Q = ann.get("Q", "")

    return {
        "spectrogram": spectrogram,
        "raw_wav": audio,
        "text": text,
        "task": task,
        "Q": Q,
        "id": ann["path"],
        "clap_embedding": clap_embedding,
    }

def preprocess_and_save(prefix, ann_path, whisper_path, output_dir):
    # 전처리에 필요한 모델 및 프로세서 로드
    whisper_processor = load_whisper_processor(whisper_path)
    clap_model, clap_processor = load_clap_models()

    # 어노테이션 파일 로드
    data = json.load(open(ann_path, "r"))["annotation"]
    # GigaSpeech 데이터 제거 (필요한 경우)
    annotations = [item for item in data if 'GigaSpeech' not in item['path']]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_samples = len(annotations)
    print(f"전체 샘플 수: {num_samples}")

    for idx, ann in enumerate(annotations):
        sample = process_sample(ann, prefix, whisper_processor, clap_model, clap_processor)
        # 파일명으로 id를 사용 (특수문자 대체)
        file_id = sample["id"].replace("/", "_").replace("\\", "_")
        out_path = os.path.join(output_dir, f"{file_id}.pt")
        torch.save(sample, out_path)
        if (idx + 1) % 10 == 0 or (idx + 1) == num_samples:
            print(f"{idx + 1}/{num_samples} 샘플 처리 완료: {out_path}")

def main(args):
    preprocess_and_save(args.prefix, args.ann_path, args.whisper_path, args.output_dir)
    print("전처리 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLAP 기반 오디오 전처리 스크립트")
    parser.add_argument("--prefix", type=str, required=True,
                        help="데이터셋의 root 경로 (예: /data/dataset)")
    parser.add_argument("--ann_path", type=str, required=True,
                        help="어노테이션 파일 경로 (json 형식)")
    parser.add_argument("--whisper_path", type=str, required=True,
                        help="Whisper 모델 경로 (예: openai/whisper-tiny)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="전처리된 결과를 저장할 출력 폴더")
    args = parser.parse_args()
    main(args)