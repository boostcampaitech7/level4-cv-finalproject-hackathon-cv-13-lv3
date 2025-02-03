#!/usr/bin/env bash
# run_caption.sh
# 예) 오디오 캡셔닝(WavCaps, Clotho 등) 전처리를 실행하는 스크립트

CONFIG_PATH="../configs/caption_config.yaml"
PY_SCRIPT="../caption_preprocessing.py"

echo "=== Running Caption Preprocessing ==="
echo "Using config: $CONFIG_PATH"

python3 $PY_SCRIPT --config $CONFIG_PATH