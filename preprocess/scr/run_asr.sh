#!/usr/bin/env bash
# run_asr.sh
# 예) ASR 전처리를 실행하는 스크립트

CONFIG_PATH="../configs/asr_config.yaml"
PY_SCRIPT="../asr_preprocessing.py"

echo "=== Running ASR Preprocessing ==="
echo "Using config: $CONFIG_PATH"

# python 실행 (예: python3)
python3 $PY_SCRIPT --config $CONFIG_PATH