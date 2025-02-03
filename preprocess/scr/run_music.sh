#!/usr/bin/env bash
# run_music.sh
# 예) MusicNet 전처리를 실행하는 스크립트

CONFIG_PATH="../configs/music_config.yaml"
PY_SCRIPT="../music_preprocessing.py"

echo "=== Running Music Preprocessing with nnAudio ==="
echo "Using config: $CONFIG_PATH"

python3 $PY_SCRIPT --config $CONFIG_PATH