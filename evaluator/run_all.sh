#!/bin/bash

export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface/transformers"

# 실행할 명령어 리스트
commands=(
    "accelerate launch --config_file ./configs/accelerate_config_1.yaml accelerate_evaluate_salmonn.py --mode submission_aac --cfg-path salmonn_eval_unslothGemma.yaml"
    "accelerate launch --config_file ./configs/accelerate_config_2.yaml accelerate_evaluate_salmonn.py --mode submission_aac --cfg-path salmonn_eval_whisperv3.yaml"
    "accelerate launch --config_file ./configs/accelerate_config_3.yaml accelerate_evaluate_salmonn.py --mode submission_asr --cfg-path salmonn_eval_whisperv3turbo.yaml"
    "accelerate launch --config_file ./configs/accelerate_config_4.yaml accelerate_evaluate_salmonn.py --mode submission_aac --cfg-path salmonn_eval_whisperv3turbo.yaml"
)


# 순차적으로 실행
for cmd in "${commands[@]}"; do
    echo "실행 중: $cmd"
    eval $cmd
    if [ $? -ne 0 ]; then
        echo "오류 발생: $cmd"
        exit 1
    fi
    echo "완료: $cmd"
done

echo "모든 작업이 완료되었습니다."
