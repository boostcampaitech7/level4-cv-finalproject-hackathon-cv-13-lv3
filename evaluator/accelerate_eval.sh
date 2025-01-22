export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface/transformers"

# 기본 옵션 설정
num_gpus=2
mode=""

# 명령줄 인자 파싱
if [ "$1" = "mode" ]; then
    mode="--mode"
fi

# 명령어 실행
accelerate launch --config_file ./accelerate_config.yaml accelerate_evaluate_salmonn.py \
    --mode submission_asr ${mode} \
    --cfg-path salmonn_eval_config.yaml \
    