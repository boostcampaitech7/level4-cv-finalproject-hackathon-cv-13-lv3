export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface/transformers"

# 기본값 설정
MODE=${2:-submission_asr}  # 두 번째 인자를 mode 값으로 사용, 없으면 submission_asr

# 명령어 실행
accelerate launch --config_file ./configs/accelerate_config.yaml \
    accelerate_evaluate_salmonn.py \
    --mode "$MODE" \
    --cfg-path salmonn_eval_config.yaml
