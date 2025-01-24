export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface/transformers"

# 기본 옵션 설정
num_gpus=2
dryrun=""

# 명령줄 인자 파싱
if [ "$1" = "dryrun" ]; then
    dryrun="--dryrun"
fi

# 명령어 실행
torchrun --nproc_per_node=${num_gpus} --master_port 29501 evaluate_salmonn.py \
    --cfg-path salmonn_eval_config.yaml \
    ${dryrun} 