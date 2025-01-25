export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface/transformers"

train_config="./configs/train_stage1.yaml"

# 기본 옵션 설정
dryrun=""

# 명령줄 인자 파싱
if [ "$1" = "dryrun" ]; then
    dryrun="--dryrun"
fi
# 저장된 config 파일로 실행
accelerate launch --config_file ./configs/accelerate_config.yaml train.py \
    --cfg-path ${train_config} \
    ${dryrun} \
    --accelerate