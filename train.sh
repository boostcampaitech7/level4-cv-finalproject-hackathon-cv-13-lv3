train_config="./configs/train_stage1.yaml"

# 기본 옵션 설정
num_gpus=2
dryrun=""

# 명령줄 인자 파싱
if [ "$1" = "dryrun" ]; then
    dryrun="--dryrun"
fi

# 명령어 실행
torchrun --nproc_per_node=${num_gpus} train.py \
    --cfg-path ${train_config} \
    ${dryrun} 