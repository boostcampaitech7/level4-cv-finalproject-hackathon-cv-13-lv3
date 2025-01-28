export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface/transformers"

# export NVIDIA_TF32_OVERRIDE=1  # 모든 CUDA 연산에 TF32 적용
# export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"  # 비동기 메모리 할당

train_config="./configs/train_stage1.yaml"

# 기본 옵션 설정
dryrun=""

# 명령줄 인자 파싱
if [ "$1" = "dryrun" ]; then
    dryrun="--dryrun"
fi
# 저장된 config 파일로 실행
accelerate launch \
    # --gradient_accumulation_steps=2 \
    # --split_batches \
    --config_file ./configs/accelerate_config.yaml \
    train.py \
    --accumulate_grad_steps 2 \
    --cfg-path ${train_config} \
    ${dryrun} \
    --accelerate