export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface/transformers"

# export NVIDIA_TF32_OVERRIDE=1  # 모든 CUDA 연산에 TF32 적용
# export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"  # 비동기 메모리 할당

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