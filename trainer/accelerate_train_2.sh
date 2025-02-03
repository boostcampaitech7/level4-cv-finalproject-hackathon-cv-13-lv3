export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface/transformers"

# export NVIDIA_TF32_OVERRIDE=1  # 모든 CUDA 연산에 TF32 적용
# export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"  # 비동기 메모리 할당

train_config="./configs/train_stage2.yaml"

# 기본 옵션 설정
dryrun=""

# 명령줄 인자 파싱
if [ "$1" = "dryrun" ]; then
    dryrun="--dryrun"
fi
# 저장된 config 파일로 실행
accelerate launch \
    --config_file ./configs/accelerate_config.yaml \
    train.py \
    --cfg-path ${train_config} \
    ${dryrun} \
    --accelerate

# /data/jyp/level4-cv-finalproject-hackathon-cv-13-lv3/trainer/outputs_stage1_only/Salmonn-whisper-large-v3-turbo-5epoch/checkpoint_best.pth
# 위 링크에 있는 체크포인트 파일을 ckpt로 넣은 train stage2 학습 예정
# llama / whisper v3 turbo / BEATs
# 

