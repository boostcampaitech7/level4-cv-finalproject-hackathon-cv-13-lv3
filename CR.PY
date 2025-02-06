import json
import os
import torch
import torchaudio
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
from multiprocessing import Manager
from tqdm import tqdm

import argparse


#####################################
# Dataset 및 collate_fn 정의
#####################################
class AudioCaptionDataset(Dataset):
    def __init__(self, entries, dirname, target_task):
        self.entries = entries
        self.dirname = dirname
        self.task = target_task
        
    def __len__(self):
        return len(self.entries)
    
    def get_task(self):
        return self.task

    def get_dirname(self):
        return self.dirname
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio_path = os.path.join(self.dirname, entry["path"])
        text = entry["text"]
        try:
            # torchaudio로 오디오 로드 (48kHz)
            waveform, sr = torchaudio.load(audio_path)
            # CLAP은 mono만 지원하므로, 다채널인 경우 평균하여 mono로 변환
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            # waveform을 numpy array로 변환 (processor가 numpy array를 입력으로 받음)
            waveform = waveform.squeeze(0).numpy()
        except Exception as e:
            print(f"오디오 로드 실패: {audio_path} - {e}")
            waveform = None
        return {"audio_path": entry["path"], "text": text, "waveform": waveform}

def collate_fn(batch, max_length_cap=480000):
    # # 오디오 로드 실패한 항목 제거
    # batch = [item for item in batch if item["waveform"] is not None]
    # if len(batch) == 0:
    #     return None
    # 각 샘플의 길이(샘플 수) 계산 후, 배치 내 최대 길이를 cap과 비교하여 결정

    lengths = [item["waveform"].shape[0] for item in batch]
    max_len = min(max(lengths), max_length_cap)
    
    audios = []
    texts = []
    paths = []
    for item in batch:
        waveform = torch.tensor(item["waveform"], dtype=torch.float32)
        l = waveform.shape[0]
        # 길이가 max_len보다 길면 자르고, 짧으면 오른쪽 패딩
        if l > max_len:
            waveform = waveform[:max_len]
        else:
            pad_amount = max_len - l
            waveform = F.pad(waveform, (0, pad_amount))
        audios.append(waveform.numpy())
        texts.append(item["text"])
        paths.append(item["audio_path"])
    return {"audios": audios, "texts": texts, "paths": paths}

#####################################
# GPU별 배치 처리 함수 (worker)
#####################################
def process_worker(rank, world_size, dataset, batch_size, similarity_threshold, return_list, device_ids):

    # GPU별 로그 파일 경로 설정
    log_file_path = f"remove_log_{rank}.txt"
    # 선택한 GPU 설정
    device = torch.device(f"cuda:{device_ids[rank]}") if torch.cuda.is_available() else "cpu"
    
    # 각 워커별로 processor와 모델 로드
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    model.eval()
    
    # DistributedSampler를 사용하여 데이터를 각 GPU에 분산
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    
    local_results = []
    remove_results = []
    pbar = tqdm(total=len(dataloader), desc=f"[GPU {device_ids[rank]}]", position=rank)
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        for batch in dataloader:
            if batch is None:
                pbar.update(1)
                continue
            # processor에 배치 단위로 오디오와 텍스트 전달
            inputs = processor(
                text=batch["texts"],
                audios=batch["audios"],
                return_tensors="pt",
                sampling_rate=48000,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # 출력 logits_per_audio의 shape는 (batch_size, batch_size)
                # 1:1 매칭은 대각선 요소이므로, diag()로 추출
                sims = torch.diag(outputs.logits_per_audio).cpu().numpy()
            
            # 각 샘플별로 결과 처리
            for i, sim in enumerate(sims):
                if sim >= similarity_threshold:
                    local_results.append({
                        "path": batch["paths"][i],
                        "text": batch["texts"][i],
                        "task": dataset.get_task()
                    })
                    # tqdm.write(f"[GPU {device_ids[rank]}] 유지: {batch['paths'][i]} - 유사도: {sim:.4f}")
                else:
                    # 임계값 미달인 경우 output 폴더에 심볼릭 링크 생성

                    log_entry = f"{batch['paths'][i]} | {batch['texts'][i]} | Task: {dataset.get_task()} | Similarity: {sim:.4f}\n"
                    log_file.write(log_entry)
                    log_file.flush()  # 즉시 파일에 기록
                    tqdm.write(f"[GPU {device_ids[rank]}] 제거: {batch['paths'][i]} - 유사도: {sim:.4f}")
                    # basename = os.path.basename(batch["paths"][i])
                    # similarity_str = f"{sim:.4f}"
                    # text_str = batch["texts"][i].replace(" ", "_")[:20]
                    # target_link = os.path.join("output", f"{similarity_str}_{text_str}_{basename}")
                    # if not os.path.exists(target_link):
                    #     try:
                    #         os.symlink(os.path.join(dataset.get_dirname(), batch["paths"][i]), target_link)
                    #     except Exception as e:
                    #         tqdm.write(f"[GPU {device_ids[rank]}] 심볼릭 링크 생성 실패: {batch['paths'][i]} - {e}")
            pbar.update(1)
    pbar.close()
    return_list.extend(local_results)

#####################################
# main 함수: 단일 또는 다중 GPU 실행
#####################################
def main():
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="CLAP Audio Captioning")
    parser.add_argument("--input_json_path", type=str, required=True, help="입력 JSON 파일 경로")
    parser.add_argument("--output_json_path", type=str, default="output.json", help="출력 JSON 파일 경로")
    parser.add_argument("--similarity_threshold", type=float, default=1, help="유사도 임계값")
    parser.add_argument("--target_task", type=str, default="audiocaption", help="처리할 태스크")
    parser.add_argument("--batch_size", type=int, default=12, help="DataLoader 배치 사이즈")
    parser.add_argument("--world_size", type=int, default=2, help="사용 GPU 수")

    args = parser.parse_args()

    # 설정 (argparse에서 가져온 값 사용)
    input_json_path = args.input_json_path
    output_json_path = args.output_json_path
    similarity_threshold = args.similarity_threshold
    target_task = args.target_task
    batch_size = args.batch_size
    world_size = args.world_size
    device_ids = [0, 1] if world_size == 2 else [1]
    
    # 출력 디렉토리 생성 (심볼릭 링크용)    
    os.makedirs("output", exist_ok=True)
    
    # JSON 데이터 로드 및 target_task 필터링
    with open(input_json_path, "r") as f:
        data = json.load(f)

    filtered_entries = [entry for entry in data["annotation"] if entry.get("task") == target_task]
    # 기존 JSON에서 target_task 항목 제거 (최종 결과에 추가할 예정)
    data["annotation"] = [entry for entry in data["annotation"] if entry.get("task") != target_task]

    dirname = os.path.dirname(input_json_path)
    dataset = AudioCaptionDataset(filtered_entries, dirname, target_task)
    
    # 단일 GPU와 다중 GPU 모두 지원: world_size가 1이면 단일 프로세스로 실행
    if world_size > 1:
        manager = Manager()
        return_list = manager.list()
        mp.spawn(
            process_worker,
            args=(world_size, dataset, batch_size, similarity_threshold, return_list, device_ids),
            nprocs=world_size,
            join=True
        )
        result_annotations = list(return_list)
    else:
        # 단일 GPU 실행
        return_list = []
        process_worker(0, 1, dataset, batch_size, similarity_threshold, return_list, device_ids)
        result_annotations = return_list
    
    # 최종 결과를 기존 JSON에 추가하고 저장
    data["annotation"].extend(result_annotations)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"최종 결과: {len(result_annotations)}개 항목 저장됨")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
