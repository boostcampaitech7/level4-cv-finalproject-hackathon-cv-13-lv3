import json
import os
import torch
import librosa
from transformers import ClapModel, ClapProcessor
import torch.multiprocessing as mp
from multiprocessing import Manager
from tqdm import tqdm

# GPU별 처리 함수 (각 프로세스에서 호출)
def worker(rank, world_size, entries_subset, device_ids, similarity_threshold, return_list):
    # 선택한 GPU 설정
    device = torch.device(f"cuda:{device_ids[rank]}") if torch.cuda.is_available() else "cpu"
    
    # CLAP 모델 및 프로세서 로드 (각 프로세스마다 독립적)
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    model.eval()
    
    local_results = []
    
    # tqdm 진행바를 사용 (각 프로세스 별로 표시, position을 rank로 지정)
    for entry in tqdm(entries_subset, desc=f"[GPU {device_ids[rank]}] Processing", position=rank):
        audio_path = entry["path"]
        text = entry["text"]
        
        try:
            # 오디오 로드 (48kHz, mono)
            waveform, sr = librosa.load(audio_path, sr=48000, mono=True)
        except Exception as e:
            print(f"[GPU {device_ids[rank]}] 오디오 로드 실패: {audio_path} - {str(e)}")
            continue

        # processor를 이용해 오디오와 텍스트를 한 번에 처리 (배치 크기 1)
        inputs = processor(
            text=[text],
            audios=[waveform],
            return_tensors="pt",
            sampling_rate=48000,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            similarity = outputs.logits_per_audio.item()

        if similarity >= similarity_threshold:
            local_results.append(entry)
            tqdm.write(f"[GPU {device_ids[rank]}] 유지: {audio_path} - 유사도: {similarity:.4f}")
        else:
            basename = os.path.basename(audio_path)
            similarity_str = f"{similarity:.4f}"
            text_str = text.replace(" ", "_")
            target_link = os.path.join("output", f"{similarity_str}_{text_str}_{basename}")
            tqdm.write(f"[GPU {device_ids[rank]}] 제거: {audio_path} - 유사도: {similarity:.4f}")
            if not os.path.exists(target_link):
                try:
                    os.symlink(os.path.abspath(audio_path), target_link)
                except Exception as e:
                    tqdm.write(f"[GPU {device_ids[rank]}] 심볼릭 링크 생성 실패: {audio_path} - {e}")

    return_list.extend(local_results)

def main():
    # 설정
    input_json_path = "/data/dataset/stage1_valid_05.json"  # 입력 JSON 파일 경로
    output_json_path = "output.json"                         # 출력 JSON 파일 경로
    similarity_threshold = 0.1                               # 유사도 임계값 (필요에 따라 조정)
    target_task = "audiocaption"                             # 처리할 태스크

    # 출력 디렉토리 생성 (심볼릭 링크 생성용)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON 파일의 폴더 경로 보정
    dirname = os.path.dirname(os.path.abspath(input_json_path))
    
    # JSON 데이터 로드
    with open(input_json_path, "r") as f:
        data = json.load(f)
    
    # 처리할 annotation만 필터링 (예: task가 "audiocaption")
    filtered_entries = [entry for entry in data["annotation"] if entry.get("task") == target_task]
    
    # 절대 경로로 변환 (데이터 통일성을 위해)
    for entry in filtered_entries:
        entry["path"] = os.path.join(dirname, entry["path"])
    
    # 원래 JSON 데이터에서 target_task 항목 제거 (최종 결과에 추가할 예정)
    data["annotation"] = [entry for entry in data["annotation"] if entry.get("task") != target_task]
    
    # filtered_entries를 2개의 부분으로 분할 (두 GPU 사용)
    num_entries = len(filtered_entries)
    split_index = num_entries // 2
    subsets = [filtered_entries[:split_index], filtered_entries[split_index:]]
    
    world_size = 2            # 사용 GPU 수 (여기서는 2개)
    device_ids = [0, 1]       # 사용하고자 하는 GPU ID 리스트
    
    # multiprocessing.Manager로 결과 공유 리스트 생성
    manager = Manager()
    return_list = manager.list()
    
    # mp.spawn을 통해 각 GPU에서 병렬 처리
    mp.spawn(
        worker,
        args=(world_size, subsets, device_ids, similarity_threshold, return_list),
        nprocs=world_size,
        join=True
    )
    
    # 모든 프로세스의 결과를 일반 리스트로 변환
    result_annotations = list(return_list)
    
    # 최종 JSON 데이터에 결과 추가 및 저장
    data["annotation"].extend(result_annotations)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"최종 결과: {len(result_annotations)}개 항목 저장됨")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
