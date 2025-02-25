import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_timing_data(json_path):
    """JSON file에서 데이터 가져오기"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["structural_groups"]

def plot_inference_times(data, output_dir, use_total_time=True, top_n_labels=20):
    """추론 시간을 시각화"""
    plt.figure(figsize=(15, 20))  # 세로 크기 증가
    
    # 색상 매핑
    colors = {
        'preprocessor_llama_model': 'skyblue',
        'llm_base_model': 'royalblue',
        'preprocessor_vision_model': 'lightgreen',
        'llm_vision_model': 'forestgreen',
        'preprocessor_projection': 'orange',
        'llm_projection': 'red'
    }
    
    # 데이터 준비
    all_names = []
    all_times = []
    all_colors = []
    
    # 각 structural group별로 데이터 처리
    for group_name, group_data in data.items():
        for layer_name, stats in group_data.items():
            all_names.append(layer_name)
            all_times.append(stats["total_time"] if use_total_time else stats["mean_time"])
            all_colors.append(colors.get(group_name, 'gray'))
    
    # 막대 그래프 그리기
    bars = plt.barh(range(len(all_names)), all_times, color=all_colors)
    
    # 상위 N개의 시간값에 대해서만 레이블 표시
    time_threshold = sorted(all_times, reverse=True)[min(top_n_labels, len(all_times)-1)]
    
    # y축 레이블 설정 (상위 N개만 표시)
    y_ticks = range(len(all_names))
    y_labels = [''] * len(all_names)  # 빈 문자열로 초기화
    for i, v in enumerate(all_times):
        if v >= time_threshold:
            y_labels[i] = all_names[i]  # 상위 N개만 레이블 표시
            plt.text(v, i, f' {v:.2f}ms', va='center')
    
    plt.yticks(y_ticks, y_labels, fontsize=8)
    
    # 그래프 스타일링
    time_type = "Total" if use_total_time else "Mean"
    plt.title(f'Layer-wise {time_type} Inference Time Analysis')
    plt.xlabel('Time (ms)')
    
    # 범례 추가
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=name) 
                      for name, color in colors.items()]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 저장
    plt.tight_layout()
    filename = f'layer_wise_{time_type.lower()}_time.png'
    plt.savefig(str(Path(output_dir) / filename), bbox_inches='tight', dpi=300)
    plt.close()

def plot_structural_subplots(data, output_dir, use_total_time=True, top_n_labels=5):
    """각 structural group별로 subplot으로 시각화"""
    n_groups = len(data)
    
    # 각 그룹의 데이터 크기 확인
    group_sizes = [len(group_data) for group_data in data.values()]
    
    # 각 subplot의 높이를 데이터 크기에 비례하여 설정
    heights = [max(3, size * 0.2) for size in group_sizes]  # 최소 높이 3
    total_height = sum(heights)
    
    # 서브플롯 생성
    fig = plt.figure(figsize=(15, total_height))
    gs = fig.add_gridspec(n_groups, 1, height_ratios=heights)
    axs = [fig.add_subplot(gs[i]) for i in range(n_groups)]
    
    # 색상 매핑
    colors = {
        'preprocessor_llama_model': 'skyblue',
        'llm_base_model': 'royalblue',
        'preprocessor_vision_model': 'lightgreen',
        'llm_vision_model': 'forestgreen',
        'preprocessor_projection': 'orange',
        'llm_projection': 'red'
    }
    
    for idx, (group_name, group_data) in enumerate(data.items()):
        ax = axs[idx]
        
        # 데이터 준비
        names = list(group_data.keys())
        times = [group_data[name]["total_time"] if use_total_time else group_data[name]["mean_time"] 
                for name in names]
        
        # 막대 그래프 그리기
        bars = ax.barh(range(len(names)), times, color=colors.get(group_name, 'gray'))
        
        # 모든 레이블 표시
        y_ticks = range(len(names))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(names, fontsize=8)
        
        # 시간 레이블 추가
        for i, v in enumerate(times):
            ax.text(v, i, f' {v:.2f}ms', va='center')
        
        # 서브플롯 제목
        total_time = sum(times)
        ax.set_title(f'{group_name} (Total: {total_time:.2f}ms)')
        
        # x축 레이블
        ax.set_xlabel('Time (ms)')
    
    # 전체 그래프 스타일링
    time_type = "Total" if use_total_time else "Mean"
    
    # 저장
    plt.tight_layout()
    filename = f'structural_groups_{time_type.lower()}_time.png'
    plt.savefig(str(Path(output_dir) / filename), bbox_inches='tight', dpi=300)
    plt.close()

def main():
    output_dir = "inference_times"
    json_path = Path(output_dir) / "layer_times_detailed.json"
    
    if not json_path.exists():
        print(f"Error: Could not find timing data at {json_path}")
        return
    
    # 데이터 로드
    structural_groups = load_timing_data(json_path)
    
    # 기존 전체 레이어 시각화
    plot_inference_times(structural_groups, output_dir, use_total_time=False)
    plot_inference_times(structural_groups, output_dir, use_total_time=True)
    
    # 구조별 서브플롯 시각화 추가
    plot_structural_subplots(structural_groups, output_dir, use_total_time=False)
    plot_structural_subplots(structural_groups, output_dir, use_total_time=True)
    
    print(f"Visualizations saved in {output_dir}:")
    print("- layer_wise_mean_time.png: 전체 레이어 평균 시간")
    print("- layer_wise_total_time.png: 전체 레이어 총 시간")
    print("- structural_groups_mean_time.png: 구조별 평균 시간")
    print("- structural_groups_total_time.png: 구조별 총 시간")

if __name__ == "__main__":
    main()
