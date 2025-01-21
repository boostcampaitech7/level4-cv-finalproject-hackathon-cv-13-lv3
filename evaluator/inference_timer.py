import os
import json
import torch
import numpy as np
from collections import defaultdict

class InferenceTimer:
    def __init__(self):
        self.layer_times = {}
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        
    def hook_fn_start(self, module, input):
        self.starter.record()
        
    def hook_fn_end(self, name):
        def hook(module, input, output):
            if name not in self.layer_times:
                self.layer_times[name] = []
            self.ender.record()
            torch.cuda.synchronize()
            cur_time = self.starter.elapsed_time(self.ender)
            self.layer_times[name].append(cur_time)
        return hook

    def register_module_hooks(self, module, prefix=""):
        """재귀적으로 모든 하위 모듈에 대한 hook을 등록"""
        for name, submodule in module.named_children():
            full_name = f"{prefix}_{name}" if prefix else name
            
            # 현재 모듈에 대한 hook 등록
            submodule.register_forward_pre_hook(self.hook_fn_start)
            submodule.register_forward_hook(self.hook_fn_end(full_name))
            
            # 자식 모듈에 대한 hook 재귀적으로 등록
            self.register_module_hooks(submodule, full_name)
    
    def analyze_model_structure(self, model):
        """모델 구조를 분석하여 구조별 컴포넌트 식별"""
        structural_groups = defaultdict(list)
        
        # 모든 모듈 분석
        for name, module in model.named_modules():
            if name:
                # 최상위 컴포넌트 이름 가져오기 (첫 번째 점 이전의 부분)
                top_component = name.split('.')[0]
                structural_groups[top_component].append(name)

        return dict(structural_groups)

    def save_measurement(self, output_dir="inference_times", models=None):
        """타이밍 측정 결과를 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 각 레이어에 대한 statistics 계산
        timing_stats = {}
        for name, times in self.layer_times.items():
            timing_stats[name] = {
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
                "min_time": float(np.min(times)),
                "max_time": float(np.max(times)),
                "total_time": float(np.sum(times)),
                "calls": len(times)
            }
        
        structural_groups = {}
        
        if models:
            # 각 모델별로 구조 분석
            for model_type, model in models.items():
                struct_map = self.analyze_model_structure(model)
                
                # 구조별 그룹에 추가
                for component, patterns in struct_map.items():
                    group_name = f"{model_type}_{component}"
                    structural_groups[group_name] = {}
                    for name, stats in timing_stats.items():
                        if any(pattern in name for pattern in patterns):
                            structural_groups[group_name][name] = stats

        # 결과 저장
        results = {
            "structural_groups": structural_groups
        }
        
        json_path = os.path.join(output_dir, "layer_times_detailed.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # 구조별 summary 출력
        print("\nStructural Groups Analysis:")
        for component, layers in structural_groups.items():
            if layers:
                total_time = sum(stats["total_time"] for stats in layers.values())
                print(f"\n{component} total time: {total_time:.2f}ms")
                print("Top 5 most time-consuming layers:")
                sorted_layers = sorted(layers.items(), key=lambda x: x[1]["total_time"], reverse=True)[:5]
                for name, stats in sorted_layers:
                    print(f"  {name}: {stats['total_time']:.2f}ms ({stats['mean_time']:.2f}ms ± {stats['std_time']:.2f}ms per call)")