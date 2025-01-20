import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_timing_data(json_path):
    """Load timing data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def plot_component_total_times(component_groups):
    """Plot total time for each major component"""
    components = []
    times = []
    
    for component, layers in component_groups.items():
        if layers:
            total_time = sum(stats["total_time"] for stats in layers.values())
            components.append(component)
            times.append(total_time)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(components, times)
    plt.title('Total Inference Time by Component')
    plt.xlabel('Component')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('inference_times/component_total_times.png')
    plt.close()

def plot_top_layers_by_component(component_groups):
    """Plot top 10 most time-consuming layers for each component"""
    for component, layers in component_groups.items():
        if not layers:
            continue
            
        # Sort layers by total time
        sorted_layers = sorted(layers.items(), key=lambda x: x[1]["total_time"], reverse=True)[:10]
        layer_names = [name.split('.')[-1] for name, _ in sorted_layers]  # Use only the last part of the name
        times = [stats["total_time"] for _, stats in sorted_layers]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(layer_names, times)
        plt.title(f'Top 10 Most Time-Consuming Layers in {component}')
        plt.xlabel('Layer')
        plt.ylabel('Total Time (ms)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}ms',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'inference_times/{component}_top_layers.png')
        plt.close()

def plot_layer_statistics(timing_stats):
    """Plot mean time vs standard deviation for all layers"""
    means = []
    stds = []
    names = []
    
    for name, stats in timing_stats.items():
        means.append(stats["mean_time"])
        stds.append(stats["std_time"])
        names.append(name)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(means, stds, alpha=0.5)
    
    # Annotate some interesting points
    for i, name in enumerate(names):
        if means[i] > np.mean(means) + np.std(means) or stds[i] > np.mean(stds) + np.std(stds):
            plt.annotate(name.split('.')[-1], (means[i], stds[i]), 
                        xytext=(5, 5), textcoords='offset points')
    
    plt.title('Layer Time Statistics')
    plt.xlabel('Mean Time (ms)')
    plt.ylabel('Standard Deviation (ms)')
    plt.tight_layout()
    plt.savefig('inference_times/layer_statistics.png')
    plt.close()

def plot_all_layer_mean_times(timing_stats):
    """Plot mean time for all layers"""
    # Sort layers by mean time
    sorted_layers = sorted(timing_stats.items(), key=lambda x: x[1]["mean_time"], reverse=True)
    layer_names = [name.split('.')[-1] for name, _ in sorted_layers]  # Use only the last part of the name
    mean_times = [stats["mean_time"] for _, stats in sorted_layers]
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(layer_names)), mean_times)
    plt.title('Mean Inference Time for All Layers')
    plt.xlabel('Layer')
    plt.ylabel('Mean Time (ms)')
    
    # Show only every nth label to avoid overcrowding
    n = max(1, len(layer_names) // 20)  # Show about 20 labels
    plt.xticks(range(0, len(layer_names), n), 
               [layer_names[i] for i in range(0, len(layer_names), n)],
               rotation=45, ha='right')
    
    # Add value labels for top 10 layers
    for i in range(min(10, len(bars))):
        bar = bars[i]
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms\n{layer_names[i]}',
                ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('inference_times/all_layers_mean_time.png')
    plt.close()

def main():
    # Load timing data
    json_path = Path('inference_times/layer_times_detailed.json')
    if not json_path.exists():
        print(f"Error: Could not find timing data at {json_path}")
        return
        
    data = load_timing_data(json_path)
    
    # Create visualizations
    plot_component_total_times(data["component_groups"])
    plot_top_layers_by_component(data["component_groups"])
    plot_layer_statistics(data["individual_layers"])
    plot_all_layer_mean_times(data["individual_layers"])
    
    print("Visualizations have been saved to the inference_times directory.")

if __name__ == '__main__':
    main() 