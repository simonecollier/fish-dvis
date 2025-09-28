#!/usr/bin/env python3
"""
Helper script to run temporal consistency analysis with model3_unmasked
"""

import os
import json
import torch
import cv2
from pathlib import Path
from temporal_consistency_analyzer import TemporalConsistencyAnalyzer
import matplotlib.pyplot as plt
import numpy as np

def aggregate_window_results(window_results):
    """Aggregate correlation results across all windows"""
    attention_types = ['self_attention', 'cross_attention', 'temporal_attention']
    aggregated = {}
    
    for attention_type in attention_types:
        correlations = []
        
        for window_result in window_results:
            if attention_type in window_result['correlation_analysis']:
                stats = window_result['correlation_analysis'][attention_type]
                if 'error' not in stats and 'mean_correlation' in stats:
                    correlations.append(stats['mean_correlation'])
        
        if correlations:
            # Compute aggregated statistics
            mean_corr = np.mean(correlations)
            std_corr = np.std(correlations)
            min_corr = np.min(correlations)
            max_corr = np.max(correlations)
            
            # Interpret aggregated results
            if mean_corr > 0.7:
                interpretation = "High appearance reliance - attention patterns are very consistent across windows"
            elif mean_corr > 0.5:
                interpretation = "Moderate appearance reliance - attention patterns are somewhat consistent across windows"
            elif mean_corr > 0.3:
                interpretation = "Mixed reliance - attention patterns show moderate variation across windows"
            elif mean_corr > 0.1:
                interpretation = "Moderate motion reliance - attention patterns change significantly across windows"
            else:
                interpretation = "High motion reliance - attention patterns change dramatically across windows"
            
            aggregated[attention_type] = {
                'mean_correlation': float(mean_corr),
                'std_correlation': float(std_corr),
                'min_correlation': float(min_corr),
                'max_correlation': float(max_corr),
                'num_windows': len(correlations),
                'all_window_correlations': correlations,
                'interpretation': interpretation
            }
        else:
            aggregated[attention_type] = {"error": "No valid correlations found across windows"}
    
    return aggregated

def create_overall_video_visualization(aggregated_results, video_id, output_dir):
    """Create overall visualization showing results across all windows"""
    print(f"Creating overall video visualization for {video_id}...")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Overall Video Analysis - {video_id}', fontsize=16)
    
    # Plot 1: Mean correlations by attention type across windows
    attention_types = []
    mean_correlations = []
    std_correlations = []
    colors = ['blue', 'red', 'green']
    
    for i, (attention_type, stats) in enumerate(aggregated_results.items()):
        if 'error' not in stats:
            attention_types.append(attention_type.replace('_', ' ').title())
            mean_correlations.append(stats['mean_correlation'])
            std_correlations.append(stats['std_correlation'])
        else:
            attention_types.append(attention_type.replace('_', ' ').title())
            mean_correlations.append(0)
            std_correlations.append(0)
    
    bars = axes[0, 0].bar(attention_types, mean_correlations, 
                         yerr=std_correlations, color=colors[:len(attention_types)], 
                         alpha=0.7, capsize=5)
    axes[0, 0].set_ylabel('Mean Temporal Correlation')
    axes[0, 0].set_title('Mean Correlation by Attention Type (Across Windows)')
    axes[0, 0].axhline(y=0.7, color='g', linestyle='--', alpha=0.7, label='High Appearance Threshold')
    axes[0, 0].axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='High Motion Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, corr in zip(bars, mean_correlations):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{corr:.3f}', ha='center', va='bottom')
    
    # Plot 2: Correlation distribution across windows
    for i, (attention_type, stats) in enumerate(aggregated_results.items()):
        if 'error' not in stats and 'all_window_correlations' in stats:
            correlations = stats['all_window_correlations']
            axes[0, 1].hist(correlations, bins=10, alpha=0.6, 
                           color=colors[i], label=attention_type.replace('_', ' ').title())
    
    axes[0, 1].set_xlabel('Temporal Correlation')
    axes[0, 1].set_ylabel('Number of Windows')
    axes[0, 1].set_title('Distribution of Correlations Across Windows')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Interpretation summary
    axes[1, 0].axis('off')
    interpretation_text = "Overall Video Interpretations:\n\n"
    
    for attention_type, stats in aggregated_results.items():
        if 'error' not in stats:
            interpretation = stats.get('interpretation', 'No interpretation available')
            interpretation_text += f"{attention_type.replace('_', ' ').title()}:\n"
            interpretation_text += f"  {interpretation}\n"
            interpretation_text += f"  Mean: {stats['mean_correlation']:.3f} ± {stats['std_correlation']:.3f}\n"
            interpretation_text += f"  Range: [{stats['min_correlation']:.3f}, {stats['max_correlation']:.3f}]\n"
            interpretation_text += f"  Windows: {stats['num_windows']}\n\n"
        else:
            interpretation_text += f"{attention_type.replace('_', ' ').title()}:\n"
            interpretation_text += f"  Error: {stats['error']}\n\n"
    
    axes[1, 0].text(0.1, 0.9, interpretation_text, transform=axes[1, 0].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Plot 4: Statistical comparison
    axes[1, 1].axis('off')
    stats_text = "Statistical Summary:\n\n"
    
    for attention_type, stats in aggregated_results.items():
        if 'error' not in stats:
            stats_text += f"{attention_type.replace('_', ' ').title()}:\n"
            stats_text += f"  Mean: {stats['mean_correlation']:.3f}\n"
            stats_text += f"  Std: {stats['std_correlation']:.3f}\n"
            stats_text += f"  Min: {stats['min_correlation']:.3f}\n"
            stats_text += f"  Max: {stats['max_correlation']:.3f}\n"
            stats_text += f"  Windows: {stats['num_windows']}\n\n"
        else:
            stats_text += f"{attention_type.replace('_', ' ').title()}:\n"
            stats_text += f"  Error: {stats['error']}\n\n"
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = Path(output_dir) / f"overall_video_analysis_{video_id}.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overall video visualization saved to {viz_file}")

def find_data_directory():
    """Try to find the data directory containing the video frames"""
    possible_paths = [
        "/data",
        "/home/simone/data",
        "/home/simone/fish-dvis/data",
        "/home/simone/store/simone/dvis-model-outputs/data",
        "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/data"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found potential data directory: {path}")
            return path
    
    return None

def main():
    """Main function to run temporal consistency analysis"""
    
    # Configuration for model3_unmasked
    model_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth"
    config_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/config.yaml"
    val_json_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/val.json"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    
    if not os.path.exists(val_json_path):
        print(f"Error: Val JSON file not found: {val_json_path}")
        return
    
    print("Found all required files!")
    
    # Find data directory
    base_data_path = find_data_directory()
    if base_data_path is None:
        print("Could not find data directory automatically.")
        print("Please update the base_data_path variable in the script.")
        return
    
    # Initialize analyzer
    print("Initializing temporal consistency analyzer...")
    analyzer = TemporalConsistencyAnalyzer(model_path, config_path)
    
    # Load first video from val.json
    print("Loading video information from val.json...")
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    
    # Get first video
    first_video = val_data['videos'][0]
    video_id = f"video_{first_video['id']}"
    
    print(f"Using video ID: {video_id}")
    print(f"Video has {len(first_video['file_names'])} frames")
    
    # Load all frames from the first video
    print("Loading all video frames...")
    all_frames = []
    frame_count = 0
    
    for frame_name in first_video['file_names']:
        frame_path = os.path.join(base_data_path, frame_name)
        
        if os.path.exists(frame_path):
            # Load frame using OpenCV
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
                all_frames.append(frame_tensor)
                frame_count += 1
        else:
            print(f"Warning: Frame not found: {frame_path}")
    
    print(f"Successfully loaded {frame_count} frames from the entire video")
    
    if not all_frames:
        print("No frames loaded. Please check the data path.")
        return
    
    # Process video in windows of 31 frames with overlap
    window_size = 31
    overlap = 15  # 50% overlap between windows
    stride = window_size - overlap
    
    print(f"Processing video in windows of {window_size} frames with {overlap} frame overlap")
    
    window_results = []
    window_attention_weights = []
    
    for window_start in range(0, len(all_frames) - window_size + 1, stride):
        window_end = window_start + window_size
        window_frames = all_frames[window_start:window_end]
        
        print(f"Processing window {len(window_results) + 1}: frames {window_start}-{window_end-1}")
        
        # Extract attention weights for this window
        window_weights = analyzer.extract_attention_weights(window_frames)
        
        if window_weights:
            window_attention_weights.append(window_weights)
            
            # Compute temporal correlations for this window
            window_correlations = analyzer.compute_temporal_correlations(window_weights)
            window_results.append({
                'window_id': len(window_results) + 1,
                'frame_range': (window_start, window_end - 1),
                'correlation_analysis': window_correlations
            })
        else:
            print(f"Failed to extract attention weights for window {len(window_results) + 1}")
    
    print(f"Processed {len(window_results)} windows")
    
    if window_results:
        print("Starting temporal consistency analysis across all windows...")
        
        # Aggregate results across all windows
        aggregated_results = aggregate_window_results(window_results)
        
        # Create visualizations for each window
        print("Creating visualizations for each window...")
        for i, window_result in enumerate(window_results):
            window_id = window_result['window_id']
            frame_range = window_result['frame_range']
            
            # Create visualizations for this window
            for attention_type, stats in window_result['correlation_analysis'].items():
                if 'error' not in stats:
                    analyzer.visualize_temporal_consistency(
                        stats.get('all_correlations', []), 
                        f"{video_id}_window{window_id}_{attention_type}"
                    )
            
            # Create combined visualization for this window
            analyzer.visualize_combined_attention_types(
                window_result['correlation_analysis'], 
                f"{video_id}_window{window_id}"
            )
            
            # Create attention pattern visualizations for this window
            analyzer.visualize_attention_patterns(
                window_attention_weights[len(window_results)-1], 
                video_id, 
                f"window{window_id}"
            )
            
            # Create attention evolution visualization for this window
            analyzer.visualize_attention_evolution(
                window_attention_weights[len(window_results)-1], 
                video_id, 
                f"window{window_id}"
            )
            
            # Create attention comparison visualization for this window
            analyzer.visualize_attention_comparison(
                window_attention_weights[len(window_results)-1], 
                video_id, 
                f"window{window_id}"
            )
        
        # Create overall video visualization
        print("Creating overall video analysis visualization...")
        create_overall_video_visualization(aggregated_results, video_id, analyzer.output_dir)
        
        # Save comprehensive results
        comprehensive_results = {
            'video_id': video_id,
            'total_frames': len(all_frames),
            'num_windows': len(window_results),
            'window_size': window_size,
            'overlap': overlap,
            'window_results': window_results,
            'aggregated_results': aggregated_results
        }
        
        results_file = analyzer.output_dir / f"temporal_consistency_{video_id}_comprehensive.json"
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"\n=== ANALYSIS COMPLETED ===")
        print(f"Comprehensive results saved to: {results_file}")
        print(f"\nAggregated correlation analysis summary:")
        for attention_type, stats in aggregated_results.items():
            if 'error' not in stats:
                print(f"  {attention_type}:")
                print(f"    Mean correlation across windows: {stats['mean_correlation']:.3f}")
                print(f"    Std correlation across windows: {stats['std_correlation']:.3f}")
                print(f"    Interpretation: {stats['interpretation']}")
            else:
                print(f"  {attention_type}: Error - {stats['error']}")
        
        print(f"\nVisualizations saved to: {analyzer.output_dir}")
        
    else:
        print("No windows processed successfully.")
        print("Please check if the model and data are accessible.")

if __name__ == "__main__":
    main()
