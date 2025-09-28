#!/usr/bin/env python3
"""
Temporal Consistency Analyzer for DVIS-DAQ
Analyzes how attention patterns change over time to determine motion vs. appearance reliance
"""

import torch
import numpy as np
import json
import cv2
from pathlib import Path
import sys
from tqdm import tqdm
import pickle
import gc
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mutual_info_score
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from memory_efficient_config import create_memory_efficient_config
from utils.model_loader import DVISModelLoader

class TemporalConsistencyAnalyzer:
    """
    Analyzes temporal consistency of attention patterns in DVIS-DAQ model
    to determine whether the model relies more on motion or appearance features.
    """
    
    def __init__(self, model_path: str, config_path: str, device: str = "cuda"):
        """
        Initialize the temporal consistency analyzer
        
        Args:
            model_path: Path to the trained DVIS-DAQ model
            config_path: Path to the model configuration
            device: Device to run analysis on
        """
        self.device = device
        self.model_path = model_path
        self.config_path = config_path
        
        # Initialize model
        self.model_loader = DVISModelLoader(config_path, model_path, device)
        self.model = self.model_loader.load_model()
        
        # Storage for attention weights
        self.attention_weights = []
        self.attention_hooks = []
        
        # Analysis results
        self.results = {}
        
        # Create output directories
        self.output_dir = Path("temporal_consistency_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Temporal Consistency Analyzer initialized on {device}")
    
    def register_attention_hooks(self):
        """Register hooks to extract attention weights from transformer layers"""
        print("Registering attention hooks...")
        
        # Storage for different types of attention weights
        self.self_attention_weights = []
        self.cross_attention_weights = []
        self.temporal_attention_weights = []
        
        def self_attention_hook(module, input, output):
            """Hook to extract self-attention weights (within-frame attention)"""
            if hasattr(output, 'attn_weights'):
                self.self_attention_weights.append(output.attn_weights.detach().cpu())
            elif isinstance(output, tuple) and len(output) > 1:
                if hasattr(output[1], 'shape'):
                    self.self_attention_weights.append(output[1].detach().cpu())
        
        def cross_attention_hook(module, input, output):
            """Hook to extract cross-attention weights (between-frame attention)"""
            if hasattr(output, 'attn_weights'):
                self.cross_attention_weights.append(output.attn_weights.detach().cpu())
            elif isinstance(output, tuple) and len(output) > 1:
                if hasattr(output[1], 'shape'):
                    self.cross_attention_weights.append(output[1].detach().cpu())
        
        def temporal_attention_hook(module, input, output):
            """Hook to extract temporal attention weights (temporal tracking)"""
            if hasattr(output, 'attn_weights'):
                self.temporal_attention_weights.append(output.attn_weights.detach().cpu())
            elif isinstance(output, tuple) and len(output) > 1:
                if hasattr(output[1], 'shape'):
                    self.temporal_attention_weights.append(output[1].detach().cpu())
        
        # Register hooks on specific attention layers
        for i, layer in enumerate(self.model.tracker.transformer_self_attention_layers):
            hook = layer.register_forward_hook(self_attention_hook)
            self.attention_hooks.append(hook)
            print(f"Registered self-attention hook on layer {i}")
        
        for i, layer in enumerate(self.model.tracker.transformer_cross_attention_layers):
            hook = layer.register_forward_hook(cross_attention_hook)
            self.attention_hooks.append(hook)
            print(f"Registered cross-attention hook on layer {i}")
        
        # Register hooks on the referring cross-attention layers (temporal tracking)
        for name, module in self.model.tracker.named_modules():
            if 'referring' in name.lower() and 'attention' in name.lower():
                hook = module.register_forward_hook(temporal_attention_hook)
                self.attention_hooks.append(hook)
                print(f"Registered temporal attention hook on: {name}")
        
        print(f"Registered {len(self.attention_hooks)} attention hooks")
        print(f"  - Self-attention layers: {len(self.model.tracker.transformer_self_attention_layers)}")
        print(f"  - Cross-attention layers: {len(self.model.tracker.transformer_cross_attention_layers)}")
        print(f"  - Temporal attention layers: Found referring cross-attention layers")
    
    def remove_attention_hooks(self):
        """Remove all registered attention hooks"""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks = []
        print("Removed all attention hooks")
    
    def extract_attention_weights(self, video_sequence: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        Extract attention weights for a video sequence
        
        Args:
            video_sequence: List of video frames as tensors
            
        Returns:
            Dictionary containing attention weights by type
        """
        print(f"Extracting attention weights for {len(video_sequence)} frames...")
        
        # Clear previous attention weights
        self.self_attention_weights = []
        self.cross_attention_weights = []
        self.temporal_attention_weights = []
        
        # Register hooks
        self.register_attention_hooks()
        
        # Prepare input for model
        batched_input = self.prepare_model_input(video_sequence)
        
        # Run inference to extract attention weights
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(batched_input)
        
        # Remove hooks
        self.remove_attention_hooks()
        
        # Process and organize attention weights
        processed_weights = self.process_attention_weights()
        
        print(f"Extracted attention weights for {len(video_sequence)} frames")
        return processed_weights
    
    def prepare_model_input(self, video_sequence: List[torch.Tensor]) -> List[Dict]:
        """Prepare video sequence for DVIS-DAQ model input"""
        # Convert frames to the format expected by DVIS-DAQ
        batched_input = []
        
        for frame in video_sequence:
            # Ensure frame is in correct format (C, H, W)
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)  # Add batch dimension
            
            # Normalize if needed
            if frame.max() > 1.0:
                frame = frame / 255.0
            
            # Create input dict
            input_dict = {
                "image": [frame],
                "height": frame.shape[-2],
                "width": frame.shape[-1]
            }
            batched_input.append(input_dict)
        
        return batched_input
    
    def process_attention_weights(self) -> Dict[str, List[torch.Tensor]]:
        """Process and organize extracted attention weights by type"""
        processed_weights = {
            'self_attention': [],
            'cross_attention': [],
            'temporal_attention': []
        }
        
        # Process self-attention weights (within-frame attention)
        for weight in self.self_attention_weights:
            if weight.dim() > 3:
                weight = weight.mean(dim=0)  # Average across heads
            if weight.dim() == 3:
                weight = weight.squeeze(0)
            processed_weights['self_attention'].append(weight)
        
        # Process cross-attention weights (between-frame attention)
        for weight in self.cross_attention_weights:
            if weight.dim() > 3:
                weight = weight.mean(dim=0)  # Average across heads
            if weight.dim() == 3:
                weight = weight.squeeze(0)
            processed_weights['cross_attention'].append(weight)
        
        # Process temporal attention weights (temporal tracking)
        for weight in self.temporal_attention_weights:
            if weight.dim() > 3:
                weight = weight.mean(dim=0)  # Average across heads
            if weight.dim() == 3:
                weight = weight.squeeze(0)
            processed_weights['temporal_attention'].append(weight)
        
        print(f"Processed attention weights:")
        print(f"  - Self-attention: {len(processed_weights['self_attention'])} frames")
        print(f"  - Cross-attention: {len(processed_weights['cross_attention'])} frames")
        print(f"  - Temporal attention: {len(processed_weights['temporal_attention'])} frames")
        
        return processed_weights
    
    def compute_temporal_correlations(self, attention_weights: Dict[str, List[torch.Tensor]]) -> Dict[str, Any]:
        """
        Compute temporal correlations for different types of attention weights
        
        Args:
            attention_weights: Dictionary containing attention weights by type
            
        Returns:
            Dictionary containing correlation statistics for each attention type
        """
        print("Computing temporal correlations for different attention types...")
        
        results = {}
        
        for attention_type, weights in attention_weights.items():
            if len(weights) < 2:
                results[attention_type] = {"error": f"Need at least 2 frames for {attention_type} correlation"}
                continue
            
            print(f"Computing correlations for {attention_type}...")
            correlations = []
            correlation_details = []
            
            # Compute correlation between consecutive frames
            for t in range(len(weights) - 1):
                # Flatten attention weights to 1D vectors
                attn_t = weights[t].flatten().numpy()
                attn_t_plus_1 = weights[t + 1].flatten().numpy()
                
                # Remove any NaN or infinite values
                valid_mask = np.isfinite(attn_t) & np.isfinite(attn_t_plus_1)
                attn_t_clean = attn_t[valid_mask]
                attn_t_plus_1_clean = attn_t_plus_1[valid_mask]
                
                if len(attn_t_clean) < 10:  # Need minimum number of points
                    correlations.append(np.nan)
                    correlation_details.append({
                        'frame_pair': (t, t + 1),
                        'correlation': np.nan,
                        'valid_points': len(attn_t_clean)
                    })
                    continue
                
                # Compute Pearson correlation
                try:
                    corr, p_value = pearsonr(attn_t_clean, attn_t_plus_1_clean)
                    correlations.append(corr)
                    correlation_details.append({
                        'frame_pair': (t, t + 1),
                        'correlation': corr,
                        'p_value': p_value,
                        'valid_points': len(attn_t_clean)
                    })
                except Exception as e:
                    print(f"Error computing correlation for {attention_type} frames {t}-{t+1}: {e}")
                    correlations.append(np.nan)
                    correlation_details.append({
                        'frame_pair': (t, t + 1),
                        'correlation': np.nan,
                        'error': str(e)
                    })
            
            # Convert to numpy array and remove NaN values for statistics
            correlations = np.array(correlations)
            valid_correlations = correlations[np.isfinite(correlations)]
            
            # Compute statistics
            stats = {
                'mean_correlation': float(np.mean(valid_correlations)) if len(valid_correlations) > 0 else np.nan,
                'std_correlation': float(np.std(valid_correlations)) if len(valid_correlations) > 0 else np.nan,
                'min_correlation': float(np.min(valid_correlations)) if len(valid_correlations) > 0 else np.nan,
                'max_correlation': float(np.max(valid_correlations)) if len(valid_correlations) > 0 else np.nan,
                'median_correlation': float(np.median(valid_correlations)) if len(valid_correlations) > 0 else np.nan,
                'num_valid_correlations': len(valid_correlations),
                'total_frame_pairs': len(correlations),
                'correlation_details': correlation_details,
                'all_correlations': correlations.tolist()
            }
            
            # Interpret results
            stats['interpretation'] = self.interpret_correlation_results(stats)
            
            results[attention_type] = stats
            
            print(f"  {attention_type}: Mean correlation = {stats['mean_correlation']:.3f}")
            print(f"  {attention_type}: {stats['interpretation']}")
        
        return results
    
    def interpret_correlation_results(self, stats: Dict[str, Any]) -> str:
        """Interpret temporal correlation results"""
        mean_corr = stats['mean_correlation']
        
        if np.isnan(mean_corr):
            return "Unable to compute correlations"
        
        if mean_corr > 0.7:
            return "High appearance reliance - attention patterns are very consistent over time"
        elif mean_corr > 0.5:
            return "Moderate appearance reliance - attention patterns are somewhat consistent"
        elif mean_corr > 0.3:
            return "Mixed reliance - attention patterns show moderate temporal variation"
        elif mean_corr > 0.1:
            return "Moderate motion reliance - attention patterns change significantly over time"
        else:
            return "High motion reliance - attention patterns change dramatically over time"
    
    def analyze_temporal_trends(self, correlations: List[float]) -> Dict[str, Any]:
        """Analyze trends in temporal correlations over time"""
        if len(correlations) < 3:
            return {"error": "Need at least 3 correlations for trend analysis"}
        
        # Remove NaN values
        valid_correlations = [c for c in correlations if not np.isnan(c)]
        valid_indices = [i for i, c in enumerate(correlations) if not np.isnan(c)]
        
        if len(valid_correlations) < 3:
            return {"error": "Insufficient valid correlations for trend analysis"}
        
        # Compute trend (linear regression)
        x = np.array(valid_indices)
        y = np.array(valid_correlations)
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Compute R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Analyze trend direction
        if slope > 0.01:
            trend_direction = "increasing"
        elif slope < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'trend_direction': trend_direction,
            'trend_strength': 'strong' if abs(slope) > 0.05 else 'weak',
            'valid_indices': valid_indices,
            'valid_correlations': valid_correlations
        }
    
    def visualize_temporal_consistency(self, correlations: List[float], video_id: str):
        """Create visualizations of temporal consistency analysis"""
        print(f"Creating visualizations for video {video_id}...")
        
        # Remove NaN values for plotting
        valid_correlations = [c for i, c in enumerate(correlations) if not np.isnan(c)]
        valid_indices = [i for i, c in enumerate(correlations) if not np.isnan(c)]
        
        if len(valid_correlations) == 0:
            print("No valid correlations to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Temporal Consistency Analysis - Video {video_id}', fontsize=16)
        
        # Plot 1: Correlation over time
        axes[0, 0].plot(valid_indices, valid_correlations, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Frame Pair Index')
        axes[0, 0].set_ylabel('Temporal Correlation')
        axes[0, 0].set_title('Temporal Correlation Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0.7, color='g', linestyle='--', alpha=0.7, label='High Appearance Threshold')
        axes[0, 0].axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='High Motion Threshold')
        axes[0, 0].legend()
        
        # Plot 2: Correlation distribution
        axes[0, 1].hist(valid_correlations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Temporal Correlation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Temporal Correlations')
        axes[0, 1].axvline(x=np.mean(valid_correlations), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(valid_correlations):.3f}')
        axes[0, 1].legend()
        
        # Plot 3: Correlation heatmap (if we have attention weights)
        if hasattr(self, 'attention_weights') and len(self.attention_weights) > 0:
            # Create correlation matrix between all frames
            num_frames = len(self.attention_weights)
            corr_matrix = np.zeros((num_frames, num_frames))
            
            for i in range(num_frames):
                for j in range(num_frames):
                    if i != j:
                        attn_i = self.attention_weights[i].flatten().numpy()
                        attn_j = self.attention_weights[j].flatten().numpy()
                        
                        valid_mask = np.isfinite(attn_i) & np.isfinite(attn_j)
                        if np.sum(valid_mask) > 10:
                            try:
                                corr, _ = pearsonr(attn_i[valid_mask], attn_j[valid_mask])
                                corr_matrix[i, j] = corr
                            except:
                                corr_matrix[i, j] = np.nan
                        else:
                            corr_matrix[i, j] = np.nan
            
            im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, 0].set_title('Attention Pattern Correlation Matrix')
            axes[1, 0].set_xlabel('Frame Index')
            axes[1, 0].set_ylabel('Frame Index')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Summary statistics
        axes[1, 1].axis('off')
        stats_text = f"""
        Temporal Consistency Summary:
        
        Mean Correlation: {np.mean(valid_correlations):.3f}
        Std Correlation: {np.std(valid_correlations):.3f}
        Min Correlation: {np.min(valid_correlations):.3f}
        Max Correlation: {np.max(valid_correlations):.3f}
        Median Correlation: {np.median(valid_correlations):.3f}
        
        Total Frame Pairs: {len(correlations)}
        Valid Correlations: {len(valid_correlations)}
        
        Interpretation:
        {self.interpret_correlation_results({'mean_correlation': np.mean(valid_correlations)})}
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / f"temporal_consistency_{video_id}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal consistency visualization saved to {viz_file}")
    
    def visualize_combined_attention_types(self, correlation_stats: Dict[str, Any], video_id: str):
        """Create combined visualization comparing all attention types"""
        print(f"Creating combined attention type visualization for video {video_id}...")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Attention Type Comparison - Video {video_id}', fontsize=16)
        
        # Plot 1: Mean correlations by attention type
        attention_types = []
        mean_correlations = []
        colors = ['blue', 'red', 'green']
        
        for i, (attention_type, stats) in enumerate(correlation_stats.items()):
            if 'error' not in stats:
                attention_types.append(attention_type.replace('_', ' ').title())
                mean_correlations.append(stats['mean_correlation'])
            else:
                attention_types.append(attention_type.replace('_', ' ').title())
                mean_correlations.append(0)  # Placeholder for error
        
        bars = axes[0, 0].bar(attention_types, mean_correlations, color=colors[:len(attention_types)], alpha=0.7)
        axes[0, 0].set_ylabel('Mean Temporal Correlation')
        axes[0, 0].set_title('Mean Correlation by Attention Type')
        axes[0, 0].axhline(y=0.7, color='g', linestyle='--', alpha=0.7, label='High Appearance Threshold')
        axes[0, 0].axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='High Motion Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, corr in zip(bars, mean_correlations):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{corr:.3f}', ha='center', va='bottom')
        
        # Plot 2: Correlation trends over time for each attention type
        for i, (attention_type, stats) in enumerate(correlation_stats.items()):
            if 'error' not in stats:
                correlations = stats.get('all_correlations', [])
                valid_correlations = [c for c in correlations if not np.isnan(c)]
                valid_indices = [i for i, c in enumerate(correlations) if not np.isnan(c)]
                
                if valid_correlations:
                    axes[0, 1].plot(valid_indices, valid_correlations, 
                                   color=colors[i], marker='o', linewidth=2, 
                                   label=attention_type.replace('_', ' ').title())
        
        axes[0, 1].set_xlabel('Frame Pair Index')
        axes[0, 1].set_ylabel('Temporal Correlation')
        axes[0, 1].set_title('Correlation Trends by Attention Type')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Interpretation summary
        axes[1, 0].axis('off')
        interpretation_text = "Attention Type Interpretations:\n\n"
        
        for attention_type, stats in correlation_stats.items():
            if 'error' not in stats:
                interpretation = stats.get('interpretation', 'No interpretation available')
                interpretation_text += f"{attention_type.replace('_', ' ').title()}:\n"
                interpretation_text += f"  {interpretation}\n\n"
            else:
                interpretation_text += f"{attention_type.replace('_', ' ').title()}:\n"
                interpretation_text += f"  Error: {stats['error']}\n\n"
        
        axes[1, 0].text(0.1, 0.9, interpretation_text, transform=axes[1, 0].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Plot 4: Statistical comparison
        axes[1, 1].axis('off')
        stats_text = "Statistical Summary:\n\n"
        
        for attention_type, stats in correlation_stats.items():
            if 'error' not in stats:
                stats_text += f"{attention_type.replace('_', ' ').title()}:\n"
                stats_text += f"  Mean: {stats['mean_correlation']:.3f}\n"
                stats_text += f"  Std: {stats['std_correlation']:.3f}\n"
                stats_text += f"  Min: {stats['min_correlation']:.3f}\n"
                stats_text += f"  Max: {stats['max_correlation']:.3f}\n"
                stats_text += f"  Valid pairs: {stats['num_valid_correlations']}\n\n"
            else:
                stats_text += f"{attention_type.replace('_', ' ').title()}:\n"
                stats_text += f"  Error: {stats['error']}\n\n"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / f"attention_types_comparison_{video_id}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined attention type visualization saved to {viz_file}")
    
    def visualize_attention_patterns(self, attention_weights: Dict[str, List[torch.Tensor]], video_id: str, window_id: str = ""):
        """Visualize the actual attention patterns for each attention type"""
        print(f"Creating attention pattern visualizations for {video_id} {window_id}...")
        
        for attention_type, weights in attention_weights.items():
            if not weights:
                continue
                
            print(f"Visualizing {attention_type} patterns...")
            
            # Create figure for this attention type
            num_frames = len(weights)
            fig, axes = plt.subplots(2, min(5, num_frames), figsize=(20, 8))
            fig.suptitle(f'{attention_type.replace("_", " ").title()} Patterns - {video_id} {window_id}', fontsize=16)
            
            # Select frames to visualize (first, middle, last, and a couple in between)
            if num_frames <= 5:
                frame_indices = list(range(num_frames))
            else:
                frame_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
            
            for i, frame_idx in enumerate(frame_indices):
                if i >= axes.shape[1]:
                    break
                    
                attention_matrix = weights[frame_idx].numpy()
                
                # Plot 1: Raw attention matrix
                im1 = axes[0, i].imshow(attention_matrix, cmap='viridis', aspect='auto')
                axes[0, i].set_title(f'Frame {frame_idx}\nRaw Attention')
                axes[0, i].set_xlabel('Key Position')
                axes[0, i].set_ylabel('Query Position')
                plt.colorbar(im1, ax=axes[0, i])
                
                # Plot 2: Attention distribution
                flattened_attention = attention_matrix.flatten()
                axes[1, i].hist(flattened_attention, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, i].set_xlabel('Attention Weight')
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].set_title(f'Frame {frame_idx}\nDistribution')
                axes[1, i].axvline(x=np.mean(flattened_attention), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(flattened_attention):.3f}')
                axes[1, i].legend()
            
            # Hide unused subplots
            for i in range(len(frame_indices), axes.shape[1]):
                axes[0, i].axis('off')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.output_dir / f"attention_patterns_{attention_type}_{video_id}_{window_id}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Attention patterns for {attention_type} saved to {viz_file}")
    
    def visualize_attention_evolution(self, attention_weights: Dict[str, List[torch.Tensor]], video_id: str, window_id: str = ""):
        """Visualize how attention patterns evolve over time"""
        print(f"Creating attention evolution visualization for {video_id} {window_id}...")
        
        for attention_type, weights in attention_weights.items():
            if not weights or len(weights) < 2:
                continue
                
            print(f"Visualizing {attention_type} evolution...")
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{attention_type.replace("_", " ").title()} Evolution - {video_id} {window_id}', fontsize=16)
            
            # Convert weights to numpy arrays
            attention_arrays = [w.numpy() for w in weights]
            
            # Plot 1: Mean attention over time
            mean_attention = [np.mean(arr) for arr in attention_arrays]
            std_attention = [np.std(arr) for arr in attention_arrays]
            
            axes[0, 0].plot(range(len(mean_attention)), mean_attention, 'b-o', linewidth=2, markersize=6)
            axes[0, 0].fill_between(range(len(mean_attention)), 
                                   [m - s for m, s in zip(mean_attention, std_attention)],
                                   [m + s for m, s in zip(mean_attention, std_attention)],
                                   alpha=0.3, color='blue')
            axes[0, 0].set_xlabel('Frame')
            axes[0, 0].set_ylabel('Mean Attention Weight')
            axes[0, 0].set_title('Mean Attention Over Time')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Attention variance over time
            attention_variance = [np.var(arr) for arr in attention_arrays]
            axes[0, 1].plot(range(len(attention_variance)), attention_variance, 'r-o', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('Attention Variance')
            axes[0, 1].set_title('Attention Variance Over Time')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Attention matrix heatmap over time
            # Create a 2D representation where x=frame, y=flattened_attention_position
            max_size = max(arr.size for arr in attention_arrays)
            evolution_matrix = np.zeros((len(attention_arrays), max_size))
            
            for i, arr in enumerate(attention_arrays):
                flattened = arr.flatten()
                evolution_matrix[i, :len(flattened)] = flattened
            
            im = axes[1, 0].imshow(evolution_matrix, cmap='viridis', aspect='auto')
            axes[1, 0].set_xlabel('Attention Position (Flattened)')
            axes[1, 0].set_ylabel('Frame')
            axes[1, 0].set_title('Attention Evolution Heatmap')
            plt.colorbar(im, ax=axes[1, 0])
            
            # Plot 4: Attention sparsity over time
            # Calculate sparsity (percentage of near-zero values)
            sparsity = []
            for arr in attention_arrays:
                flattened = arr.flatten()
                # Count values close to zero (less than 1% of max value)
                threshold = 0.01 * np.max(flattened)
                sparse_count = np.sum(np.abs(flattened) < threshold)
                sparsity.append(sparse_count / len(flattened) * 100)
            
            axes[1, 1].plot(range(len(sparsity)), sparsity, 'g-o', linewidth=2, markersize=6)
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('Sparsity (%)')
            axes[1, 1].set_title('Attention Sparsity Over Time')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.output_dir / f"attention_evolution_{attention_type}_{video_id}_{window_id}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Attention evolution for {attention_type} saved to {viz_file}")
    
    def visualize_attention_comparison(self, attention_weights: Dict[str, List[torch.Tensor]], video_id: str, window_id: str = ""):
        """Compare attention patterns between different attention types"""
        print(f"Creating attention comparison visualization for {video_id} {window_id}...")
        
        # Find the attention type with the most frames
        max_frames = max(len(weights) for weights in attention_weights.values() if weights)
        if max_frames == 0:
            return
        
        # Select a middle frame for comparison
        frame_idx = max_frames // 2
        
        # Create figure
        num_types = len([w for w in attention_weights.values() if w])
        fig, axes = plt.subplots(2, num_types, figsize=(5*num_types, 10))
        fig.suptitle(f'Attention Type Comparison - Frame {frame_idx} - {video_id} {window_id}', fontsize=16)
        
        col_idx = 0
        for attention_type, weights in attention_weights.items():
            if not weights or frame_idx >= len(weights):
                continue
                
            attention_matrix = weights[frame_idx].numpy()
            
            # Plot 1: Attention matrix
            im = axes[0, col_idx].imshow(attention_matrix, cmap='viridis', aspect='auto')
            axes[0, col_idx].set_title(f'{attention_type.replace("_", " ").title()}\nAttention Matrix')
            axes[0, col_idx].set_xlabel('Key Position')
            axes[0, col_idx].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[0, col_idx])
            
            # Plot 2: Attention distribution
            flattened_attention = attention_matrix.flatten()
            axes[1, col_idx].hist(flattened_attention, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, col_idx].set_xlabel('Attention Weight')
            axes[1, col_idx].set_ylabel('Frequency')
            axes[1, col_idx].set_title(f'{attention_type.replace("_", " ").title()}\nDistribution')
            axes[1, col_idx].axvline(x=np.mean(flattened_attention), color='red', linestyle='--', 
                                    label=f'Mean: {np.mean(flattened_attention):.3f}')
            axes[1, col_idx].legend()
            
            col_idx += 1
        
        # Hide unused subplots
        for i in range(col_idx, axes.shape[1]):
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / f"attention_comparison_{video_id}_{window_id}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention comparison saved to {viz_file}")

    def analyze_video(self, video_path: str, video_id: str) -> Dict[str, Any]:
        """
        Analyze temporal consistency for a single video
        
        Args:
            video_path: Path to video file
            video_id: Identifier for the video
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"Analyzing temporal consistency for video {video_id}...")
        
        # Load video frames
        frames = self.load_video_frames(video_path)
        if not frames:
            return {"error": f"Could not load video: {video_path}"}
        
        print(f"Loaded {len(frames)} frames from video")
        
        # Extract attention weights
        attention_weights = self.extract_attention_weights(frames)
        if not attention_weights:
            return {"error": "Could not extract attention weights"}
        
        # Compute temporal correlations for each attention type
        correlation_stats = self.compute_temporal_correlations(attention_weights)
        
        # Analyze temporal trends for each attention type
        trend_analysis = {}
        for attention_type, stats in correlation_stats.items():
            if 'error' not in stats:
                trend_analysis[attention_type] = self.analyze_temporal_trends(stats.get('all_correlations', []))
            else:
                trend_analysis[attention_type] = {"error": stats['error']}
        
        # Create visualizations for each attention type
        for attention_type, stats in correlation_stats.items():
            if 'error' not in stats:
                self.visualize_temporal_consistency(stats.get('all_correlations', []), f"{video_id}_{attention_type}")
        
        # Create combined visualization
        self.visualize_combined_attention_types(correlation_stats, video_id)
        
        # Compile results
        results = {
            'video_id': video_id,
            'num_frames': len(frames),
            'correlation_analysis': correlation_stats,
            'trend_analysis': trend_analysis,
            'attention_weights_shape': {
                attention_type: [w.shape for w in weights] if weights else []
                for attention_type, weights in attention_weights.items()
            }
        }
        
        # Save results
        results_file = self.output_dir / f"temporal_consistency_{video_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Analysis results saved to {results_file}")
        
        return results
    
    def load_video_frames(self, video_path: str) -> List[torch.Tensor]:
        """Load video frames from file"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor and normalize
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
                
                frames.append(frame_tensor)
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return []
    
    def analyze_multiple_videos(self, video_paths: List[str], video_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze temporal consistency for multiple videos
        
        Args:
            video_paths: List of video file paths
            video_ids: List of video identifiers
            
        Returns:
            Dictionary containing analysis results for all videos
        """
        print(f"Analyzing temporal consistency for {len(video_paths)} videos...")
        
        all_results = {}
        
        for video_path, video_id in tqdm(zip(video_paths, video_ids), total=len(video_paths)):
            try:
                result = self.analyze_video(video_path, video_id)
                all_results[video_id] = result
            except Exception as e:
                print(f"Error analyzing video {video_id}: {e}")
                all_results[video_id] = {"error": str(e)}
        
        # Save combined results
        combined_file = self.output_dir / "temporal_consistency_all_videos.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Create summary visualization
        self.create_summary_visualization(all_results)
        
        print(f"Combined analysis results saved to {combined_file}")
        return all_results
    
    def create_summary_visualization(self, all_results: Dict[str, Any]):
        """Create summary visualization comparing all videos"""
        print("Creating summary visualization...")
        
        # Extract correlation statistics
        video_ids = []
        mean_correlations = []
        interpretations = []
        
        for video_id, result in all_results.items():
            if 'error' not in result:
                video_ids.append(video_id)
                mean_correlations.append(result['correlation_analysis']['mean_correlation'])
                interpretations.append(result['correlation_analysis']['interpretation'])
        
        if not video_ids:
            print("No valid results to visualize")
            return
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Temporal Consistency Analysis - All Videos', fontsize=16)
        
        # Plot 1: Mean correlations by video
        bars = ax1.bar(range(len(video_ids)), mean_correlations, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Video ID')
        ax1.set_ylabel('Mean Temporal Correlation')
        ax1.set_title('Mean Temporal Correlation by Video')
        ax1.set_xticks(range(len(video_ids)))
        ax1.set_xticklabels(video_ids, rotation=45, ha='right')
        ax1.axhline(y=0.7, color='g', linestyle='--', alpha=0.7, label='High Appearance Threshold')
        ax1.axhline(y=0.3, color='r', linestyle='--', alpha=0.7, label='High Motion Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, corr in zip(bars, mean_correlations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom')
        
        # Plot 2: Interpretation distribution
        interpretation_counts = {}
        for interpretation in interpretations:
            # Extract key part of interpretation
            if 'appearance' in interpretation.lower():
                key = 'Appearance Reliance'
            elif 'motion' in interpretation.lower():
                key = 'Motion Reliance'
            else:
                key = 'Mixed Reliance'
            
            interpretation_counts[key] = interpretation_counts.get(key, 0) + 1
        
        if interpretation_counts:
            labels = list(interpretation_counts.keys())
            sizes = list(interpretation_counts.values())
            colors = ['lightgreen', 'lightcoral', 'lightblue']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Distribution of Reliance Types')
        
        plt.tight_layout()
        
        # Save summary visualization
        summary_file = self.output_dir / "temporal_consistency_summary.png"
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary visualization saved to {summary_file}")


def main():
    """Main function to run temporal consistency analysis"""
    
    # Configuration for model3_unmasked
    model_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth"
    config_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/config.yaml"
    
    # Initialize analyzer
    analyzer = TemporalConsistencyAnalyzer(model_path, config_path)
    
    # Load first video from val.json
    val_json_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/val.json"
    
    # Parse val.json to get first video
    import json
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    
    # Get first video
    first_video = val_data['videos'][0]
    video_id = f"video_{first_video['id']}"
    
    print(f"Using video ID: {video_id}")
    print(f"Video has {len(first_video['file_names'])} frames")
    
    # For now, we'll need to construct the video path or load frames directly
    # Let's assume the frames are in a data directory
    # You may need to adjust this path based on your actual data location
    base_data_path = "/path/to/your/data/directory"  # Update this path
    
    # Load frames from the first video
    frames = []
    for frame_name in first_video['file_names'][:31]:  # Use first 31 frames (as per config)
        frame_path = os.path.join(base_data_path, frame_name)
        if os.path.exists(frame_path):
            # Load frame using OpenCV
            import cv2
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
                frames.append(frame_tensor)
    
    if frames:
        print(f"Loaded {len(frames)} frames")
        
        # Extract attention weights
        attention_weights = analyzer.extract_attention_weights(frames)
        
        if attention_weights:
            # Compute temporal correlations
            correlation_stats = analyzer.compute_temporal_correlations(attention_weights)
            
            # Analyze temporal trends
            trend_analysis = {}
            for attention_type, stats in correlation_stats.items():
                if 'error' not in stats:
                    trend_analysis[attention_type] = analyzer.analyze_temporal_trends(stats.get('all_correlations', []))
                else:
                    trend_analysis[attention_type] = {"error": stats['error']}
            
            # Create visualizations
            for attention_type, stats in correlation_stats.items():
                if 'error' not in stats:
                    analyzer.visualize_temporal_consistency(stats.get('all_correlations', []), f"{video_id}_{attention_type}")
            
            # Create combined visualization
            analyzer.visualize_combined_attention_types(correlation_stats, video_id)
            
            # Save results
            results = {
                'video_id': video_id,
                'num_frames': len(frames),
                'correlation_analysis': correlation_stats,
                'trend_analysis': trend_analysis,
                'attention_weights_shape': {
                    attention_type: [w.shape for w in weights] if weights else []
                    for attention_type, weights in attention_weights.items()
                }
            }
            
            results_file = analyzer.output_dir / f"temporal_consistency_{video_id}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Analysis completed! Results saved to {results_file}")
            print(f"Correlation analysis summary:")
            for attention_type, stats in correlation_stats.items():
                if 'error' not in stats:
                    print(f"  {attention_type}: Mean correlation = {stats['mean_correlation']:.3f}")
                    print(f"  {attention_type}: {stats['interpretation']}")
        else:
            print("Failed to extract attention weights")
    else:
        print("No frames loaded. Please check the data path.")
        print("You may need to update the base_data_path variable in the script.")


if __name__ == "__main__":
    main()
