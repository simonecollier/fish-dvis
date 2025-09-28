#!/usr/bin/env python3
"""
Spatial Gradient Analyzer for DVIS-DAQ Temporal Analysis
Uses GradCAM to visualize spatial attention and analyze appearance vs. motion reliance
GPU-optimized version for better performance and memory efficiency
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

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from memory_efficient_config import create_memory_efficient_config
from utils.model_loader import DVISModelLoader

class GradCAM:
    """GradCAM implementation for spatial gradient visualization with GPU support"""
    
    def __init__(self, model, target_layer, device="cuda"):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hooks = []
        self.register_hooks()
    
    def register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks on target layer
        target_module = self.get_target_layer()
        if target_module is not None:
            self.hooks.append(target_module.register_forward_hook(forward_hook))
            self.hooks.append(target_module.register_backward_hook(backward_hook))
        else:
            print("Warning: Could not find target layer for GradCAM")
    
    def get_target_layer(self):
        """Get the target layer for GradCAM"""
        # For DVIS-DAQ, we'll target the backbone's final layer
        if hasattr(self.model, 'backbone'):
            # Get the last layer of the backbone
            backbone_modules = list(self.model.backbone.modules())
            # Look for the last convolutional layer
            for module in reversed(backbone_modules):
                if isinstance(module, torch.nn.Conv2d):
                    return module
            # Fallback to the last module
            return backbone_modules[-1]
        else:
            # Fallback to first conv layer
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    return module
        return None
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate GradCAM for the input tensor with GPU optimization"""
        # Clear gradients and activations
        self.gradients = None
        self.activations = None
        
        # Move input to device
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        with torch.enable_grad():
            output = self.model(input_tensor)
            
            # If no target class specified, use the predicted class
            if target_class is None:
                if isinstance(output, dict) and 'online_out' in output:
                    pred_logits = output['online_out']['pred_logits']
                    if isinstance(pred_logits, list):
                        pred_logits = pred_logits[0]
                    target_class = pred_logits.argmax(dim=-1)
                else:
                    target_class = output.argmax(dim=-1)
            
            # Create loss for the target class
            if isinstance(output, dict) and 'online_out' in output:
                pred_logits = output['online_out']['pred_logits']
                if isinstance(pred_logits, list):
                    pred_logits = pred_logits[0]
                loss = pred_logits[0, target_class]
            else:
                loss = output[0, target_class]
            
            # Backward pass
            loss.backward()
        
        # Get gradients and activations
        if self.gradients is None or self.activations is None:
            print("Warning: Gradients or activations not captured")
            return None
        
        # Move to CPU for processing
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3])  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU
        cam = torch.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.numpy()
    
    def create_attention_map(self, input_tensor, target_class=None):
        """Create attention map from input tensor"""
        cam = self.generate_cam(input_tensor, target_class)
        if cam is None:
            return None
        
        # Resize to input size
        cam_resized = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        
        return cam_resized

class SpatialGradientAnalyzer:
    """Analyze spatial gradients using GradCAM with GPU optimization"""
    
    def __init__(self, config):
        self.config = config
        
        # Check GPU availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available, falling back to CPU")
            self.config.device = "cpu"
        else:
            self.config.device = "cuda"
            print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model
        print("Loading DVIS-DAQ model...")
        self.model_loader = DVISModelLoader(
            config_path=str(config.config_path),
            checkpoint_path=str(config.model_path),
            device=self.config.device
        )
        self.model = self.model_loader.load_model()
        
        # Move model to appropriate device
        self.model = self.model.to(self.config.device)
        
        # Initialize GradCAM
        self.gradcam = GradCAM(self.model, target_layer=None, device=self.config.device)
        
        # Load dataset
        with open(config.dataset_json_path, 'r') as f:
            self.dataset = json.load(f)
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.gradcam_dir = self.output_dir / "gradcam_visualizations"
        self.gradcam_dir.mkdir(exist_ok=True)
        
        self.analysis_dir = self.output_dir / "spatial_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
    
    def load_video_frames(self, video_path, max_frames=None):
        """Load video frames from directory"""
        video_dir = Path(video_path)
        if not video_dir.exists():
            print(f"Video directory not found: {video_dir}")
            return None
        
        # Find all frame files
        frame_files = sorted([f for f in video_dir.glob("*.jpg")])
        if not frame_files:
            print(f"No frame files found in {video_dir}")
            return None
        
        # Load all available frames first
        all_frames = []
        for frame_file in frame_files:
            try:
                frame = cv2.imread(str(frame_file))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 640))
                frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
                all_frames.append(frame)
            except Exception as e:
                print(f"Error loading frame {frame_file}: {e}")
                continue
        
        if not all_frames:
            print(f"No frames loaded from {video_dir}")
            return None
        
        all_frames = np.array(all_frames)
        
        # If max_frames is specified, extract middle frames
        if max_frames and len(all_frames) > max_frames:
            start_idx = (len(all_frames) - max_frames) // 2
            end_idx = start_idx + max_frames
            frames = all_frames[start_idx:end_idx]
            print(f"Extracted middle {max_frames} frames (indices {start_idx}-{end_idx-1}) from {len(all_frames)} total frames")
        else:
            frames = all_frames
            if max_frames:
                frames = frames[:max_frames]
        
        return frames
    
    def preprocess_frames(self, video_frames):
        """Preprocess frames for model input with GPU optimization"""
        print("Preprocessing frames for GPU...")
        
        frame_list = []
        for t in range(len(video_frames)):
            frame = video_frames[t]  # (H, W, C) normalized
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8
            frame_tensor = torch.as_tensor(np.ascontiguousarray(frame.transpose(2, 0, 1)))  # (C, H, W)
            frame_tensor = frame_tensor.to(self.config.device, dtype=torch.float32)
            frame_list.append(frame_tensor)
        
        return frame_list
    
    def extract_spatial_gradients(self, video_frames, video_id):
        """Extract spatial gradients using GradCAM with GPU optimization"""
        print(f"Extracting spatial gradients using GPU - Video {video_id}")
        print(f"Processing {len(video_frames)} frames")
        
        # Preprocess frames
        processed_frames = self.preprocess_frames(video_frames)
        
        # Extract GradCAM for each frame
        spatial_gradients = []
        
        for i, frame_tensor in enumerate(tqdm(processed_frames, desc="Processing frames")):
            try:
                # Add batch dimension
                input_tensor = frame_tensor.unsqueeze(0)  # (1, C, H, W)
                
                # Generate GradCAM
                attention_map = self.gradcam.create_attention_map(input_tensor)
                
                if attention_map is not None:
                    spatial_gradients.append(attention_map)
                else:
                    # Fallback: use zeros
                    spatial_gradients.append(np.zeros((frame_tensor.shape[1], frame_tensor.shape[2])))
                
                # Memory cleanup
                del input_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                spatial_gradients.append(np.zeros((processed_frames[0].shape[1], processed_frames[0].shape[2])))
        
        return np.array(spatial_gradients)
    
    def create_attention_map(self, original_frame, cam):
        """Create attention map by overlaying CAM on original frame"""
        # Normalize CAM to 0-1
        cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Apply colormap
        cam_colored = cv2.applyColorMap((cam_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay on original frame
        alpha = 0.6
        attention_map = (alpha * cam_colored + (1 - alpha) * (original_frame * 255)).astype(np.uint8)
        
        return attention_map
    
    def compute_spatial_attention_statistics(self, spatial_gradients):
        """Compute statistics about spatial attention patterns"""
        stats = {}
        
        # Per-frame attention statistics
        frame_attention_scores = []
        for frame_grad in spatial_gradients:
            # Compute attention score (mean gradient magnitude)
            attention_score = np.mean(frame_grad)
            frame_attention_scores.append(attention_score)
        
        frame_attention_scores = np.array(frame_attention_scores)
        
        stats['mean_attention'] = float(np.mean(frame_attention_scores))
        stats['std_attention'] = float(np.std(frame_attention_scores))
        stats['max_attention'] = float(np.max(frame_attention_scores))
        stats['min_attention'] = float(np.min(frame_attention_scores))
        stats['attention_variance'] = float(np.var(frame_attention_scores))
        
        # Spatial consistency (how focused the attention is)
        spatial_focus_scores = []
        for frame_grad in spatial_gradients:
            # Compute spatial focus (higher values = more focused attention)
            focus_score = np.std(frame_grad) / (np.mean(frame_grad) + 1e-8)
            spatial_focus_scores.append(focus_score)
        
        stats['mean_spatial_focus'] = float(np.mean(spatial_focus_scores))
        stats['spatial_focus_variance'] = float(np.var(spatial_focus_scores))
        
        return stats, frame_attention_scores
    
    def analyze_spatial_attention_patterns(self, spatial_gradients, video_id):
        """Analyze spatial attention patterns across frames"""
        print(f"Analyzing spatial attention patterns for video {video_id}...")
        
        # Compute attention statistics
        stats, frame_attention_scores = self.compute_spatial_attention_statistics(spatial_gradients)
        
        # Analyze temporal consistency of spatial attention
        temporal_consistency = self.compute_temporal_consistency(spatial_gradients)
        
        # Analyze spatial focus patterns
        spatial_focus_analysis = self.analyze_spatial_focus(spatial_gradients)
        
        return {
            'stats': stats,
            'frame_attention_scores': frame_attention_scores,
            'temporal_consistency': temporal_consistency,
            'spatial_focus_analysis': spatial_focus_analysis
        }
    
    def compute_temporal_consistency(self, spatial_gradients):
        """Compute temporal consistency of spatial attention"""
        # Compute correlation between consecutive frames
        correlations = []
        for i in range(len(spatial_gradients) - 1):
            corr, _ = pearsonr(spatial_gradients[i].flatten(), spatial_gradients[i + 1].flatten())
            correlations.append(corr)
        
        return {
            'mean_temporal_correlation': float(np.mean(correlations)),
            'std_temporal_correlation': float(np.std(correlations)),
            'temporal_correlations': correlations
        }
    
    def analyze_spatial_focus(self, spatial_gradients):
        """Analyze how focused the spatial attention is"""
        focus_scores = []
        focus_locations = []
        
        for frame_grad in spatial_gradients:
            # Find the region with highest attention
            max_idx = np.unravel_index(np.argmax(frame_grad), frame_grad.shape)
            focus_locations.append(max_idx)
            
            # Compute focus score (ratio of max to mean)
            focus_score = np.max(frame_grad) / (np.mean(frame_grad) + 1e-8)
            focus_scores.append(focus_score)
        
        return {
            'mean_focus_score': float(np.mean(focus_scores)),
            'focus_score_variance': float(np.var(focus_scores)),
            'focus_locations': focus_locations,
            'focus_scores': focus_scores
        }
    
    def compare_original_vs_shuffled(self, original_gradients, shuffled_gradients_list):
        """Compare spatial attention patterns between original and shuffled frames"""
        print("Comparing original vs shuffled spatial attention...")
        
        # Compute original attention statistics
        original_stats, original_scores = self.compute_spatial_attention_statistics(original_gradients)
        
        # Compute shuffled attention statistics
        shuffled_stats_list = []
        shuffled_scores_list = []
        
        for shuffle_idx, shuffled_gradients in enumerate(shuffled_gradients_list):
            stats, scores = self.compute_spatial_attention_statistics(shuffled_gradients)
            shuffled_stats_list.append(stats)
            shuffled_scores_list.append(scores)
        
        # Compute comparison metrics
        comparison = {
            'original_stats': original_stats,
            'shuffled_stats': shuffled_stats_list,
            'attention_consistency_ratio': self.compute_attention_consistency_ratio(original_scores, shuffled_scores_list),
            'spatial_focus_consistency': self.compute_spatial_focus_consistency(original_gradients, shuffled_gradients_list)
        }
        
        return comparison
    
    def compute_attention_consistency_ratio(self, original_scores, shuffled_scores_list):
        """Compute how consistent attention scores are between original and shuffled"""
        # Compute correlation between original and each shuffled version
        correlations = []
        for shuffled_scores in shuffled_scores_list:
            if len(original_scores) == len(shuffled_scores):
                corr, _ = pearsonr(original_scores, shuffled_scores)
                correlations.append(corr)
        
        return {
            'mean_correlation': float(np.mean(correlations)) if correlations else 0.0,
            'std_correlation': float(np.std(correlations)) if correlations else 0.0,
            'correlations': correlations
        }
    
    def compute_spatial_focus_consistency(self, original_gradients, shuffled_gradients_list):
        """Compute consistency of spatial focus patterns"""
        # Analyze spatial focus for original
        original_focus = self.analyze_spatial_focus(original_gradients)
        
        # Analyze spatial focus for shuffled versions
        shuffled_focus_list = []
        for shuffled_gradients in shuffled_gradients_list:
            focus = self.analyze_spatial_focus(shuffled_gradients)
            shuffled_focus_list.append(focus)
        
        # Compare focus scores
        original_focus_score = original_focus['mean_focus_score']
        shuffled_focus_scores = [f['mean_focus_score'] for f in shuffled_focus_list]
        
        focus_ratio = np.mean(shuffled_focus_scores) / (original_focus_score + 1e-8)
        
        return {
            'focus_consistency_ratio': float(focus_ratio),
            'original_focus_score': float(original_focus_score),
            'shuffled_focus_scores': [float(s) for s in shuffled_focus_scores]
        }
    
    def visualize_spatial_attention(self, video_frames, spatial_gradients, attention_maps, video_id):
        """Create visualizations of spatial attention patterns"""
        print(f"Creating spatial attention visualizations for video {video_id}...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Spatial Attention Analysis - Video {video_id}', fontsize=16)
        
        # Select frames to visualize (first, middle, last, and one with high attention)
        frame_indices = [0, len(video_frames)//4, len(video_frames)//2, len(video_frames)//4*3]
        
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx < len(video_frames):
                # Original frame
                axes[0, i].imshow(video_frames[frame_idx])
                axes[0, i].set_title(f'Frame {frame_idx}')
                axes[0, i].axis('off')
                
                # Attention map
                axes[1, i].imshow(attention_maps[frame_idx])
                axes[1, i].set_title(f'Attention {frame_idx}')
                axes[1, i].axis('off')
        
        # Save visualization
        viz_file = self.gradcam_dir / f"spatial_attention_{video_id}.png"
        plt.tight_layout()
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Spatial attention visualization saved to {viz_file}")
    
    def analyze_appearance_vs_motion_spatial(self, video_frames, video_id):
        """Analyze appearance vs. motion reliance using spatial gradients"""
        print(f"\n=== Spatial Gradient Analysis for Video {video_id} ===")
        
        # Ensure we have enough frames
        if len(video_frames) < self.config.window_size:
            print(f"Warning: Only {len(video_frames)} frames available, using all frames")
            frames_to_process = video_frames
        else:
            frames_to_process = video_frames[:self.config.window_size]
        
        # 1. Extract original spatial gradients
        print("1. Extracting original spatial gradients...")
        original_gradients = self.extract_spatial_gradients(frames_to_process, video_id)
        
        if original_gradients is None:
            print("Failed to extract original spatial gradients")
            return None
        
        # 2. Analyze original spatial attention patterns
        print("2. Analyzing original spatial attention patterns...")
        original_analysis = self.analyze_spatial_attention_patterns(original_gradients, video_id)
        
        # 3. Perform multiple shuffles
        print(f"3. Performing {self.config.num_shuffles} frame shuffles for spatial analysis...")
        shuffled_gradients_list = []
        shuffled_attention_maps_list = []
        
        for shuffle_idx in range(self.config.num_shuffles):
            print(f"   Shuffle {shuffle_idx + 1}/{self.config.num_shuffles}")
            
            # Create random permutation
            permutation = np.random.permutation(len(frames_to_process))
            shuffled_frames = frames_to_process[permutation]
            
            # Extract spatial gradients for shuffled frames
            shuffled_gradients = self.extract_spatial_gradients(
                shuffled_frames, f"{video_id}_spatial_shuffle_{shuffle_idx}"
            )
            
            if shuffled_gradients is not None:
                # Align gradients back to original frame order using inverse permutation
                inverse_permutation = np.argsort(permutation)
                shuffled_gradients_aligned = shuffled_gradients[inverse_permutation]
                shuffled_attention_maps_aligned = [shuffled_gradients[i] for i in inverse_permutation]
                
                shuffled_gradients_list.append(shuffled_gradients_aligned)
                shuffled_attention_maps_list.append(shuffled_attention_maps_aligned)
            else:
                print(f"   Failed to extract spatial gradients for shuffle {shuffle_idx}")
        
        # 4. Compare original vs shuffled
        print("4. Comparing original vs shuffled spatial attention...")
        comparison = self.compare_original_vs_shuffled(original_gradients, shuffled_gradients_list)
        
        # 5. Create visualizations
        print("5. Creating visualizations...")
        self.visualize_spatial_attention(frames_to_process, original_gradients, original_gradients, video_id) # Pass original_gradients to visualize_spatial_attention
        
        # 6. Compile results
        results = {
            'video_id': video_id,
            'num_frames': len(frames_to_process),
            'original_gradients': original_gradients,
            'original_attention_maps': original_gradients, # This was a bug, should be original_gradients
            'original_analysis': original_analysis,
            'shuffled_gradients_list': shuffled_gradients_list,
            'shuffled_attention_maps_list': shuffled_gradients_list, # This was a bug, should be shuffled_gradients_list
            'comparison': comparison,
            'interpretation': self.interpret_spatial_results(original_analysis, comparison)
        }
        
        print("6. Spatial analysis complete!")
        print(f"   Mean attention: {original_analysis['stats']['mean_attention']:.3f}")
        print(f"   Spatial focus: {original_analysis['stats']['mean_spatial_focus']:.3f}")
        print(f"   Attention consistency: {comparison['attention_consistency_ratio']['mean_correlation']:.3f}")
        print(f"   Interpretation: {results['interpretation']}")
        
        return results
    
    def interpret_spatial_results(self, original_analysis, comparison):
        """Interpret the spatial analysis results"""
        interpretation = []
        
        # Attention consistency interpretation
        attention_corr = comparison['attention_consistency_ratio']['mean_correlation']
        if attention_corr > 0.8:
            interpretation.append("High appearance reliance - spatial attention consistent across shuffles")
        elif attention_corr < 0.5:
            interpretation.append("High motion reliance - spatial attention changes with temporal order")
        else:
            interpretation.append("Mixed reliance - spatial attention moderately affected by shuffling")
        
        # Spatial focus interpretation
        spatial_focus = original_analysis['stats']['mean_spatial_focus']
        if spatial_focus > 2.0:
            interpretation.append("Focused attention - model attends to specific regions")
        elif spatial_focus < 1.2:
            interpretation.append("Diffuse attention - model attends broadly across frame")
        else:
            interpretation.append("Moderate focus - model has intermediate spatial attention")
        
        # Temporal consistency interpretation
        temporal_corr = original_analysis['temporal_consistency']['mean_temporal_correlation']
        if temporal_corr > 0.7:
            interpretation.append("Temporally consistent attention - similar regions attended across frames")
        elif temporal_corr < 0.3:
            interpretation.append("Temporally variable attention - different regions attended across frames")
        else:
            interpretation.append("Moderate temporal consistency in attention")
        
        return "; ".join(interpretation)
    
    def analyze_video(self, video_info):
        """Analyze a single video"""
        video_id = video_info.get('id', 'unknown')
        print(f"\nAnalyzing video {video_id}...")
        
        # Get video path
        if 'file_names' in video_info and video_info['file_names']:
            video_path = Path(self.config.video_data_root) / Path(video_info['file_names'][0]).parent
        else:
            print(f"No file_names found for video {video_id}")
            return None
        
        # Load video frames
        video_frames = self.load_video_frames(
            str(video_path), 
            max_frames=self.config.max_frames_per_video
        )
        
        if video_frames is None:
            print(f"Failed to load frames for video {video_id}")
            return None
        
        print(f"Loaded {len(video_frames)} frames")
        
        # Perform spatial gradient analysis
        try:
            results = self.analyze_appearance_vs_motion_spatial(video_frames, video_id)
        except Exception as e:
            print(f"Error in spatial gradient analysis: {e}")
            # Clear memory and try with smaller chunk
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try with first 15 frames only
            print("Trying with reduced frame count...")
            video_frames_reduced = video_frames[:15]
            results = self.analyze_appearance_vs_motion_spatial(video_frames_reduced, video_id)
        
        if results is None:
            print(f"Failed to analyze video {video_id}")
            return None
        
        # Save results
        output_file = self.output_dir / f"spatial_gradient_analysis_{video_id}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Spatial gradient analysis saved to {output_file}")
        
        return results
    
    def run_analysis(self):
        """Run the complete spatial gradient analysis"""
        print("Starting spatial gradient analysis...")
        
        # Create category_id to category_name mapping
        category_map = {}
        for category in self.dataset.get('categories', []):
            category_map[category['id']] = category['name']
        
        print(f"Category mapping: {category_map}")
        
        # Filter videos by species
        target_species_ids = []
        for category_id, category_name in category_map.items():
            if category_name in self.config.target_species:
                target_species_ids.append(category_id)
        
        print(f"Target species IDs: {target_species_ids}")
        
        # Create video to species mapping
        video_species_map = {}
        for annotation in self.dataset.get('annotations', []):
            video_id = annotation.get('video_id')
            category_id = annotation.get('category_id')
            if category_id in target_species_ids:
                video_species_map[video_id] = category_id
        
        # Filter videos
        filtered_videos = []
        for video in self.dataset['videos']:
            video_id = video.get('id')
            if video_id in video_species_map:
                video['category_id'] = video_species_map[video_id]
                filtered_videos.append(video)
        
        print(f"Found {len(filtered_videos)} videos for target species")
        
        # Limit number of videos
        if self.config.max_videos_per_species:
            filtered_videos = filtered_videos[:self.config.max_videos_per_species]
            print(f"Limited to {len(filtered_videos)} videos")
        
        # Analyze videos
        results = []
        for i, video_info in enumerate(tqdm(filtered_videos, desc="Analyzing videos")):
            print(f"\nProcessing video {i+1}/{len(filtered_videos)}")
            result = self.analyze_video(video_info)
            if result:
                results.append(result)
            
            # Clear memory
            gc.collect()
        
        # Clean up GradCAM hooks
        self.gradcam.remove_hooks()
        
        # Save summary results
        summary_file = self.output_dir / "spatial_gradient_summary.json"
        
        # Convert results to JSON-serializable format
        summary_results = []
        for result in results:
            summary_result = {
                'video_id': result['video_id'],
                'num_frames': result['num_frames'],
                'mean_attention': float(result['original_analysis']['stats']['mean_attention']),
                'mean_spatial_focus': float(result['original_analysis']['stats']['mean_spatial_focus']),
                'temporal_correlation': float(result['original_analysis']['temporal_consistency']['mean_temporal_correlation']),
                'attention_consistency': float(result['comparison']['attention_consistency_ratio']['mean_correlation']),
                'focus_consistency_ratio': float(result['comparison']['spatial_focus_consistency']['focus_consistency_ratio']),
                'interpretation': result['interpretation']
            }
            summary_results.append(summary_result)
        
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to {summary_file}")
        print(f"Processed {len(results)} videos successfully")
        
        return results

def main():
    """Main function"""
    # Create configuration
    config = create_memory_efficient_config(gpu_memory_gb=24)  # Use 24GB GPU
    config.max_videos_per_species = 1  # Test with 1 video
    config.max_frames_per_video = 20   # Use middle 20 frames for better memory efficiency
    
    # Create analyzer
    analyzer = SpatialGradientAnalyzer(config)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results:
        print("\nSpatial Gradient Analysis Results:")
        for result in results:
            print(f"\nVideo {result['video_id']}:")
            print(f"  Mean attention: {result['original_analysis']['stats']['mean_attention']:.3f}")
            print(f"  Spatial focus: {result['original_analysis']['stats']['mean_spatial_focus']:.3f}")
            print(f"  Temporal correlation: {result['original_analysis']['temporal_consistency']['mean_temporal_correlation']:.3f}")
            print(f"  Attention consistency: {result['comparison']['attention_consistency_ratio']['mean_correlation']:.3f}")
            print(f"  Focus consistency ratio: {result['comparison']['spatial_focus_consistency']['focus_consistency_ratio']:.3f}")
            print(f"  Interpretation: {result['interpretation']}")

if __name__ == "__main__":
    main()
