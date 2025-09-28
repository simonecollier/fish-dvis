#!/usr/bin/env python3
"""
Multi-Window Temporal Gradient Analyzer for DVIS-DAQ Model

This script analyzes temporal gradients across all 30-frame windows in a video,
extracting both gradient information and classification confidence for each window.
"""

import os
import sys
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple, Optional
import gc

# Add the DVIS-DAQ path to sys.path
sys.path.insert(0, '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ')

from dvis_daq import add_daq_config
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

class MultiWindowGradientAnalyzer:
    def __init__(self, config_path: str, model_path: str, device: str = 'cpu'):
        """
        Initialize the multi-window gradient analyzer.
        
        Args:
            config_path: Path to the model configuration file
            model_path: Path to the trained model weights
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device
        self.config_path = config_path
        self.model_path = model_path
        self.model = None
        self.window_size = 30  # Standard DVIS-DAQ window size
        
        # Load the model
        self._load_model()
        
    def _load_model(self):
        """Load the DVIS-DAQ model with proper configuration."""
        print(f"Loading DVIS-DAQ model from {self.model_path}")
        
        # Load configuration
        cfg = get_cfg()
        add_daq_config(cfg)
        cfg.merge_from_file(self.config_path)
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.MODEL.DEVICE = self.device
        
        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self.model = self.predictor.model
        
        # Force model to CPU if needed
        if self.device == 'cpu':
            self._force_model_to_cpu()
        
        print(f"✅ Model loaded successfully on {self.device}")
        
    def _force_model_to_cpu(self):
        """Force all model components to CPU."""
        print("Forcing model to CPU...")
        
        # Move model to CPU
        self.model = self.model.cpu()
        
        # Move all parameters and buffers to CPU
        for param in self.model.parameters():
            param.data = param.data.cpu()
        
        for buffer in self.model.buffers():
            buffer.data = buffer.data.cpu()
        
        # Move all submodules to CPU
        for module in self.model.modules():
            module.to('cpu')
        
        print("✅ Model successfully moved to CPU")
    
    def _extract_classification_confidence(self, outputs: Dict) -> Tuple[float, int]:
        """
        Extract classification confidence and predicted class from model outputs.
        
        Args:
            outputs: Model outputs dictionary
            
        Returns:
            Tuple of (confidence_score, predicted_class)
        """
        try:
            if 'online_out' in outputs and 'pred_logits' in outputs['online_out']:
                pred_logits = outputs['online_out']['pred_logits']
                
                # Handle different tensor shapes
                if len(pred_logits.shape) == 3:  # (batch, num_instances, num_classes)
                    logits = pred_logits[0]  # Take first batch
                elif len(pred_logits.shape) == 2:  # (num_instances, num_classes)
                    logits = pred_logits
                else:
                    print(f"Unexpected pred_logits shape: {pred_logits.shape}")
                    return 0.0, -1
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Get the highest probability and corresponding class for each instance
                max_probs, predicted_classes = torch.max(probs, dim=-1)
                
                # Return the highest confidence instance (excluding background class)
                # Background is typically the last class
                valid_mask = predicted_classes < (probs.shape[-1] - 1)  # Exclude background
                
                if valid_mask.any():
                    # Get the instance with highest confidence
                    max_conf_idx = torch.argmax(max_probs[valid_mask])
                    confidence = max_probs[valid_mask][max_conf_idx].item()
                    predicted_class = predicted_classes[valid_mask][max_conf_idx].item()
                else:
                    # No valid instances found
                    confidence = 0.0
                    predicted_class = -1
                
                return confidence, predicted_class
            else:
                print("No pred_logits found in outputs")
                return 0.0, -1
                
        except Exception as e:
            print(f"Error extracting classification confidence: {e}")
            return 0.0, -1
    
    def _extract_temporal_gradients(self, frame_list: List[torch.Tensor], 
                                  outputs: Dict) -> Optional[np.ndarray]:
        """
        Extract temporal gradients from model outputs with respect to input frames.
        
        Args:
            frame_list: List of input frame tensors
            outputs: Model outputs dictionary
            
        Returns:
            Temporal gradients array of shape (num_frames, H, W) or None if failed
        """
        try:
            # Create a loss for gradient computation
            if 'online_out' in outputs and 'pred_logits' in outputs['online_out']:
                pred_logits = outputs['online_out']['pred_logits']
                
                # Use the sum of logits as loss (this preserves gradients)
                if isinstance(pred_logits, list):
                    pred_logits = pred_logits[0]
                loss = pred_logits.sum()
                
            elif 'online_out' in outputs and 'pred_masks' in outputs['online_out']:
                pred_masks = outputs['online_out']['pred_masks']
                if isinstance(pred_masks, list):
                    pred_masks = pred_masks[0]
                loss = pred_masks.sum()
                
            else:
                # Fallback: use the first available output
                first_output = list(outputs.values())[0]
                if isinstance(first_output, list):
                    first_output = first_output[0]
                loss = first_output.sum()
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Extract gradients for each frame
            temporal_gradients = []
            gradients_found = False
            
            for frame in frame_list:
                if frame.grad is not None:
                    # Compute gradient magnitude across channels
                    grad_magnitude = torch.norm(frame.grad, dim=0)  # (H, W)
                    temporal_gradients.append(grad_magnitude.detach().cpu().numpy())
                    gradients_found = True
                else:
                    # No gradients, use zeros
                    temporal_gradients.append(np.zeros((frame.shape[1], frame.shape[2])))
            
            if gradients_found:
                return np.array(temporal_gradients)
            else:
                print("❌ No gradients found in model forward pass")
                return None
                
        except Exception as e:
            print(f"Error extracting temporal gradients: {e}")
            return None
    
    def _load_video_frames(self, video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
        """
        Load video frames from a directory of frame images.
        
        Args:
            video_path: Path to the video directory containing frame images
            max_frames: Maximum number of frames to load (None for all)
            
        Returns:
            Array of frames with shape (num_frames, H, W, C)
        """
        video_dir = Path(video_path)
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_path}")
        
        # Get all frame files
        frame_files = sorted([f for f in video_dir.glob("*.jpg")])
        if not frame_files:
            raise FileNotFoundError(f"No frame files found in {video_path}")
        
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        print(f"Loading {len(frame_files)} frames from {video_path}")
        
        frames = []
        for frame_file in tqdm(frame_files, desc="Loading frames"):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"Warning: Could not load frame {frame_file}")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        if not frames:
            raise ValueError("No valid frames loaded")
        
        return np.array(frames)
    
    def _preprocess_frames(self, frames: np.ndarray) -> List[torch.Tensor]:
        """
        Preprocess frames for model input.
        
        Args:
            frames: Array of frames with shape (num_frames, H, W, C)
            
        Returns:
            List of preprocessed frame tensors
        """
        processed_frames = []
        
        for frame in frames:
            # Normalize to [0, 1]
            frame_normalized = frame.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
            
            # Move to device
            frame_tensor = frame_tensor.to(self.device)
            frame_tensor.requires_grad_(True)
            
            processed_frames.append(frame_tensor)
        
        return processed_frames
    
    def analyze_video_windows(self, video_path: str, output_dir: str) -> Dict:
        """
        Analyze temporal gradients for all windows in a video.
        
        Args:
            video_path: Path to the video directory
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing analysis results for all windows
        """
        print(f"\n🔍 Analyzing video: {video_path}")
        
        # Load all frames
        frames = self._load_video_frames(video_path)
        total_frames = len(frames)
        
        print(f"Total frames loaded: {total_frames}")
        
        # Calculate number of windows
        num_windows = max(1, (total_frames - self.window_size) // (self.window_size // 2) + 1)
        print(f"Number of 30-frame windows: {num_windows}")
        
        # Results storage
        window_results = []
        
        # Process each window
        for window_idx in range(num_windows):
            start_frame = window_idx * (self.window_size // 2)
            end_frame = min(start_frame + self.window_size, total_frames)
            
            # Ensure we have enough frames for a full window
            if end_frame - start_frame < self.window_size // 2:
                print(f"Skipping window {window_idx}: insufficient frames ({end_frame - start_frame})")
                continue
            
            print(f"\n📊 Processing window {window_idx + 1}/{num_windows}: frames {start_frame}-{end_frame}")
            
            # Extract window frames
            window_frames = frames[start_frame:end_frame]
            
            # Preprocess frames
            frame_list = self._preprocess_frames(window_frames)
            
            # Create input tensor
            images_tensor = torch.cat(frame_list, dim=0)
            
            try:
                # Enable gradients
                torch.enable_grad()
                
                # Forward pass through the model
                print("Running model forward pass...")
                
                # Use the gradient-preserving inference path
                features = self.model.backbone(images_tensor)
                out = self.model.sem_seg_head(features)
                
                # Create outputs in the same format as common_inference
                outputs = {
                    'frame_embeds': out['pred_embds'],
                    'mask_features': out['mask_features'],
                    'online_out': {
                        'pred_logits': out['pred_logits'],
                        'pred_masks': out['pred_masks']
                    },
                    'instance_embeds': out['pred_embds'],
                    'padding_masks': torch.zeros(1, out['pred_logits'].shape[1], len(images_tensor), dtype=torch.bool, device=self.device),
                    'seq_id_list': [],
                    'dead_seq_id_list': []
                }
                
                # Extract classification confidence
                confidence, predicted_class = self._extract_classification_confidence(outputs)
                
                # Extract temporal gradients
                temporal_gradients = self._extract_temporal_gradients(frame_list, outputs)
                
                # Calculate gradient statistics
                if temporal_gradients is not None:
                    mean_gradient = np.mean(temporal_gradients)
                    max_gradient = np.max(temporal_gradients)
                    std_gradient = np.std(temporal_gradients)
                    
                    # Calculate temporal gradient statistics (changes between consecutive frames)
                    temporal_changes = []
                    for i in range(1, len(temporal_gradients)):
                        change = np.mean(np.abs(temporal_gradients[i] - temporal_gradients[i-1]))
                        temporal_changes.append(change)
                    
                    mean_temporal_change = np.mean(temporal_changes) if temporal_changes else 0.0
                    max_temporal_change = np.max(temporal_changes) if temporal_changes else 0.0
                else:
                    mean_gradient = max_gradient = std_gradient = 0.0
                    mean_temporal_change = max_temporal_change = 0.0
                
                # Store window results
                window_result = {
                    'window_idx': window_idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'num_frames': end_frame - start_frame,
                    'classification_confidence': confidence,
                    'predicted_class': predicted_class,
                    'mean_gradient': float(mean_gradient),
                    'max_gradient': float(max_gradient),
                    'std_gradient': float(std_gradient),
                    'mean_temporal_change': float(mean_temporal_change),
                    'max_temporal_change': float(max_temporal_change),
                    'temporal_gradients': temporal_gradients.tolist() if temporal_gradients is not None else None
                }
                
                window_results.append(window_result)
                
                print(f"✅ Window {window_idx + 1} completed:")
                print(f"   Confidence: {confidence:.4f}")
                print(f"   Predicted class: {predicted_class}")
                print(f"   Mean gradient: {mean_gradient:.6f}")
                print(f"   Max gradient: {max_gradient:.6f}")
                print(f"   Mean temporal change: {mean_temporal_change:.6f}")
                
                # Clear gradients for next iteration
                for frame in frame_list:
                    if frame.grad is not None:
                        frame.grad.zero_()
                
                # Clear GPU memory
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error processing window {window_idx}: {e}")
                window_results.append({
                    'window_idx': window_idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'error': str(e)
                })
        
        # Create summary results
        summary = {
            'video_path': video_path,
            'total_frames': total_frames,
            'num_windows': len(window_results),
            'window_size': self.window_size,
            'window_results': window_results,
            'summary_stats': self._compute_summary_stats(window_results)
        }
        
        # Save results
        output_path = Path(output_dir) / f"{Path(video_path).name}_multi_window_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        
        return summary
    
    def _compute_summary_stats(self, window_results: List[Dict]) -> Dict:
        """Compute summary statistics across all windows."""
        valid_results = [r for r in window_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid windows processed'}
        
        confidences = [r['classification_confidence'] for r in valid_results]
        mean_gradients = [r['mean_gradient'] for r in valid_results]
        max_gradients = [r['max_gradient'] for r in valid_results]
        temporal_changes = [r['mean_temporal_change'] for r in valid_results]
        
        return {
            'mean_confidence': float(np.mean(confidences)),
            'max_confidence': float(np.max(confidences)),
            'mean_gradient_across_windows': float(np.mean(mean_gradients)),
            'max_gradient_across_windows': float(np.max(max_gradients)),
            'mean_temporal_change_across_windows': float(np.mean(temporal_changes)),
            'most_confident_window': int(np.argmax(confidences)),
            'highest_gradient_window': int(np.argmax(max_gradients))
        }

def main():
    """Main function to run the multi-window gradient analysis."""
    
    # Configuration
    config_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/DAQ_Fishway_config.yaml"
    model_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth"
    dataset_json_path = "/home/simone/shared-data/fishway_ytvis/val.json"
    video_data_root = "/home/simone/shared-data/fishway_ytvis/all_videos"
    output_dir = "/home/simone/fish-dvis/temporal_analysis/results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)
    
    # Create video species mapping
    video_species_map = {}
    for ann in dataset['annotations']:
        video_id = ann['video_id']
        category_id = ann['category_id']
        if video_id not in video_species_map:
            video_species_map[video_id] = category_id
    
    # Get category names
    category_names = {cat['id']: cat['name'] for cat in dataset['categories']}
    
    # Initialize analyzer
    analyzer = MultiWindowGradientAnalyzer(config_path, model_path, device='cpu')
    
    # Process videos
    for video_info in dataset['videos']:  # Process all videos
        video_id = video_info['id']
        video_name = video_info['file_names'][0].split('/')[0]  # Extract video name from path
        category_id = video_species_map.get(video_id, -1)
        category_name = category_names.get(category_id, 'unknown')
        
        print(f"\n🎬 Processing video {video_id}: {video_name} ({category_name})")
        
        # Construct video path
        video_path = os.path.join(video_data_root, video_name)
        
        if not os.path.exists(video_path):
            print(f"⚠️  Video directory not found: {video_path}")
            continue
        
        try:
            # Analyze video windows
            results = analyzer.analyze_video_windows(video_path, output_dir)
            
            # Print summary
            if 'summary_stats' in results and 'error' not in results['summary_stats']:
                stats = results['summary_stats']
                print(f"\n📊 Summary for {video_name}:")
                print(f"   Mean confidence: {stats['mean_confidence']:.4f}")
                print(f"   Mean gradient: {stats['mean_gradient_across_windows']:.6f}")
                print(f"   Most confident window: {stats['most_confident_window']}")
                print(f"   Highest gradient window: {stats['highest_gradient_window']}")
            
        except Exception as e:
            print(f"❌ Error processing video {video_name}: {e}")

if __name__ == "__main__":
    main()
