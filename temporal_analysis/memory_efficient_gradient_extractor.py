#!/usr/bin/env python3
"""
Memory-Efficient Gradient Extractor for DVIS-DAQ
Processes full 31-frame sequences without chunking using memory optimization techniques
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
from torch.utils.checkpoint import checkpoint

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from memory_efficient_config import create_memory_efficient_config
from utils.model_loader import DVISModelLoader

class MemoryEfficientGradientExtractor:
    """Extract temporal gradients for full sequences using memory optimization"""
    
    def __init__(self, config):
        self.config = config
        self.config.device = "cpu"
        self.config.force_cpu_offload = True
        
        # Load model with careful device handling
        print("Loading patched DVIS-DAQ model...")
        self.model_loader = DVISModelLoader(
            config_path=str(config.config_path),
            checkpoint_path=str(config.model_path),
            device="cpu"
        )
        self.model = self.model_loader.load_model()
        
        # Force model to CPU
        self._force_model_to_cpu()
        
        # Load dataset
        with open(config.dataset_json_path, 'r') as f:
            self.dataset = json.load(f)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _force_model_to_cpu(self):
        """Force all model components to CPU"""
        print("Forcing model to CPU...")
        
        # Move model to CPU
        self.model = self.model.cpu()
        
        # Force all parameters to CPU
        for param in self.model.parameters():
            param.data = param.data.cpu()
        
        # Force all buffers to CPU
        for buffer in self.model.buffers():
            buffer.data = buffer.data.cpu()
        
        # Force all modules to CPU recursively
        for module in self.model.modules():
            module.to('cpu')
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Verify model is on CPU
        device = next(self.model.parameters()).device
        print(f"Model device after forcing to CPU: {device}")
        
        if device.type != 'cpu':
            raise RuntimeError(f"Failed to move model to CPU. Model is on {device}")
    
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
        
        # Limit frames if specified
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        frames = []
        for frame_file in frame_files:
            try:
                frame = cv2.imread(str(frame_file))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 640))
                frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
                frames.append(frame)
            except Exception as e:
                print(f"Error loading frame {frame_file}: {e}")
                continue
        
        if not frames:
            print(f"No frames loaded from {video_dir}")
            return None
        
        return np.array(frames)
    
    def preprocess_frames(self, video_frames):
        """Preprocess frames for model input with memory optimization"""
        print("Preprocessing frames with memory optimization...")
        
        frame_list = []
        for t in range(len(video_frames)):
            frame = video_frames[t]  # (H, W, C) normalized
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8
            frame_tensor = torch.as_tensor(np.ascontiguousarray(frame.transpose(2, 0, 1)))  # (C, H, W)
            frame_tensor = frame_tensor.to("cpu", dtype=torch.float32)
            frame_tensor.requires_grad_(True)
            frame_list.append(frame_tensor)
        
        return frame_list
    
    def extract_temporal_gradients_full_sequence(self, video_frames, video_id):
        """Extract temporal gradients for full sequence using memory optimization"""
        print(f"Extracting temporal gradients for full sequence - Video {video_id}")
        print(f"Processing {len(video_frames)} frames without chunking")
        
        # Preprocess all frames
        processed_frames = self.preprocess_frames(video_frames)
        
        try:
            # Method 1: Use gradient checkpointing for memory efficiency
            print("Using gradient checkpointing for memory-efficient full sequence processing...")
            gradients = self._extract_gradients_with_checkpointing(processed_frames)
            
            if gradients is not None:
                print("✅ Successfully extracted gradients with checkpointing")
                return gradients
            
            print("❌ Gradient checkpointing failed")
            return None
            
        except Exception as e:
            print(f"Error in full sequence processing: {e}")
            return None
    
    def _extract_gradients_with_checkpointing(self, processed_frames):
        """Extract gradients using gradient checkpointing for memory efficiency"""
        try:
            # Stack all frames
            images_tensor = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
            
            # Enable gradients
            torch.enable_grad()
            
            # Use gradient checkpointing for memory efficiency
            def forward_with_checkpointing(input_tensor):
                # Forward pass through the model
                features = self.model.backbone(input_tensor)
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
                    'padding_masks': torch.zeros(1, out['pred_logits'].shape[1], len(input_tensor), dtype=torch.bool, device=self.config.device),
                    'seq_id_list': [],
                    'dead_seq_id_list': []
                }
                
                # Create a loss for gradient computation
                if 'online_out' in outputs and 'pred_logits' in outputs['online_out']:
                    pred_logits = outputs['online_out']['pred_logits']
                    if isinstance(pred_logits, list):
                        pred_logits = pred_logits[0]
                    loss = pred_logits.sum()
                else:
                    # Fallback
                    first_output = list(outputs.values())[0]
                    if isinstance(first_output, list):
                        first_output = first_output[0]
                    loss = first_output.sum()
                
                return loss
            
            # Use checkpointing
            loss = checkpoint(forward_with_checkpointing, images_tensor)
            
            # Backward pass
            loss.backward()
            
            # Extract gradients
            temporal_gradients = []
            for frame in processed_frames:
                if frame.grad is not None:
                    grad_magnitude = torch.norm(frame.grad, dim=0)  # (H, W)
                    temporal_gradients.append(grad_magnitude.detach().cpu().numpy())
                else:
                    temporal_gradients.append(np.zeros((frame.shape[1], frame.shape[2])))
            
            return np.array(temporal_gradients)
            
        except Exception as e:
            print(f"Checkpointing method failed: {e}")
            return None
    
    def _extract_gradients_full_sequence(self, processed_frames):
        """Extract gradients for full sequence without checkpointing"""
        try:
            # Stack all frames
            images_tensor = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
            
            # Enable gradients
            torch.enable_grad()
            
            # Forward pass through the model
            print("Running model forward pass...")
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
                'padding_masks': torch.zeros(1, out['pred_logits'].shape[1], len(images_tensor), dtype=torch.bool, device=self.config.device),
                'seq_id_list': [],
                'dead_seq_id_list': []
            }
            
            # Create a loss for gradient computation
            if 'online_out' in outputs and 'pred_logits' in outputs['online_out']:
                pred_logits = outputs['online_out']['pred_logits']
                if isinstance(pred_logits, list):
                    pred_logits = pred_logits[0]
                loss = pred_logits.sum()
            else:
                # Fallback
                first_output = list(outputs.values())[0]
                if isinstance(first_output, list):
                    first_output = first_output[0]
                loss = first_output.sum()
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Extract gradients for each frame
            temporal_gradients = []
            for frame in processed_frames:
                if frame.grad is not None:
                    grad_magnitude = torch.norm(frame.grad, dim=0)  # (H, W)
                    temporal_gradients.append(grad_magnitude.detach().cpu().numpy())
                else:
                    temporal_gradients.append(np.zeros((frame.shape[1], frame.shape[2])))
            
            return np.array(temporal_gradients)
            
        except Exception as e:
            print(f"Full sequence method failed: {e}")
            return None
    
    def _extract_gradients_sliding_window(self, processed_frames):
        """Extract gradients using sliding window approach"""
        try:
            window_size = 15  # Smaller window size
            stride = 5        # Overlap between windows
            
            all_gradients = []
            
            for start_idx in range(0, len(processed_frames), stride):
                end_idx = min(start_idx + window_size, len(processed_frames))
                window_frames = processed_frames[start_idx:end_idx]
                
                print(f"Processing window {start_idx//stride + 1}: frames {start_idx}-{end_idx-1}")
                
                # Process this window
                window_gradients = self._extract_gradients_full_sequence(window_frames)
                
                if window_gradients is not None:
                    # Add gradients for this window
                    if start_idx == 0:
                        # First window: add all gradients
                        all_gradients.extend(window_gradients)
                    else:
                        # Subsequent windows: add only non-overlapping gradients
                        overlap_size = window_size - stride
                        non_overlap_gradients = window_gradients[overlap_size:]
                        all_gradients.extend(non_overlap_gradients)
                
                # Clear gradients for next window
                for frame in window_frames:
                    if frame.grad is not None:
                        frame.grad.zero_()
                
                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Ensure we have exactly one gradient per frame
            if all_gradients:
                temporal_gradients = np.array(all_gradients)
                
                # Pad or truncate to match frame count
                if len(temporal_gradients) < len(processed_frames):
                    padding_shape = (len(processed_frames) - len(temporal_gradients),) + temporal_gradients.shape[1:]
                    padding = np.zeros(padding_shape)
                    temporal_gradients = np.concatenate([temporal_gradients, padding], axis=0)
                elif len(temporal_gradients) > len(processed_frames):
                    temporal_gradients = temporal_gradients[:len(processed_frames)]
                
                return temporal_gradients
            
            return None
            
        except Exception as e:
            print(f"Sliding window method failed: {e}")
            return None
    
    def analyze_video(self, video_info):
        """Analyze a single video with memory-efficient gradient extraction"""
        video_id = video_info.get('id', 'unknown')
        print(f"\nAnalyzing video {video_id} with memory-efficient approach...")
        
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
        
        # Extract temporal gradients using memory-efficient approach
        temporal_gradients = self.extract_temporal_gradients_full_sequence(video_frames, video_id)
        
        if temporal_gradients is None:
            print(f"Failed to extract temporal gradients for video {video_id}")
            return None
        
        # Save results
        output_file = self.output_dir / f"memory_efficient_gradients_{video_id}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump({
                'video_id': video_id,
                'num_frames': len(video_frames),
                'temporal_gradients': temporal_gradients,
                'gradient_shape': temporal_gradients.shape,
                'mean_gradient': float(np.mean(temporal_gradients)),
                'std_gradient': float(np.std(temporal_gradients))
            }, f)
        
        print(f"Memory-efficient gradients saved to {output_file}")
        print(f"Gradient shape: {temporal_gradients.shape}")
        print(f"Mean gradient: {np.mean(temporal_gradients):.6f}")
        
        return {
            'video_id': video_id,
            'temporal_gradients': temporal_gradients,
            'num_frames': len(video_frames)
        }
    
    def run_analysis(self):
        """Run the complete memory-efficient analysis"""
        print("Starting memory-efficient gradient extraction...")
        
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
        
        print(f"\nAnalysis complete! Processed {len(results)} videos successfully")
        
        return results

def main():
    """Main function"""
    # Create configuration
    config = create_memory_efficient_config(gpu_memory_gb=8)
    config.max_videos_per_species = 1  # Test with 1 video
    config.max_frames_per_video = 31   # Use full 31-frame window
    
    # Create extractor
    extractor = MemoryEfficientGradientExtractor(config)
    
    # Run analysis
    results = extractor.run_analysis()
    
    if results:
        print("\nMemory-Efficient Gradient Extraction Results:")
        for result in results:
            print(f"\nVideo {result['video_id']}:")
            print(f"  Frames processed: {result['num_frames']}")
            print(f"  Gradient shape: {result['temporal_gradients'].shape}")
            print(f"  Mean gradient: {np.mean(result['temporal_gradients']):.6f}")
    else:
        print("No results obtained from analysis")

if __name__ == "__main__":
    main()
