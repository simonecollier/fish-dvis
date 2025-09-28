#!/usr/bin/env python3
"""
GPU-Based Gradient Extractor for DVIS-DAQ
Uses GPU memory optimization techniques for 31-frame sequences
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

class GPUGradientExtractor:
    """Extract temporal gradients using GPU with memory optimization"""
    
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
        
        # Load dataset
        with open(config.dataset_json_path, 'r') as f:
            self.dataset = json.load(f)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
            frame_tensor.requires_grad_(True)
            frame_list.append(frame_tensor)
        
        return frame_list
    
    def extract_temporal_gradients_gpu(self, video_frames, video_id):
        """Extract temporal gradients using GPU with memory optimization"""
        print(f"Extracting temporal gradients using GPU - Video {video_id}")
        print(f"Processing {len(video_frames)} frames")
        
        # Preprocess all frames
        processed_frames = self.preprocess_frames(video_frames)
        
        try:
            # Method 1: Try with gradient checkpointing
            print("Attempting GPU processing with gradient checkpointing...")
            gradients = self._extract_gradients_gpu_checkpointing(processed_frames)
            
            if gradients is not None:
                print("✅ Successfully extracted gradients with GPU checkpointing")
                return gradients
            
            # Method 2: Try with reduced precision
            print("Attempting GPU processing with reduced precision...")
            gradients = self._extract_gradients_gpu_fp16(processed_frames)
            
            if gradients is not None:
                print("✅ Successfully extracted gradients with FP16")
                return gradients
            
            # Method 3: Try with memory-efficient processing
            print("Attempting GPU processing with memory optimization...")
            gradients = self._extract_gradients_gpu_memory_optimized(processed_frames)
            
            if gradients is not None:
                print("✅ Successfully extracted gradients with memory optimization")
                return gradients
            
            print("❌ All GPU methods failed")
            return None
            
        except Exception as e:
            print(f"Error in GPU processing: {e}")
            return None
    
    def _extract_gradients_gpu_checkpointing(self, processed_frames):
        """Extract gradients using GPU with gradient checkpointing"""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
            print(f"GPU checkpointing method failed: {e}")
            return None
    
    def _extract_gradients_gpu_fp16(self, processed_frames):
        """Extract gradients using GPU with FP16 precision"""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Stack all frames and convert to FP16
            images_tensor = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
            images_tensor = images_tensor.half()  # Convert to FP16
            
            # Enable gradients
            torch.enable_grad()
            
            # Forward pass through the model
            print("Running model forward pass with FP16...")
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
            print(f"GPU FP16 method failed: {e}")
            return None
    
    def _extract_gradients_gpu_memory_optimized(self, processed_frames):
        """Extract gradients using GPU with memory optimization"""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process in smaller batches to reduce memory usage
            batch_size = 10  # Process 10 frames at a time
            all_gradients = []
            
            for i in range(0, len(processed_frames), batch_size):
                batch_frames = processed_frames[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}: frames {i}-{min(i+batch_size-1, len(processed_frames)-1)}")
                
                # Stack batch frames
                batch_tensor = torch.stack(batch_frames, dim=0)  # (batch_size, C, H, W)
                
                # Enable gradients
                torch.enable_grad()
                
                # Forward pass through the model
                features = self.model.backbone(batch_tensor)
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
                    'padding_masks': torch.zeros(1, out['pred_logits'].shape[1], len(batch_tensor), dtype=torch.bool, device=self.config.device),
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
                
                # Extract gradients for this batch
                for frame in batch_frames:
                    if frame.grad is not None:
                        grad_magnitude = torch.norm(frame.grad, dim=0)  # (H, W)
                        all_gradients.append(grad_magnitude.detach().cpu().numpy())
                    else:
                        all_gradients.append(np.zeros((frame.shape[1], frame.shape[2])))
                
                # Clear gradients for next batch
                for frame in batch_frames:
                    if frame.grad is not None:
                        frame.grad.zero_()
                
                # Memory cleanup
                del batch_tensor, features, out, outputs, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return np.array(all_gradients)
            
        except Exception as e:
            print(f"GPU memory optimization method failed: {e}")
            return None
    
    def analyze_video(self, video_info):
        """Analyze a single video with GPU gradient extraction"""
        video_id = video_info.get('id', 'unknown')
        print(f"\nAnalyzing video {video_id} with GPU approach...")
        
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
        
        # Extract temporal gradients using GPU
        temporal_gradients = self.extract_temporal_gradients_gpu(video_frames, video_id)
        
        if temporal_gradients is None:
            print(f"Failed to extract temporal gradients for video {video_id}")
            return None
        
        # Save results
        output_file = self.output_dir / f"gpu_gradients_{video_id}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump({
                'video_id': video_id,
                'num_frames': len(video_frames),
                'temporal_gradients': temporal_gradients,
                'gradient_shape': temporal_gradients.shape,
                'mean_gradient': float(np.mean(temporal_gradients)),
                'std_gradient': float(np.std(temporal_gradients))
            }, f)
        
        print(f"GPU gradients saved to {output_file}")
        print(f"Gradient shape: {temporal_gradients.shape}")
        print(f"Mean gradient: {np.mean(temporal_gradients):.6f}")
        
        return {
            'video_id': video_id,
            'temporal_gradients': temporal_gradients,
            'num_frames': len(video_frames)
        }
    
    def run_analysis(self):
        """Run the complete GPU analysis"""
        print("Starting GPU-based gradient extraction...")
        
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\nAnalysis complete! Processed {len(results)} videos successfully")
        
        return results

def main():
    """Main function"""
    # Create configuration
    config = create_memory_efficient_config(gpu_memory_gb=24)  # Use 24GB GPU
    config.max_videos_per_species = 1  # Test with 1 video
    config.max_frames_per_video = 20   # Use middle 20 frames for better memory efficiency
    
    # Create extractor
    extractor = GPUGradientExtractor(config)
    
    # Run analysis
    results = extractor.run_analysis()
    
    if results:
        print("\nGPU Gradient Extraction Results:")
        for result in results:
            print(f"\nVideo {result['video_id']}:")
            print(f"  Frames processed: {result['num_frames']}")
            print(f"  Gradient shape: {result['temporal_gradients'].shape}")
            print(f"  Mean gradient: {np.mean(result['temporal_gradients']):.6f}")
    else:
        print("No results obtained from analysis")

if __name__ == "__main__":
    main()
