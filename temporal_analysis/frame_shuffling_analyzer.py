#!/usr/bin/env python3
"""
Frame Shuffling Gradient Analyzer for DVIS-DAQ Temporal Analysis
Analyzes whether the model relies more on appearance vs. motion for fish species classification
Uses GPU-based gradient extraction for unbiased temporal gradients
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
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Optional, Any

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from memory_efficient_config import create_memory_efficient_config
from utils.model_loader import DVISModelLoader

class FrameShufflingAnalyzer:
    """Analyze appearance vs. motion reliance using frame shuffling with GPU gradients"""
    
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
        
        # Analysis settings
        self.window_size = 31  # Full model window size
        self.num_shuffles = 5  # Number of random shuffles to test
        
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
        
        # Ensure model is in FP32 mode at the start
        self.model = self.model.float()
        
        try:
            # Method 1: Try with gradient checkpointing (full sequence)
            print("Attempting GPU processing with gradient checkpointing (full sequence)...")
            gradients = self._extract_gradients_gpu_checkpointing(processed_frames)
            
            if gradients is not None:
                print("✅ Successfully extracted gradients with GPU checkpointing (full sequence)")
                return gradients
            
            # Reset model to FP32 after failed attempt
            self.model = self.model.float()
            
            # Method 2: Try with reduced precision (full sequence)
            print("Attempting GPU processing with reduced precision (full sequence)...")
            gradients = self._extract_gradients_gpu_fp16(processed_frames)
            
            if gradients is not None:
                print("✅ Successfully extracted gradients with FP16 (full sequence)")
                return gradients
            
            # Reset model to FP32 after failed attempt
            self.model = self.model.float()
            
            # Method 3: Try with memory-efficient processing (batched as fallback)
            print("Attempting GPU processing with memory optimization (batched fallback)...")
            gradients = self._extract_gradients_gpu_memory_optimized(processed_frames)
            
            if gradients is not None:
                print("✅ Successfully extracted gradients with memory optimization (batched)")
                return gradients
            
            # Reset model to FP32 after failed attempt
            self.model = self.model.float()
            
            # Method 4: Try with reduced frame count
            print("Attempting GPU processing with reduced frame count...")
            if len(processed_frames) > 15:
                reduced_frames = processed_frames[:15]
                print(f"Reducing from {len(processed_frames)} to 15 frames")
                gradients = self._extract_gradients_gpu_memory_optimized(reduced_frames)
                if gradients is not None:
                    # Pad with zeros to match original length
                    padding = np.zeros((len(processed_frames) - 15, gradients.shape[1], gradients.shape[2]))
                    gradients = np.concatenate([gradients, padding], axis=0)
                    print("✅ Successfully extracted gradients with reduced frame count")
                    return gradients
            
            # Method 5: Try with very small batches (FP32 only)
            print("Attempting GPU processing with very small batches (FP32 only)...")
            gradients = self._extract_gradients_gpu_small_batches(processed_frames)
            if gradients is not None:
                print("✅ Successfully extracted gradients with small batches")
                return gradients
            
            print("❌ All GPU methods failed")
            return None
            
        except Exception as e:
            print(f"Error in GPU processing: {e}")
            # Ensure model is reset to FP32 on error
            self.model = self.model.float()
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
            from torch.utils.checkpoint import checkpoint
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
            
            # Convert model to FP16 for mixed precision
            print("Converting model to FP16 for mixed precision...")
            model_fp16 = self.model.half()
            
            # Stack all frames and convert to FP16
            images_tensor = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
            images_tensor = images_tensor.half()  # Convert to FP16
            
            # Enable gradients
            torch.enable_grad()
            
            # Forward pass through the model
            print("Running model forward pass with FP16...")
            features = model_fp16.backbone(images_tensor)
            out = model_fp16.sem_seg_head(features)
            
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
            
            # Convert model back to FP32
            self.model = self.model.float()
            
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
            
            # For 12-frame windows, try to process all frames at once first
            if len(processed_frames) <= 12:
                # Try to process all frames at once first
                print("Attempting to process all frames at once...")
                try:
                    all_frames_tensor = torch.stack(processed_frames, dim=0)
                    
                    # Enable gradients
                    torch.enable_grad()
                    
                    # Forward pass through the model
                    features = self.model.backbone(all_frames_tensor)
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
                        'padding_masks': torch.zeros(1, out['pred_logits'].shape[1], len(all_frames_tensor), dtype=torch.bool, device=self.config.device),
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
                    
                    # Extract gradients for all frames
                    all_gradients = []
                    for frame in processed_frames:
                        if frame.grad is not None:
                            grad_magnitude = torch.norm(frame.grad, dim=0)  # (H, W)
                            all_gradients.append(grad_magnitude.detach().cpu().numpy())
                        else:
                            all_gradients.append(np.zeros((frame.shape[1], frame.shape[2])))
                    
                    print("✅ Successfully processed all frames at once")
                    return np.array(all_gradients)
                    
                except Exception as e:
                    print(f"Failed to process all frames at once: {e}")
                    # Fall back to batching
                    pass
            
            # Fallback to batching with larger batch size for 20-frame windows
            batch_size = 15  # Process 15 frames at a time (larger than before)
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
    
    def _extract_gradients_gpu_small_batches(self, processed_frames):
        """Extract gradients using GPU with very small batches (FP32 only)"""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try progressively larger batch sizes for 12-frame windows
            batch_sizes = [12, 8, 6, 5]  # Try full 12 frames first, then fall back
            
            for batch_size in batch_sizes:
                try:
                    print(f"Trying batch size: {batch_size}")
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
                    
                    print(f"✅ Successfully processed with batch size {batch_size}")
                    return np.array(all_gradients)
                    
                except Exception as e:
                    print(f"Failed with batch size {batch_size}: {e}")
                    # Clear memory and try next batch size
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            print("❌ All batch sizes failed")
            return None
            
            for i in range(0, len(processed_frames), batch_size):
                batch_frames = processed_frames[i:i+batch_size]
                print(f"Processing small batch {i//batch_size + 1}: frames {i}-{min(i+batch_size-1, len(processed_frames)-1)}")
                
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
            print(f"GPU small batches method failed: {e}")
            return None
    
    def compute_optical_flow(self, video_frames: np.ndarray) -> np.ndarray:
        """Compute optical flow between consecutive frames"""
        # Ensure frames are in (T, H, W, C) format
        if video_frames.shape[1] == 3:  # (T, C, H, W)
            video_frames = np.transpose(video_frames, (0, 2, 3, 1))
        
        T, H, W, C = video_frames.shape
        
        # Convert to grayscale for optical flow
        gray_frames = []
        for frame in video_frames:
            if C == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame.squeeze()
            gray_frames.append(gray)
        
        gray_frames = np.array(gray_frames)
        
        # Compute optical flow
        motion_magnitudes = []
        
        for i in range(T - 1):
            prev_frame = gray_frames[i]
            curr_frame = gray_frames[i + 1]
            
            # Compute optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, 
                flags=0
            )
            
            # Compute motion magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mean_magnitude = np.mean(magnitude)
            motion_magnitudes.append(mean_magnitude)
        
        return np.array(motion_magnitudes)
    
    def compute_gradient_magnitudes(self, temporal_gradients):
        """Compute per-frame gradient magnitudes"""
        # Compute mean gradient magnitude for each frame
        gradient_magnitudes = []
        for frame_grad in temporal_gradients:
            magnitude = np.mean(frame_grad)  # Mean across spatial dimensions
            gradient_magnitudes.append(magnitude)
        return np.array(gradient_magnitudes)
    
    def analyze_appearance_vs_motion(self, video_frames, video_id):
        """Analyze appearance vs. motion reliance using frame shuffling"""
        # Extract video name from video_id if it contains window info
        video_name = video_id
        if '_window_' in video_id:
            base_video_id = video_id.split('_window_')[0]
            # Try to get the original video name from the dataset
            for video_info in self.dataset['videos']:
                if str(video_info.get('id')) == base_video_id:
                    if 'file_names' in video_info and video_info['file_names']:
                        video_path = Path(video_info['file_names'][0])
                        video_name = f"{base_video_id} ({video_path.parent.name})"
                    break
        
        print(f"\n=== Frame Shuffling Analysis for Video {video_name} ===")
        
        # Ensure we have enough frames
        if len(video_frames) < self.window_size:
            print(f"Warning: Only {len(video_frames)} frames available, using all frames")
            frames_to_process = video_frames
        else:
            frames_to_process = video_frames[:self.window_size]
        
        # 1. Extract original temporal gradients
        print("1. Extracting original temporal gradients...")
        original_gradients = self.extract_temporal_gradients_gpu(frames_to_process, video_id)
        
        if original_gradients is None:
            print("Failed to extract original gradients")
            return None
        
        print(f"Original gradients shape: {original_gradients.shape}")
        print(f"Number of frames processed: {len(frames_to_process)}")
        print(f"Number of gradients extracted: {len(original_gradients)}")
        
        # 2. Compute optical flow for motion analysis
        print("2. Computing optical flow...")
        motion_magnitudes = self.compute_optical_flow(frames_to_process)
        
        # 3. Extract original gradient magnitudes
        original_gradient_magnitudes = self.compute_gradient_magnitudes(original_gradients)
        
        # 4. Perform multiple shuffles
        print(f"3. Performing {self.num_shuffles} frame shuffles...")
        shuffled_results = []
        
        for shuffle_idx in range(self.num_shuffles):
            print(f"   Shuffle {shuffle_idx + 1}/{self.num_shuffles}")
            
            # Create random permutation
            permutation = np.random.permutation(len(frames_to_process))
            shuffled_frames = frames_to_process[permutation]
            
            # Extract gradients for shuffled frames
            shuffled_gradients = self.extract_temporal_gradients_gpu(shuffled_frames, f"{video_id}_shuffle_{shuffle_idx}")
            
            if shuffled_gradients is not None:
                print(f"   Shuffled gradients shape: {shuffled_gradients.shape}")
                print(f"   Shuffled frames processed: {len(shuffled_frames)}")
                print(f"   Shuffled gradients extracted: {len(shuffled_gradients)}")
                # Check if we have the same number of gradients as original
                if len(shuffled_gradients) == len(original_gradients):
                    # Align gradients back to original frame order using inverse permutation
                    inverse_permutation = np.argsort(permutation)
                    # Only use the indices that exist in the gradients
                    valid_indices = inverse_permutation[:len(shuffled_gradients)]
                    shuffled_gradients_aligned = shuffled_gradients[valid_indices]
                    
                    # Compute gradient magnitudes for aligned shuffled gradients
                    shuffled_gradient_magnitudes = self.compute_gradient_magnitudes(shuffled_gradients_aligned)
                    
                    # Store both aligned and original order for comparison
                    shuffled_results.append({
                        'permutation': permutation,
                        'shuffled_gradients': shuffled_gradients_aligned,
                        'shuffled_gradient_magnitudes': shuffled_gradient_magnitudes,
                        'shuffled_gradients_original_order': shuffled_gradients,
                        'shuffled_magnitudes_original_order': self.compute_gradient_magnitudes(shuffled_gradients)
                    })
                else:
                    print(f"   Warning: Shuffled gradients length ({len(shuffled_gradients)}) doesn't match original ({len(original_gradients)})")
                    # Use the shuffled gradients as-is, but only for the frames we have
                    shuffled_gradient_magnitudes = self.compute_gradient_magnitudes(shuffled_gradients)
                    
                    shuffled_results.append({
                        'permutation': permutation,
                        'shuffled_gradients': shuffled_gradients,
                        'shuffled_gradient_magnitudes': shuffled_gradient_magnitudes
                    })
            else:
                print(f"   Failed to extract gradients for shuffle {shuffle_idx}")
        
        if not shuffled_results:
            print("No successful shuffles completed")
            return None
        
        # 5. Compute motion reliance metrics
        print("4. Computing motion reliance metrics...")
        
        # Compute ratios for each shuffle
        all_ratios = []
        all_flow_correlations_original = []
        all_flow_correlations_shuffled = []
        
        for shuffle_result in shuffled_results:
            # Use the aligned shuffled magnitudes (same frame positions as original)
            shuffled_magnitudes = shuffle_result['shuffled_gradient_magnitudes']
            
            # Ensure we have matching lengths for comparison
            min_length = min(len(original_gradient_magnitudes), len(shuffled_magnitudes))
            if min_length > 0:
                # Compute per-frame ratios (avoid division by zero)
                # This compares the same frame's gradient in original vs shuffled context
                eps = 1e-8
                ratios = shuffled_magnitudes[:min_length] / (original_gradient_magnitudes[:min_length] + eps)
                all_ratios.append(ratios)
                
                # Compute optical flow correlations
                if len(motion_magnitudes) >= min_length - 1:
                    # Original correlation
                    flow_corr_orig, _ = pearsonr(motion_magnitudes[:min_length-1], original_gradient_magnitudes[1:min_length])
                    all_flow_correlations_original.append(flow_corr_orig)
                    
                    # Shuffled correlation
                    flow_corr_shuf, _ = pearsonr(motion_magnitudes[:min_length-1], shuffled_magnitudes[1:min_length])
                    all_flow_correlations_shuffled.append(flow_corr_shuf)
            else:
                print(f"   Warning: No valid gradients for comparison in shuffle")
        
        # Aggregate results
        all_ratios = np.array(all_ratios)  # Shape: (num_shuffles, num_frames)
        
        # Compute motion reliance metrics
        motion_reliance_ratio = np.median(all_ratios)  # Lower = more motion reliance
        relative_drop = 1.0 - motion_reliance_ratio
        
        # Debug output
        print(f"   Debug - Original gradient magnitudes: {np.mean(original_gradient_magnitudes):.6f}")
        print(f"   Debug - Shuffled gradient magnitudes (mean across shuffles): {np.mean([np.mean(sr['shuffled_gradient_magnitudes']) for sr in shuffled_results]):.6f}")
        print(f"   Debug - Motion reliance ratio: {motion_reliance_ratio:.6f}")
        print(f"   Debug - All ratios: {all_ratios.flatten()}")
        print(f"   Debug - Interpretation: ratio = shuffled_gradient / original_gradient for same frame")
        
        # Flow correlation analysis
        if all_flow_correlations_original and all_flow_correlations_shuffled:
            mean_flow_corr_original = np.mean(all_flow_correlations_original)
            mean_flow_corr_shuffled = np.mean(all_flow_correlations_shuffled)
            flow_correlation_drop = mean_flow_corr_original - mean_flow_corr_shuffled
        else:
            mean_flow_corr_original = np.nan
            mean_flow_corr_shuffled = np.nan
            flow_correlation_drop = np.nan
        
        # 6. Create static baseline (repeat first frame)
        print("5. Computing static baseline...")
        static_frames = np.repeat(frames_to_process[0:1], len(frames_to_process), axis=0)
        static_gradients = self.extract_temporal_gradients_gpu(static_frames, f"{video_id}_static")
        
        if static_gradients is not None:
            static_gradient_magnitudes = self.compute_gradient_magnitudes(static_gradients)
            static_ratio = np.mean(static_gradient_magnitudes) / (np.mean(original_gradient_magnitudes) + eps)
        else:
            static_gradient_magnitudes = None
            static_ratio = np.nan
        
        # 7. Compile results
        results = {
            'video_id': video_id,
            'num_frames': len(frames_to_process),
            'original_gradients': original_gradients,
            'original_gradient_magnitudes': original_gradient_magnitudes,
            'motion_magnitudes': motion_magnitudes,
            'shuffled_results': shuffled_results,
            'motion_reliance_ratio': motion_reliance_ratio,
            'relative_drop': relative_drop,
            'flow_correlation_original': mean_flow_corr_original,
            'flow_correlation_shuffled': mean_flow_corr_shuffled,
            'flow_correlation_drop': flow_correlation_drop,
            'static_gradient_magnitudes': static_gradient_magnitudes,
            'static_ratio': static_ratio,
            'all_ratios': all_ratios,
            'interpretation': self.interpret_results(motion_reliance_ratio, relative_drop, flow_correlation_drop)
        }
        
        print("6. Analysis complete!")
        print(f"   Motion reliance ratio: {motion_reliance_ratio:.3f}")
        print(f"   Relative drop: {relative_drop:.3f}")
        print(f"   Flow correlation drop: {flow_correlation_drop:.3f}")
        print(f"   Interpretation: {results['interpretation']}")
        
        return results
    
    def interpret_results(self, motion_reliance_ratio, relative_drop, flow_correlation_drop):
        """Interpret the motion reliance results"""
        interpretation = []
        
        # Motion reliance interpretation
        # motion_reliance_ratio = shuffled_gradients / original_gradients
        # If ratio is close to 1.0, shuffled gradients are similar to original = appearance reliance
        # If ratio is much lower, shuffled gradients are weaker = motion reliance
        if motion_reliance_ratio > 0.9:
            interpretation.append("Very high appearance reliance - model primarily uses visual features")
        elif motion_reliance_ratio > 0.7:
            interpretation.append("High appearance reliance - model primarily uses visual features")
        elif motion_reliance_ratio < 0.3:
            interpretation.append("High motion reliance - model heavily depends on temporal order")
        elif motion_reliance_ratio < 0.5:
            interpretation.append("Moderate motion reliance - model depends on temporal order")
        else:
            interpretation.append("Mixed reliance - model uses both appearance and motion")
        
        # Relative drop interpretation
        if relative_drop > 0.3:
            interpretation.append("Strong temporal context dependence")
        elif relative_drop < 0.1:
            interpretation.append("Weak temporal context dependence")
        else:
            interpretation.append("Moderate temporal context dependence")
        
        # Flow correlation interpretation
        if not np.isnan(flow_correlation_drop):
            if flow_correlation_drop > 0.2:
                interpretation.append("Gradients strongly track motion patterns")
            elif flow_correlation_drop < 0.05:
                interpretation.append("Gradients weakly track motion patterns")
            else:
                interpretation.append("Gradients moderately track motion patterns")
        
        return "; ".join(interpretation)
    
    def analyze_video(self, video_info):
        """Analyze a single video using multiple overlapping windows"""
        video_id = video_info.get('id', 'unknown')
        
        # Get video name from file path if available
        video_name = "Unknown"
        if 'file_names' in video_info and video_info['file_names']:
            video_path = Path(video_info['file_names'][0])
            video_name = video_path.parent.name  # Get the directory name as video name
        
        print(f"\nAnalyzing video {video_id} ({video_name}) with multiple windows...")
        
        # Get video path
        if 'file_names' in video_info and video_info['file_names']:
            video_path = Path(self.config.video_data_root) / Path(video_info['file_names'][0]).parent
        else:
            print(f"No file_names found for video {video_id}")
            return None
        
        # Load all video frames (no max_frames limit)
        all_video_frames = self.load_video_frames(str(video_path))
        
        if all_video_frames is None:
            print(f"Failed to load frames for video {video_id}")
            return None
        
        print(f"Loaded {len(all_video_frames)} total frames")
        
        # Process multiple overlapping windows
        window_size = self.config.max_frames_per_video  # 20 frames
        window_stride = window_size // 2  # 50% overlap between windows
        
        all_results = []
        
        for window_idx in range(0, len(all_video_frames) - window_size + 1, window_stride):
            start_idx = window_idx
            end_idx = start_idx + window_size
            window_frames = all_video_frames[start_idx:end_idx]
            
            print(f"Processing {video_name} - window {window_idx//window_stride + 1}: frames {start_idx}-{end_idx-1}")
            
            # Analyze this window
            try:
                window_result = self.analyze_appearance_vs_motion(window_frames, f"{video_id}_window_{window_idx//window_stride + 1}")
                
                if window_result:
                    window_result['window_info'] = {
                        'window_idx': window_idx // window_stride + 1,
                        'start_frame': start_idx,
                        'end_frame': end_idx - 1,
                        'frame_indices': list(range(start_idx, end_idx))
                    }
                    all_results.append(window_result)
                else:
                    print(f"Failed to analyze window {window_idx//window_stride + 1}")
                    
            except Exception as e:
                print(f"Error analyzing window {window_idx//window_stride + 1}: {e}")
                # Clear memory and continue
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        if all_results:
            # Aggregate results across windows
            aggregated_result = self.aggregate_window_results(all_results, video_id)
            
            # Save results
            output_file = self.output_dir / f"frame_shuffling_analysis_{video_id}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(aggregated_result, f)
            
            print(f"Frame shuffling analysis saved to {output_file}")
            
            return aggregated_result
        else:
            print(f"Failed to analyze video {video_id}")
            return None
    
    def aggregate_window_results(self, window_results, video_id):
        """Aggregate results from multiple windows"""
        print(f"Aggregating results from {len(window_results)} windows...")
        
        # Extract metrics from each window
        motion_reliance_ratios = []
        relative_drops = []
        flow_correlation_drops = []
        static_ratios = []
        
        for result in window_results:
            motion_reliance_ratios.append(result['motion_reliance_ratio'])
            relative_drops.append(result['relative_drop'])
            flow_correlation_drops.append(result['flow_correlation_drop'])
            if 'static_ratio' in result and not np.isnan(result['static_ratio']):
                static_ratios.append(result['static_ratio'])
        
        # Compute aggregated metrics
        aggregated_result = {
            'video_id': video_id,
            'num_windows': len(window_results),
            'window_results': window_results,
            'aggregated_metrics': {
                'mean_motion_reliance_ratio': np.mean(motion_reliance_ratios),
                'std_motion_reliance_ratio': np.std(motion_reliance_ratios),
                'mean_relative_drop': np.mean(relative_drops),
                'std_relative_drop': np.std(relative_drops),
                'mean_flow_correlation_drop': np.mean(flow_correlation_drops),
                'std_flow_correlation_drop': np.std(flow_correlation_drops),
                'mean_static_ratio': np.mean(static_ratios) if static_ratios else np.nan,
                'std_static_ratio': np.std(static_ratios) if static_ratios else np.nan
            },
            'window_metrics': {
                'motion_reliance_ratios': motion_reliance_ratios,
                'relative_drops': relative_drops,
                'flow_correlation_drops': flow_correlation_drops,
                'static_ratios': static_ratios
            }
        }
        
        # Generate aggregated interpretation
        mean_motion_ratio = aggregated_result['aggregated_metrics']['mean_motion_reliance_ratio']
        mean_flow_drop = aggregated_result['aggregated_metrics']['mean_flow_correlation_drop']
        
        if mean_motion_ratio < 0.5:
            motion_reliance = "High motion reliance"
        elif mean_motion_ratio < 0.8:
            motion_reliance = "Moderate motion reliance"
        else:
            motion_reliance = "High appearance reliance"
        
        if mean_flow_drop > 0.3:
            temporal_context = "Strong temporal context dependence"
        elif mean_flow_drop > 0.1:
            temporal_context = "Moderate temporal context dependence"
        else:
            temporal_context = "Weak temporal context dependence"
        
        aggregated_result['interpretation'] = f"{motion_reliance} - model uses {'motion' if mean_motion_ratio < 0.8 else 'appearance'} features; {temporal_context}; Gradients {'strongly' if mean_flow_drop > 0.3 else 'moderately' if mean_flow_drop > 0.1 else 'weakly'} track motion patterns"
        
        return aggregated_result
    
    def run_analysis(self):
        """Run the complete frame shuffling analysis"""
        print("Starting frame shuffling analysis...")
        
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
        
        # Save summary results
        summary_file = self.output_dir / "frame_shuffling_summary.json"
        
        # Convert results to JSON-serializable format
        summary_results = []
        for result in results:
            # Handle both single window and multi-window results
            if 'aggregated_metrics' in result:
                # Multi-window aggregated result
                summary_result = {
                    'video_id': result['video_id'],
                    'num_windows': result['num_windows'],
                    'mean_motion_reliance_ratio': float(result['aggregated_metrics']['mean_motion_reliance_ratio']),
                    'std_motion_reliance_ratio': float(result['aggregated_metrics']['std_motion_reliance_ratio']),
                    'mean_relative_drop': float(result['aggregated_metrics']['mean_relative_drop']),
                    'std_relative_drop': float(result['aggregated_metrics']['std_relative_drop']),
                    'mean_flow_correlation_drop': float(result['aggregated_metrics']['mean_flow_correlation_drop']),
                    'std_flow_correlation_drop': float(result['aggregated_metrics']['std_flow_correlation_drop']),
                    'mean_static_ratio': float(result['aggregated_metrics']['mean_static_ratio']) if not np.isnan(result['aggregated_metrics']['mean_static_ratio']) else None,
                    'std_static_ratio': float(result['aggregated_metrics']['std_static_ratio']) if not np.isnan(result['aggregated_metrics']['std_static_ratio']) else None,
                    'interpretation': result['interpretation']
                }
            else:
                # Single window result (fallback)
                summary_result = {
                    'video_id': result['video_id'],
                    'num_windows': 1,
                    'mean_motion_reliance_ratio': float(result['motion_reliance_ratio']),
                    'std_motion_reliance_ratio': 0.0,
                    'mean_relative_drop': float(result['relative_drop']),
                    'std_relative_drop': 0.0,
                    'mean_flow_correlation_drop': float(result['flow_correlation_drop']) if not np.isnan(result['flow_correlation_drop']) else None,
                    'std_flow_correlation_drop': 0.0,
                    'mean_static_ratio': float(result['static_ratio']) if not np.isnan(result['static_ratio']) else None,
                    'std_static_ratio': 0.0,
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
    config.max_videos_per_species = 15  # Test with 1 video
    config.max_frames_per_video = 12   # Use 12-frame windows for better memory efficiency
    
    # Create analyzer
    analyzer = FrameShufflingAnalyzer(config)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results:
        print("\nFrame Shuffling Analysis Results:")
        for result in results:
            print(f"\nVideo {result['video_id']}:")
            if 'aggregated_metrics' in result:
                # Multi-window results
                print(f"  Number of windows: {result['num_windows']}")
                print(f"  Mean motion reliance ratio: {result['aggregated_metrics']['mean_motion_reliance_ratio']:.3f} ± {result['aggregated_metrics']['std_motion_reliance_ratio']:.3f}")
                print(f"  Mean relative drop: {result['aggregated_metrics']['mean_relative_drop']:.3f} ± {result['aggregated_metrics']['std_relative_drop']:.3f}")
                print(f"  Mean flow correlation drop: {result['aggregated_metrics']['mean_flow_correlation_drop']:.3f} ± {result['aggregated_metrics']['std_flow_correlation_drop']:.3f}")
                if not np.isnan(result['aggregated_metrics']['mean_static_ratio']):
                    print(f"  Mean static ratio: {result['aggregated_metrics']['mean_static_ratio']:.3f} ± {result['aggregated_metrics']['std_static_ratio']:.3f}")
                print(f"  Interpretation: {result['interpretation']}")
            else:
                # Single window results (fallback)
                print(f"  Motion reliance ratio: {result['motion_reliance_ratio']:.3f}")
                print(f"  Relative drop: {result['relative_drop']:.3f}")
                print(f"  Flow correlation drop: {result['flow_correlation_drop']:.3f}")
                print(f"  Static ratio: {result['static_ratio']:.3f}")
                print(f"  Interpretation: {result['interpretation']}")
    else:
        print("No results obtained from analysis")

if __name__ == "__main__":
    main()
