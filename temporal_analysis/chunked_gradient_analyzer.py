#!/usr/bin/env python3
"""
Chunked Gradient Analyzer for DVIS-DAQ Temporal Analysis
Processes 31-frame windows in smaller chunks while preserving full temporal context
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

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from memory_efficient_config import create_memory_efficient_config
from utils.model_loader import DVISModelLoader

class ChunkedGradientAnalyzer:
    """Extract temporal gradients using chunked processing for 31-frame windows"""
    
    def __init__(self, config):
        self.config = config
        self.config.device = "cpu"
        self.config.force_cpu_offload = True
        
        # Chunking settings
        self.window_size = 31  # Full model window size
        self.chunk_size = 6    # Process 6 frames at a time
        self.chunk_overlap = 2 # Overlap between chunks
        
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
            print(f"No valid frames loaded from {video_path}")
            return None
        
        return np.array(frames)
    
    def preprocess_frames(self, frames):
        """Preprocess frames for model input"""
        processed_frames = []
        
        for frame in frames:
            # Normalize to [0, 1]
            frame_normalized = frame.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
            
            # Move to device
            frame_tensor = frame_tensor.to(self.config.device)
            frame_tensor.requires_grad_(True)
            
            processed_frames.append(frame_tensor)
        
        return processed_frames
    
    def extract_chunk_gradients(self, chunk_frames, chunk_idx):
        """Extract gradients for a single chunk"""
        try:
            # Create input tensor for chunk
            images_tensor = torch.cat(chunk_frames, dim=0)
            
            print(f"Processing chunk {chunk_idx}: {len(chunk_frames)} frames")
            print(f"Input tensor requires_grad: {images_tensor.requires_grad}")
            print(f"Input tensor grad_fn: {images_tensor.grad_fn}")
            
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
            
            # Extract gradients for each frame in chunk
            chunk_gradients = []
            gradients_found = False
            
            for frame in chunk_frames:
                if frame.grad is not None:
                    # Compute gradient magnitude across channels
                    grad_magnitude = torch.norm(frame.grad, dim=0)  # (H, W)
                    chunk_gradients.append(grad_magnitude.detach().cpu().numpy())
                    gradients_found = True
                else:
                    # No gradients, use zeros
                    chunk_gradients.append(np.zeros((frame.shape[1], frame.shape[2])))
            
            if gradients_found:
                print(f"✅ Successfully extracted gradients for chunk {chunk_idx}")
                return np.array(chunk_gradients)
            else:
                print(f"❌ No gradients found for chunk {chunk_idx}")
                return None
                
        except Exception as e:
            print(f"Error extracting gradients for chunk {chunk_idx}: {e}")
            return None
    
    def extract_temporal_gradients_chunked(self, video_frames, video_id):
        """Extract temporal gradients using chunked processing"""
        print(f"Extracting temporal gradients for video {video_id} using chunked processing...")
        print(f"Video frames: {len(video_frames)}, Window size: {self.window_size}, Chunk size: {self.chunk_size}")
        
        # Ensure we have enough frames
        if len(video_frames) < self.window_size:
            print(f"Warning: Only {len(video_frames)} frames available, using all frames")
            frames_to_process = video_frames
        else:
            frames_to_process = video_frames[:self.window_size]
        
        # Preprocess all frames
        print("Preprocessing frames...")
        processed_frames = self.preprocess_frames(frames_to_process)
        
        # Calculate chunk boundaries
        num_chunks = (len(processed_frames) - self.chunk_overlap) // (self.chunk_size - self.chunk_overlap)
        if num_chunks < 1:
            num_chunks = 1
        
        print(f"Processing {len(processed_frames)} frames in {num_chunks} chunks")
        
        # Process chunks and collect gradients
        all_gradients = []
        
        for chunk_idx in range(num_chunks):
            # Calculate chunk boundaries
            start_idx = chunk_idx * (self.chunk_size - self.chunk_overlap)
            end_idx = min(start_idx + self.chunk_size, len(processed_frames))
            
            # Extract chunk frames
            chunk_frames = processed_frames[start_idx:end_idx]
            
            # Extract gradients for this chunk
            chunk_gradients = self.extract_chunk_gradients(chunk_frames, chunk_idx)
            
            if chunk_gradients is not None:
                # Add chunk gradients to results (handle overlap)
                if chunk_idx == 0:
                    # First chunk: add all gradients
                    all_gradients.extend(chunk_gradients)
                else:
                    # Subsequent chunks: add only non-overlapping gradients
                    overlap_size = self.chunk_overlap
                    non_overlap_gradients = chunk_gradients[overlap_size:]
                    all_gradients.extend(non_overlap_gradients)
                
                print(f"Chunk {chunk_idx}: Added {len(chunk_gradients)} gradients")
            else:
                print(f"Chunk {chunk_idx}: Failed to extract gradients")
            
            # Clear gradients for next chunk
            for frame in chunk_frames:
                if frame.grad is not None:
                    frame.grad.zero_()
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Convert to numpy array
        if all_gradients:
            temporal_gradients = np.array(all_gradients)
            print(f"Final temporal gradients shape: {temporal_gradients.shape}")
            return temporal_gradients
        else:
            print("No gradients extracted from any chunk")
            return None
    
    def analyze_video(self, video_info):
        """Analyze a single video using chunked processing"""
        video_id = video_info.get('id', 'unknown')
        print(f"\nAnalyzing video {video_id}...")
        
        # Get video path
        if 'file_names' in video_info and video_info['file_names']:
            video_path = Path(self.config.video_data_root) / Path(video_info['file_names'][0]).parent
        else:
            print(f"No file_names found for video {video_id}")
            return None
        
        # Load video frames (load more than window size to ensure we have enough)
        video_frames = self.load_video_frames(
            str(video_path), 
            max_frames=max(self.window_size, self.config.max_frames_per_video)
        )
        
        if video_frames is None:
            print(f"Failed to load frames for video {video_id}")
            return None
        
        print(f"Loaded {len(video_frames)} frames")
        
        # Extract temporal gradients using chunked processing
        temporal_gradients = self.extract_temporal_gradients_chunked(video_frames, video_id)
        
        if temporal_gradients is None:
            print(f"Failed to extract temporal gradients for video {video_id}")
            return None
        
        # Calculate gradient statistics
        gradient_stats = {
            'mean_gradient': float(np.mean(temporal_gradients)),
            'max_gradient': float(np.max(temporal_gradients)),
            'std_gradient': float(np.std(temporal_gradients)),
            'min_gradient': float(np.min(temporal_gradients)),
            'gradient_range': float(np.max(temporal_gradients) - np.min(temporal_gradients))
        }
        
        # Save results
        result = {
            'video_id': video_id,
            'num_frames': len(video_frames),
            'temporal_gradients_shape': temporal_gradients.shape,
            'window_size': self.window_size,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            **gradient_stats
        }
        
        # Save gradients to file
        output_file = self.output_dir / f"chunked_temporal_gradients_{video_id}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(temporal_gradients, f)
        
        print(f"Temporal gradients saved to {output_file}")
        print(f"Gradient stats: mean={gradient_stats['mean_gradient']:.6f}, "
              f"max={gradient_stats['max_gradient']:.6f}, "
              f"std={gradient_stats['std_gradient']:.6f}")
        
        return result

def main():
    """Main function"""
    # Create configuration
    config = create_memory_efficient_config()
    
    # Initialize analyzer
    analyzer = ChunkedGradientAnalyzer(config)
    
    # Test with one video
    test_video = {
        'id': 1,
        'file_names': ['Credit__2023__05122023-05232023__23  05  12  10  40__4/00000.jpg']
    }
    
    result = analyzer.analyze_video(test_video)
    
    if result:
        print("✅ Chunked analysis successful!")
        print(f"Results: {result}")
    else:
        print("❌ Chunked analysis failed")

if __name__ == "__main__":
    main()
