#!/usr/bin/env python3
"""
Patched Gradient Analyzer for DVIS-DAQ Temporal Analysis
Uses the patched model to extract true temporal gradients on CPU
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

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from memory_efficient_config import create_memory_efficient_config
from utils.model_loader import DVISModelLoader

class PatchedGradientAnalyzer:
    """Extract temporal gradients from patched DVIS-DAQ model"""
    
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
    
    def extract_temporal_gradients(self, video_frames, video_id):
        """Extract temporal gradients from the patched DVIS-DAQ model"""
        print(f"Extracting temporal gradients for video {video_id}...")
        
        try:
            # Ensure frames are on CPU
            if isinstance(video_frames, torch.Tensor):
                video_frames = video_frames.cpu().numpy()
            
            num_frames = len(video_frames)
            if num_frames < 2:
                print(f"Not enough frames for temporal analysis: {num_frames}")
                return None
            
            # Convert frames to the format expected by the model
            frame_list = []
            for t in range(num_frames):
                frame = video_frames[t]  # (H, W, C) normalized
                frame = (frame * 255).astype(np.uint8)  # Convert to uint8
                frame_tensor = torch.as_tensor(np.ascontiguousarray(frame.transpose(2, 0, 1)))  # (C, H, W)
                frame_list.append(frame_tensor)
            
                                            # Forward pass with gradient computation
            with torch.enable_grad():
                # Ensure all frames require gradients and are on CPU
                for i, frame in enumerate(frame_list):
                    frame_float = frame.to("cpu", dtype=torch.float32)
                    frame_float.requires_grad_(True)
                    frame_list[i] = frame_float
                
                # Stack frames for model input
                images_tensor = torch.stack(frame_list, dim=0)  # (T, C, H, W)
                
                # Verify tensor is on CPU
                if images_tensor.device.type != 'cpu':
                    images_tensor = images_tensor.cpu()
                
                # Debug: Check gradient requirements
                print(f"Input tensor requires_grad: {images_tensor.requires_grad}")
                print(f"Input tensor grad_fn: {images_tensor.grad_fn}")
                
                # Set model to eval mode
                self.model.eval()
                
                # Forward pass through the patched model
                try:
                    # Use the DVIS-DAQ common_inference method
                    self.model.keep = False
                    
                    # Bypass the no_grad context by calling the internal methods directly
                    # This preserves gradients for temporal analysis
                    print("Using gradient-preserving inference...")
                    
                    # Call backbone and segmentation head directly to preserve gradients
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
                        'padding_masks': torch.zeros(1, out['pred_logits'].shape[1], len(images_tensor), dtype=torch.bool, device='cpu'),
                        'seq_id_list': [],
                        'dead_seq_id_list': []
                    }
                    
                    # Debug: Check if outputs are connected to input
                    print(f"Outputs require_grad: {getattr(outputs, 'requires_grad', 'N/A (dict)')}")
                    if isinstance(outputs, dict):
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor):
                                print(f"  {key} requires_grad: {value.requires_grad}, grad_fn: {value.grad_fn}")
                    else:
                        print(f"Outputs grad_fn: {getattr(outputs, 'grad_fn', 'N/A')}")
                    
                    # Debug: Print output structure
                    print(f"Model outputs type: {type(outputs)}")
                    if isinstance(outputs, dict):
                        print(f"Model outputs keys: {list(outputs.keys())}")
                        for key, value in outputs.items():
                            print(f"  {key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
                        
                        # Debug online_out specifically
                        if 'online_out' in outputs:
                            online_out = outputs['online_out']
                            print(f"  online_out keys: {list(online_out.keys())}")
                            for key, value in online_out.items():
                                print(f"    {key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
                    else:
                        print(f"Model outputs shape: {getattr(outputs, 'shape', 'N/A')}")
                    
                    # Create a loss for gradient computation
                    # Use the model outputs to create a meaningful loss
                    if isinstance(outputs, dict):
                        # Try to use online_out first (this contains the actual predictions)
                        if 'online_out' in outputs:
                            online_out = outputs['online_out']
                            if 'pred_logits' in online_out:
                                pred_logits = online_out['pred_logits']
                                if isinstance(pred_logits, list):
                                    pred_logits = pred_logits[0]
                                loss = pred_logits.sum()
                                print(f"Using pred_logits from online_out, shape: {pred_logits.shape}")
                            elif 'pred_masks' in online_out:
                                pred_masks = online_out['pred_masks']
                                if isinstance(pred_masks, list):
                                    pred_masks = pred_masks[0]
                                loss = pred_masks.sum()
                                print(f"Using pred_masks from online_out, shape: {pred_masks.shape}")
                            else:
                                # Use the first available output from online_out
                                first_output = list(online_out.values())[0]
                                if isinstance(first_output, list):
                                    first_output = first_output[0]
                                loss = first_output.sum()
                                print(f"Using first output from online_out, shape: {first_output.shape}")
                        elif 'pred_logits' in outputs:
                            # Use classification logits
                            pred_logits = outputs['pred_logits']
                            if isinstance(pred_logits, list):
                                pred_logits = pred_logits[0]  # Take first element if it's a list
                            loss = pred_logits.sum()
                        elif 'pred_masks' in outputs:
                            # Use mask predictions
                            pred_masks = outputs['pred_masks']
                            if isinstance(pred_masks, list):
                                pred_masks = pred_masks[0]  # Take first element if it's a list
                            loss = pred_masks.sum()
                        else:
                            # Use the first available output
                            first_output = list(outputs.values())[0]
                            if isinstance(first_output, list):
                                first_output = first_output[0]
                            loss = first_output.sum()
                    else:
                        # If outputs is a tensor, use it directly
                        loss = outputs.sum()
                    
                    # Backward pass to compute gradients
                    loss.backward()
                    
                    # Extract gradients for each frame
                    temporal_gradients = []
                    gradients_found = False
                    for i, frame in enumerate(frame_list):
                        if frame.grad is not None:
                            # Compute gradient magnitude across channels
                            grad_magnitude = torch.norm(frame.grad, dim=0)  # (H, W)
                            temporal_gradients.append(grad_magnitude.detach().cpu().numpy())
                            gradients_found = True
                        else:
                            # No gradients, use zeros
                            temporal_gradients.append(np.zeros((video_frames.shape[1], video_frames.shape[2])))
                    
                    # Stack gradients
                    temporal_gradients = np.array(temporal_gradients)
                    print(f"Temporal gradients shape: {temporal_gradients.shape}")
                    
                    if gradients_found:
                        print("✅ Successfully extracted gradients from model forward pass")
                        return temporal_gradients
                    else:
                        print("❌ No gradients found in model forward pass")
                        print("This indicates the model outputs are not properly connected to the input")
                        return None
                    
                except Exception as e:
                    print(f"Error in model forward pass: {e}")
                    print("This indicates the model outputs are not properly connected to the input")
                    return None
                        
        except Exception as e:
            print(f"Error extracting temporal gradients: {e}")
            return None
    
    def _extract_gradients_alternative(self, frame_list, video_frames):
        """Alternative gradient extraction method"""
        print("Using alternative gradient extraction method...")
        
        temporal_gradients = []
        
        for i, frame in enumerate(frame_list):
            try:
                # Create a simple loss based on frame features
                frame_float = frame.to("cpu", dtype=torch.float32)
                frame_float.requires_grad_(True)
                
                # Try to pass through the first layer of the model
                try:
                    # Get the first layer (backbone)
                    if hasattr(self.model, 'backbone'):
                        first_layer = self.model.backbone
                        # Forward pass through backbone
                        output = first_layer(frame_float.unsqueeze(0))  # Add batch dimension
                        
                        # Handle backbone output (usually a dict)
                        if isinstance(output, dict):
                            # Use one of the backbone outputs
                            if 'res2' in output:
                                loss = output['res2'].sum()
                            elif 'res3' in output:
                                loss = output['res3'].sum()
                            elif 'res4' in output:
                                loss = output['res4'].sum()
                            elif 'res5' in output:
                                loss = output['res5'].sum()
                            else:
                                # Use the first available output
                                first_output = list(output.values())[0]
                                loss = first_output.sum()
                        else:
                            # Direct tensor output
                            loss = output.sum()
                        
                        loss.backward()
                    else:
                        # Fallback to simple loss
                        loss = frame_float.sum()
                        loss.backward()
                    
                    if frame_float.grad is not None:
                        grad_magnitude = torch.norm(frame_float.grad, dim=0)
                        temporal_gradients.append(grad_magnitude.detach().cpu().numpy())
                    else:
                        temporal_gradients.append(np.zeros((video_frames.shape[1], video_frames.shape[2])))
                        
                except Exception as e:
                    print(f"Error in first layer forward pass: {e}")
                    # Fallback to simple loss
                    loss = frame_float.sum()
                    loss.backward()
                    
                    if frame_float.grad is not None:
                        grad_magnitude = torch.norm(frame_float.grad, dim=0)
                        temporal_gradients.append(grad_magnitude.detach().cpu().numpy())
                    else:
                        temporal_gradients.append(np.zeros((video_frames.shape[1], video_frames.shape[2])))
                    
            except Exception as e:
                print(f"Error in alternative gradient extraction for frame {i}: {e}")
                temporal_gradients.append(np.zeros((video_frames.shape[1], video_frames.shape[2])))
        
        temporal_gradients = np.array(temporal_gradients)
        print(f"Alternative gradients shape: {temporal_gradients.shape}")
        
        return temporal_gradients
    
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
        
        # Extract temporal gradients with memory management
        try:
            temporal_gradients = self.extract_temporal_gradients(video_frames, video_id)
        except Exception as e:
            print(f"Error extracting gradients: {e}")
            # Clear memory and try with smaller chunk
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try with first 10 frames only
            print("Trying with reduced frame count...")
            video_frames_reduced = video_frames[:10]
            temporal_gradients = self.extract_temporal_gradients(video_frames_reduced, video_id)
        
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
            **gradient_stats
        }
        
        # Save gradients to file
        output_file = self.output_dir / f"temporal_gradients_{video_id}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(temporal_gradients, f)
        
        print(f"Temporal gradients saved to {output_file}")
        print(f"Gradient stats: mean={gradient_stats['mean_gradient']:.6f}, "
              f"max={gradient_stats['max_gradient']:.6f}, "
              f"std={gradient_stats['std_gradient']:.6f}")
        
        return result
    
    def run_analysis(self):
        """Run the complete analysis"""
        print("Starting patched gradient analysis...")
        
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
        summary_file = self.output_dir / "patched_gradient_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to {summary_file}")
        print(f"Processed {len(results)} videos successfully")
        
        return results

def main():
    """Main function"""
    # Create configuration
    config = create_memory_efficient_config(gpu_memory_gb=8)
    config.max_videos_per_species = 1  # Test with just 1 video
    config.max_frames_per_video = 5    # Start with 5 frames
    
    # Create analyzer
    analyzer = PatchedGradientAnalyzer(config)
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if results:
        print("\nPatched Gradient Analysis Results:")
        for result in results:
            print(f"Video {result['video_id']}: ")
            print(f"  Mean gradient: {result['mean_gradient']:.6f}")
            print(f"  Max gradient: {result['max_gradient']:.6f}")
            print(f"  Std gradient: {result['std_gradient']:.6f}")
            print(f"  Min gradient: {result['min_gradient']:.6f}")
            print(f"  Gradient range: {result['gradient_range']:.6f}")

if __name__ == "__main__":
    main()
