#!/usr/bin/env python3
"""
Plot cross-attention maps for the top-scoring prediction per video_id from results_temporal.json.

This script:
1. Loads results_temporal.json and finds the top-scoring prediction for each video_id
2. Extracts predictor_query_ids for each top prediction
3. Loads the corresponding cross-attention maps
4. Maps attention weights from feature space to image coordinates
5. Overlays attention as heatmaps on individual video frames
6. Saves the visualization results organized by video_id and layer
"""

import os
import sys
import json
import argparse
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2

# Add project paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DVIS_PLUS_ROOT = os.path.join(_PROJECT_ROOT, "DVIS_Plus")
for p in (_PROJECT_ROOT, _DVIS_PLUS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_results(results_path: str) -> List[Dict]:
    """Load results_temporal.json."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def find_top_prediction(results: List[Dict]) -> Optional[Dict]:
    """Find the top-scoring prediction."""
    if not results:
        return None
    return max(results, key=lambda x: x.get('score', 0.0))


def find_top_prediction_per_video(results: List[Dict]) -> Dict[int, Dict]:
    """
    Find the top-scoring prediction for each video_id.
    
    Args:
        results: List of prediction dictionaries
    
    Returns:
        Dictionary mapping video_id to top prediction dict for that video
    """
    top_per_video = {}
    
    for pred in results:
        video_id = pred.get('video_id')
        if video_id is None:
            continue
        
        score = pred.get('score', 0.0)
        
        # If we haven't seen this video_id yet, or this prediction has a higher score
        if video_id not in top_per_video or score > top_per_video[video_id].get('score', 0.0):
            top_per_video[video_id] = pred
    
    return top_per_video


def load_query_ids_from_tracker_attention(tracker_attention_json_path: str, video_id: int, query_choice: str = "top"):
    """
    Load query IDs or attention weights from tracker_attention_top5.json.
    
    Args:
        tracker_attention_json_path: Path to tracker_attention_top5.json file
        video_id: Video ID to extract query IDs for
        query_choice: Method to choose query ID - "top" (use layer 5's top index) or "weighted_sum" (return attention weights)
    
    Returns:
        For "top": Dictionary mapping frame numbers (as strings) to query IDs (int)
        For "weighted_sum": Dictionary mapping frame numbers (as strings) to attention weights arrays (np.ndarray)
    """
    if not os.path.exists(tracker_attention_json_path):
        raise FileNotFoundError(f"Tracker attention JSON not found: {tracker_attention_json_path}")
    
    with open(tracker_attention_json_path, 'r') as f:
        data = json.load(f)
    
    video_id_str = str(video_id)
    if video_id_str not in data:
        raise ValueError(f"Video {video_id} not found in tracker_attention_top5.json")
    
    video_data = data[video_id_str]
    frames_data = video_data.get('frames', {})
    
    query_ids_by_frame = {}
    
    if query_choice == "top":
        # Use layer 5's first index from top5_indices list
        layer_5_key = "5"
        
        for frame_num_str, frame_data in frames_data.items():
            if layer_5_key not in frame_data:
                print(f"Warning: Frame {frame_num_str} missing layer {layer_5_key}, skipping")
                continue
            
            layer_5_data = frame_data[layer_5_key]
            top5_indices = layer_5_data.get('top5_indices')
            
            if top5_indices is None or len(top5_indices) == 0:
                print(f"Warning: Frame {frame_num_str} layer {layer_5_key} missing top5_indices, skipping")
                continue
            
            # Get the first (top) index from the list
            top_index = int(top5_indices[0])
            query_ids_by_frame[frame_num_str] = top_index
    
    elif query_choice == "weighted_sum":
        # Return attention weights from layer 5 for each frame
        layer_5_key = "5"
        
        for frame_num_str, frame_data in frames_data.items():
            if layer_5_key not in frame_data:
                print(f"Warning: Frame {frame_num_str} missing layer {layer_5_key}, skipping")
                continue
            
            layer_5_data = frame_data[layer_5_key]
            attention_weights = layer_5_data.get('attention_weights')
            
            if attention_weights is None or len(attention_weights) == 0:
                print(f"Warning: Frame {frame_num_str} layer {layer_5_key} missing attention_weights, skipping")
                continue
            
            # Convert to numpy array
            query_ids_by_frame[frame_num_str] = np.array(attention_weights, dtype=np.float32)
    
    else:
        raise ValueError(f"Unknown query_choice: {query_choice}. Must be 'top' or 'weighted_sum'")
    
    return query_ids_by_frame


def find_available_attention_windows(attn_dir: str, video_id: int) -> List[Tuple[int, int]]:
    """
    Find available attention map windows by scanning for predictor cross-attention files.
    
    Args:
        attn_dir: Directory containing attention maps
        video_id: Video ID
    
    Returns:
        List of (frame_start, frame_end) tuples for available windows
    """
    import glob
    pattern = os.path.join(attn_dir, f"video_{video_id}_frames*-*_predictor_cross_attn_layer_0*.npz")
    files = glob.glob(pattern)
    
    windows = []
    for f in files:
        # Extract frame range from filename: video_75_frames0-99_predictor_cross_attn_layer_0.npz
        match = re.search(rf'video_{video_id}_frames(\d+)-(\d+)_predictor_cross_attn_layer_0', f)
        if match:
            frame_start = int(match.group(1))
            frame_end = int(match.group(2))
            windows.append((frame_start, frame_end))
    
    # Sort by frame_start
    windows.sort(key=lambda x: x[0])
    return windows


def load_attention_map(attn_dir: str, video_id: int, frame_start: int, frame_end: int, 
                      layer_idx: int, feature_level: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """
    Load attention map from .npz file.
    
    Returns:
        attention_weights: numpy array of shape (num_frames, num_queries, spatial_length)
        metadata: dictionary with metadata
    """
    # Construct filename
    if feature_level is not None:
        filename = f"video_{video_id}_frames{frame_start}-{frame_end}_predictor_cross_attn_layer_{layer_idx}_level_{feature_level}"
    else:
        filename = f"video_{video_id}_frames{frame_start}-{frame_end}_predictor_cross_attn_layer_{layer_idx}"
    
    npz_path = os.path.join(attn_dir, f"{filename}.npz")
    meta_path = os.path.join(attn_dir, f"{filename}.meta.json")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Attention map not found: {npz_path}")
    
    # Load attention weights
    data = np.load(npz_path)
    attention_weights = data['attention_weights']
    
    # Load metadata
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    return attention_weights, metadata


def map_attention_to_image(attention_1d: np.ndarray, H_feat: int, W_feat: int,
                           original_img_height: int, original_img_width: int,
                           feature_stride: int) -> np.ndarray:
    """
    Map 1D attention weights to 2D image coordinates.
    
    Each attention value corresponds to a patch center, not a corner. We map each
    attention value to the center pixel of its corresponding patch in the image.
    
    Args:
        attention_1d: 1D attention weights of shape (H_feat * W_feat,)
        H_feat, W_feat: Feature map dimensions
        original_img_height, original_img_width: Original image dimensions
        feature_stride: Downsampling factor (e.g., 8, 16, or 32)
    
    Returns:
        attention_2d: 2D attention map of shape (original_img_height, original_img_width)
    """
    # Reshape to 2D feature map
    attention_2d_feat = attention_1d.reshape(H_feat, W_feat)
    
    # Each feature position (h, w) represents a patch covering:
    #   Image pixels [h*stride : (h+1)*stride, w*stride : (w+1)*stride]
    # The center of this patch is at ((h+0.5)*stride, (w+0.5)*stride) in image coordinates
    
    # To properly map attention values to patch centers, we need to account for
    # the 0.5 offset. We'll use scipy's interpolation which allows precise coordinate mapping
    try:
        from scipy.ndimage import map_coordinates
        
        # Create coordinate grids for the output image
        y_coords = np.arange(original_img_height, dtype=np.float32)
        x_coords = np.arange(original_img_width, dtype=np.float32)
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Map image coordinates to feature map coordinates
        # Each feature position h represents a patch covering [h*stride : (h+1)*stride]
        # The center of this patch is at image coordinate (h+0.5)*stride
        # 
        # The attention value is stored at integer feature position h, but represents
        # the patch center, which is conceptually at feature coordinate h+0.5.
        # 
        # When mapping image coordinate y to feature coordinate for interpolation:
        # - If y = (h+0.5)*stride (patch center), we want to sample at feature coord h
        #   (where the value is stored)
        # - So: y/stride = h+0.5, therefore feature_coord = y/stride - 0.5 = h
        # - This shifts the coordinate system to account for centers vs corners
        feature_y = (y_grid / feature_stride) - 0.5
        feature_x = (x_grid / feature_stride) - 0.5
        
        # Stack coordinates for map_coordinates (needs shape (2, H, W))
        coordinates = np.stack([feature_y, feature_x])
        
        # Interpolate attention values at patch center coordinates
        # Use 'nearest' mode to extend edge values smoothly to frame boundaries
        # This ensures attention maps extend smoothly to the edges rather than having hard cutoffs
        attention_2d = map_coordinates(
            attention_2d_feat.astype(np.float32),
            coordinates,
            order=1,  # Bilinear interpolation
            mode='nearest',  # Extend edge values to boundaries (smooth extension)
            prefilter=False
        )
        
    except ImportError:
        # Fallback to cv2.resize if scipy is not available
        # This treats positions as corners, not centers, so there will be a slight offset
        # We can partially compensate by padding and adjusting
        import warnings
        warnings.warn("scipy not available, using cv2.resize (may have slight offset from patch centers)")
        
        # Pad the attention map to account for center offset and extend edges smoothly
        # Use 'edge' mode to extend the boundary values, creating smooth extension to frame edges
        padded_attn = np.pad(attention_2d_feat, ((1, 1), (1, 1)), mode='edge')
        
        # Upsample to slightly larger size to account for padding
        upsampled_h = int(original_img_height * (H_feat + 2) / H_feat)
        upsampled_w = int(original_img_width * (W_feat + 2) / W_feat)
        
        attention_upsampled = cv2.resize(
            padded_attn,
            (upsampled_w, upsampled_h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Crop to remove padding offset and get to original size
        # Crop from (1, 1) to account for the padding we added
        crop_y = int((upsampled_h - original_img_height) / 2)
        crop_x = int((upsampled_w - original_img_width) / 2)
        attention_2d = attention_upsampled[crop_y:crop_y+original_img_height, 
                                           crop_x:crop_x+original_img_width]
    
    return attention_2d


def overlay_attention_heatmap(image: np.ndarray, attention_2d: np.ndarray,
                              alpha: float = 0.5, colormap: str = 'jet',
                              attn_min: Optional[float] = None, attn_max: Optional[float] = None) -> np.ndarray:
    """
    Overlay attention heatmap on image.
    
    Args:
        image: Original image as numpy array (H, W, 3) in RGB format
        attention_2d: 2D attention map (H, W) - can be pre-normalized or raw values
        alpha: Transparency factor for heatmap overlay
        colormap: Matplotlib colormap name
        attn_min: Optional minimum value for normalization (if None, uses attention_2d.min())
        attn_max: Optional maximum value for normalization (if None, uses attention_2d.max())
    
    Returns:
        Overlaid image
    """
    # Normalize attention to [0, 1]
    if attn_min is None:
        attn_min = attention_2d.min()
    if attn_max is None:
        attn_max = attention_2d.max()
    
    attn_norm = (attention_2d - attn_min) / (attn_max - attn_min + 1e-8)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap) if hasattr(cm, 'get_cmap') else plt.get_cmap(colormap)
    heatmap = cmap(attn_norm)[:, :, :3]  # Remove alpha channel, shape (H, W, 3)
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Overlay on image
    overlaid = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    
    return overlaid


def find_val_json_in_model_dir(directory):
    """
    Find the val.json file in the model directory.
    
    The model directory is determined by going up from the directory until we find
    a directory containing a val.json file. If the directory path contains "foldX"
    (where X is a number), it will look for val_foldX.json instead.
    
    Args:
        directory: Path to directory (e.g., /path/to/attn_extract_3231)
    
    Returns:
        Path to val.json or val_foldX.json file
        Raises FileNotFoundError if not found or multiple found
    """
    # Go up from directory to find model directory
    current_dir = Path(directory).resolve()
    
    # Check if directory path contains "foldX" pattern
    directory_str = str(directory)
    fold_match = re.search(r'fold(\d+)', directory_str, re.IGNORECASE)
    fold_number = fold_match.group(1) if fold_match else None
    
    # Look for val.json files
    val_json_files = []
    for parent in [current_dir] + list(current_dir.parents):
        # If we found a fold number, try val_foldX.json first
        if fold_number:
            val_fold_path = parent / f"val_fold{fold_number}.json"
            if val_fold_path.exists():
                val_json_files.append(str(val_fold_path))
                continue  # Prefer fold-specific file
        
        # Also check for regular val.json
        val_json_path = parent / "val.json"
        if val_json_path.exists():
            val_json_files.append(str(val_json_path))
    
    if len(val_json_files) == 0:
        if fold_number:
            raise FileNotFoundError(
                f"Could not find val.json or val_fold{fold_number}.json in model directory "
                f"(searched from {directory})"
            )
        else:
            raise FileNotFoundError(f"Could not find val.json in model directory (searched from {directory})")
    elif len(val_json_files) > 1:
        # If we have both val.json and val_foldX.json, prefer the fold-specific one
        fold_specific = [f for f in val_json_files if f'val_fold{fold_number}.json' in f] if fold_number else []
        if fold_specific:
            return fold_specific[0]
        raise ValueError(f"Multiple val.json files found: {val_json_files}. Please specify model directory.")
    
    return val_json_files[0]


def get_video_info_from_val_json(val_json_path, video_id):
    """
    Get video information from val.json file.
    
    Args:
        val_json_path: Path to val.json file
        video_id: Video ID number
    
    Returns:
        dict with video info including 'file_names' list
        Raises KeyError if video not found
    """
    with open(val_json_path, 'r') as f:
        data = json.load(f)
    
    videos = {v['id']: v for v in data['videos']}
    if video_id not in videos:
        raise KeyError(f"Video {video_id} not found in val.json")
    
    return videos[video_id]


def get_image_root_directory(model_dir):
    """
    Determine the image root directory based on model directory name.
    
    Args:
        model_dir: Path to model directory (e.g., /path/to/model_camera_ft123)
    
    Returns:
        Path to image root directory:
        - /home/simone/shared-data/fishway_ytvis/all_videos if model_dir contains "camera"
        - /home/simone/shared-data/fishway_ytvis/all_videos_mask if model_dir contains "silhouette"
        Raises ValueError if neither found
    """
    model_dir_str = str(model_dir)
    if "camera" in model_dir_str:
        return "/home/simone/shared-data/fishway_ytvis/all_videos"
    elif "silhouette" in model_dir_str:
        return "/home/simone/shared-data/fishway_ytvis/all_videos_mask"
    else:
        raise ValueError(f"Could not determine image root directory from model directory: {model_dir}. "
                         f"Model directory must contain 'camera' or 'silhouette'.")


def load_resize_and_pad_image(image_path, original_height, original_width, 
                              pad_top=0, pad_left=0, pad_bottom=0, pad_right=0):
    """
    Load an image, resize it to original dimensions, and apply padding.
    
    Args:
        image_path: Path to image file
        original_height: Target height after resize (before padding)
        original_width: Target width after resize (before padding)
        pad_top, pad_left, pad_bottom, pad_right: Padding values in pixels
    
    Returns:
        PIL Image object (resized and padded)
    """
    # Load image
    img = Image.open(image_path)
    
    # Resize to original dimensions
    img_resized = img.resize((original_width, original_height), Image.Resampling.LANCZOS)
    
    # Apply padding
    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        # Calculate final dimensions
        final_width = original_width + pad_left + pad_right
        final_height = original_height + pad_top + pad_bottom
        
        # Create new image with padding (black background)
        img_padded = Image.new('RGB', (final_width, final_height), (0, 0, 0))
        
        # Paste resized image at the correct position
        img_padded.paste(img_resized, (pad_left, pad_top))
        
        return img_padded
    else:
        return img_resized


def load_video_frames(directory: str, video_id: int, frame_start: int, frame_end: int,
                     original_img_height: int, original_img_width: int,
                     pad_top: int = 0, pad_left: int = 0, pad_bottom: int = 0, pad_right: int = 0,
                     val_json_path: Optional[str] = None) -> List[np.ndarray]:
    """
    Load video frames using the same approach as plot_spatial_attn.py.
    
    Args:
        directory: Path to directory containing 'attention_maps' folder (e.g., /path/to/attn_extract_3231)
        video_id: Video ID
        frame_start: Start frame index
        frame_end: End frame index (inclusive)
        original_img_height, original_img_width: Original image dimensions
        pad_top, pad_left, pad_bottom, pad_right: Padding values
        val_json_path: Optional path to val.json file. If not provided, will search automatically.
    
    Returns:
        List of frames as numpy arrays (H, W, 3) in RGB format
    """
    frames = []
    
    try:
        # Find val.json in model directory if not provided
        if val_json_path is None:
            val_json_path = find_val_json_in_model_dir(directory)
        
        # Get video info from val.json
        video_info = get_video_info_from_val_json(val_json_path, video_id)
        file_names = video_info['file_names']
        
        # Determine image root directory based on model directory
        model_dir = os.path.dirname(val_json_path)
        image_root = get_image_root_directory(model_dir)
        
        # Load frames
        for frame_idx in range(frame_start, min(frame_end + 1, len(file_names))):
            frame_file = file_names[frame_idx]
            frame_path = os.path.join(image_root, frame_file)
            
            if os.path.exists(frame_path):
                # Load, resize, and pad the frame image
                img_processed = load_resize_and_pad_image(
                    frame_path,
                    original_img_height,
                    original_img_width,
                    pad_top=pad_top,
                    pad_left=pad_left,
                    pad_bottom=pad_bottom,
                    pad_right=pad_right
                )
                # Convert PIL Image to numpy array (RGB)
                frame = np.array(img_processed)
                frames.append(frame)
            else:
                print(f"Warning: Frame {frame_idx} image not found: {frame_path}")
                frames.append(None)
    except Exception as e:
        print(f"Warning: Could not load frames: {e}")
        print(f"Note: Will create dummy frames for visualization.")
    
    return frames


def load_temporal_attention_weights(directory: str, video_id: int) -> Optional[np.ndarray]:
    """
    Load temporal attention weights (activation_proj vector) from JSON file.
    
    Args:
        directory: Base directory containing inference/ subdirectory
        video_id: Video ID
    
    Returns:
        Temporal attention weights array of shape (num_frames,) or None if not found
    """
    # Look for activation_proj_top_predictions.json in inference/ subdirectory
    json_path = os.path.join(directory, "inference", "activation_proj_top_predictions.json")
    
    if not os.path.exists(json_path):
        print(f"Warning: activation_proj_top_predictions.json not found at {json_path}")
        return None
    
    try:
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert video_id to string for lookup
        video_id_str = str(video_id)
        
        if video_id_str not in data:
            print(f"Warning: Video {video_id} not found in activation_proj_top_predictions.json")
            return None
        
        video_data = data[video_id_str]
        
        if 'activation_vector' not in video_data:
            print(f"Warning: 'activation_vector' not found for video {video_id}")
            return None
        
        # Convert list to numpy array
        activation_vector = np.array(video_data['activation_vector'])
        
        print(f"Loaded activation_proj vector for video {video_id}: shape {activation_vector.shape}, refiner_id={video_data.get('refiner_id', 'unknown')}")
        
        return activation_vector
        
    except Exception as e:
        print(f"Warning: Could not load activation_proj weights from {json_path}: {e}")
        return None


def plot_attention_for_query(attn_dir: str, output_dir: str, video_id: int,
                            predictor_query_ids_by_frame: Optional[Dict],
                            frame_start: int, frame_end: int,
                            layer_indices: List[int], original_img_height: int,
                            original_img_width: int, frames: List[np.ndarray],
                            size_list: Optional[List[Tuple[int, int]]] = None,
                            output_img_height: Optional[int] = None,
                            output_img_width: Optional[int] = None,
                            query_choice: str = "top",
                            colour_scale: str = "per_frame",
                            temporal_attention_weights: Optional[np.ndarray] = None):
    """
    Plot attention maps for queries across multiple layers, using per-frame query IDs.
    
    Colour scaling modes:
    
    1. "per_frame" (default):
       - Normalize each frame independently: attn_norm = (attn - attn.min()) / (attn.max() - attn.min())
       - Each frame's highest attention value gets deepest red
       - Colour scale is NOT consistent across frames
    
    2. "across_frames":
       - Compute global min/max across ALL frames: global_min = min(all_frames), global_max = max(all_frames)
       - Normalize each frame using global min/max: attn_norm = (attn - global_min) / (global_max - global_min)
       - Red is assigned to the highest attention value across ALL frames
       - Colour scale IS consistent across frames (allows comparison of attention magnitudes)
    
    3. "temporal_per_frame":
       - Step 1: Normalize each frame independently to [0, 1]: attn_norm = (attn - attn.min()) / (attn.max() - attn.min())
       - Step 2: Scale normalized frame by temporal attention weight: attn_scaled = attn_norm * temporal_weight[frame]
       - Step 3: Use global max across all scaled frames as peak value for heatmap (min = 0)
       - Each frame is normalized independently, then modulated by temporal importance
       - Frames with higher temporal attention get amplified
       - Color scale is consistent across frames using global max as peak
    
    4. "temporal_across_frames":
       - Step 1: Scale each frame by temporal attention weight: attn_scaled = attn * temporal_weight[frame]
       - Step 2: Compute global min/max across ALL scaled frames: global_min = min(all_scaled_frames), global_max = max(all_scaled_frames)
       - Step 3: Normalize each scaled frame using global min/max: attn_norm = (attn_scaled - global_min) / (global_max - global_min)
       - Equivalent to: multiply decoder attention by temporal weights, then normalize across all frames
       - Colour scale IS consistent across frames, with temporal weighting applied
    
    Args:
        attn_dir: Directory containing attention maps
        output_dir: Output directory for plots
        video_id: Video ID
        predictor_query_ids_by_frame: Dictionary mapping frame numbers (as strings, starting from '1') to query IDs (int) 
                                      or attention weights (np.ndarray for weighted_sum).
                                      If None, will use a fallback approach (for backward compatibility).
        frame_start: Start frame index
        frame_end: End frame index
        layer_indices: List of layer indices to visualize
        original_img_height, original_img_width: Original image dimensions
        frames: List of video frames
        size_list: Optional list of (H_feat, W_feat) tuples for each feature level [level0, level1, level2]
        query_choice: Method used - "top" or "weighted_sum"
        colour_scale: Colour scaling mode - "per_frame", "across_frames", "temporal_per_frame", or "temporal_across_frames"
        temporal_attention_weights: Optional array of temporal attention weights, shape (num_frames,)
    """
    # Plot for each layer
    for layer_idx in layer_indices:
        # Create layer-specific output directory: output_dir/{colour_scale}/video_{video_id}/layer_{layer_idx}/
        layer_output_dir = os.path.join(output_dir, colour_scale, f"video_{video_id}", f"layer_{layer_idx}")
        os.makedirs(layer_output_dir, exist_ok=True)
        # Determine feature level from layer index (cycles every 3 layers)
        feature_level = layer_idx % 3
        
        # Load attention map for this layer
        # Try both with and without feature_level suffix (different windows may use different naming)
        attention_weights = None
        metadata = None
        load_attempts = []
        
        # Try different filename patterns
        # Pattern 1: With level suffix (for windows that have it)
        try:
            attention_weights, metadata = load_attention_map(
                attn_dir, video_id, frame_start, frame_end,
                layer_idx, feature_level
            )
        except FileNotFoundError as e:
            load_attempts.append(f"with level_{feature_level}: {e}")
            # Pattern 2: Without level suffix (for windows that don't have it)
            try:
                attention_weights, metadata = load_attention_map(
                    attn_dir, video_id, frame_start, frame_end,
                    layer_idx, None
                )
            except FileNotFoundError as e2:
                load_attempts.append(f"without level suffix: {e2}")
                # If both fail, raise with helpful message
                raise FileNotFoundError(
                    f"Could not find attention map for layer {layer_idx} (level {feature_level}) "
                    f"for video {video_id}, frames {frame_start}-{frame_end}. "
                    f"Attempted: {', '.join(load_attempts)}"
                )
        
        num_frames, num_queries, spatial_length = attention_weights.shape
        
        # Get spatial dimensions for this feature level
        spatial_shape = metadata.get('spatial_shape')
        if spatial_shape and isinstance(spatial_shape, list) and len(spatial_shape) == 2:
            H_feat, W_feat = spatial_shape[0], spatial_shape[1]
        elif size_list is not None and feature_level < len(size_list) and size_list[feature_level] is not None:
            # Use size_list if available
            H_feat, W_feat = size_list[feature_level]
        else:
            # Infer from spatial_length
            aspect_ratio = original_img_width / original_img_height
            factor_pairs = []
            for h in range(1, int(np.sqrt(spatial_length)) + 1):
                if spatial_length % h == 0:
                    w = spatial_length // h
                    factor_pairs.append((h, w))
                    if h != w:
                        factor_pairs.append((w, h))
            
            if not factor_pairs:
                print(f"Warning: Cannot find factors for spatial_length={spatial_length} for layer {layer_idx}, skipping")
                continue
            
            # Find best matching aspect ratio
            best_h, best_w = None, None
            min_diff = float('inf')
            for h, w in factor_pairs:
                computed_ar = w / h
                diff = abs(computed_ar - aspect_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_h, best_w = h, w
            
            H_feat, W_feat = best_h, best_w
        
        # Get feature stride for this level
        feature_stride = metadata.get('feature_stride')
        if feature_stride is None:
            # Infer from spatial dimensions
            feature_stride_h = original_img_height / H_feat
            feature_stride_w = original_img_width / W_feat
            feature_stride = int((feature_stride_h + feature_stride_w) / 2)
            if feature_stride == 0:
                feature_stride = 8  # Default fallback
        
        print(f"Layer {layer_idx} (feature_level={feature_level}): H_feat={H_feat}, W_feat={W_feat}, spatial_length={spatial_length}, stride={feature_stride}")
        
        # Verify dimensions match
        if H_feat * W_feat != spatial_length:
            print(f"WARNING: Layer {layer_idx}: H_feat * W_feat ({H_feat * W_feat}) != spatial_length ({spatial_length})")
            # Try to fix
            for h in range(1, int(np.sqrt(spatial_length)) + 1):
                if spatial_length % h == 0:
                    w = spatial_length // h
                    H_feat, W_feat = h, w
                    break
            print(f"Adjusted: H_feat={H_feat}, W_feat={W_feat}")
        
        # Collect all attention maps first (needed for across_frames scaling modes)
        all_attention_2d_maps = [None] * num_frames  # Indexed by frame_idx
        query_ids_by_frame_idx = [None] * num_frames  # Store query IDs for filename generation, indexed by frame_idx
        
        for frame_idx in range(num_frames):
            absolute_frame_num = frame_start + frame_idx
            frame_key = str(absolute_frame_num)
            
            # Determine if we should plot attention for this frame
            should_plot_attention = False
            query_id_for_frame = None
            tracker_attention_weights_for_frame = None
            
            if predictor_query_ids_by_frame is not None and frame_key in predictor_query_ids_by_frame:
                frame_value = predictor_query_ids_by_frame[frame_key]
                
                if query_choice == "weighted_sum":
                    if isinstance(frame_value, np.ndarray):
                        tracker_attention_weights_for_frame = frame_value
                        if len(tracker_attention_weights_for_frame) == num_queries:
                            should_plot_attention = True
                else:
                    query_id_for_frame = int(frame_value)
                    if query_id_for_frame < num_queries:
                        should_plot_attention = True
            
            if should_plot_attention:
                # Compute attention map for this frame
                if query_choice == "weighted_sum" and tracker_attention_weights_for_frame is not None:
                    decoder_attn_maps = attention_weights[frame_idx, :, :]
                    weighted_sum_attn_1d = np.sum(
                        tracker_attention_weights_for_frame[:, np.newaxis] * decoder_attn_maps,
                        axis=0
                    )
                    attn_1d = weighted_sum_attn_1d
                elif query_id_for_frame is not None:
                    attn_1d = attention_weights[frame_idx, query_id_for_frame, :]
                else:
                    attn_1d = None
                
                if attn_1d is not None:
                    # Map to image coordinates
                    attn_2d = map_attention_to_image(
                        attn_1d, H_feat, W_feat,
                        original_img_height, original_img_width,
                        feature_stride
                    )
                    all_attention_2d_maps[frame_idx] = attn_2d
                    query_ids_by_frame_idx[frame_idx] = query_id_for_frame
        
        # Apply colour scaling based on mode
        # Steps for each mode:
        # 1. per_frame: Normalize each frame independently (min/max per frame) - handled in overlay function
        # 2. across_frames: Normalize across all frames (global min/max)
        # 3. temporal_per_frame: 
        #    a) Normalize each frame independently to [0, 1]
        #    b) Scale each normalized frame by its temporal attention weight
        #    c) Use global max across all scaled frames as peak value for heatmap
        # 4. temporal_across_frames:
        #    a) Scale each frame by its temporal attention weight
        #    b) Normalize across all frames (global min/max of scaled values)
        
        # Step 1: Apply temporal scaling BEFORE normalization for temporal_across_frames
        if colour_scale == "temporal_across_frames":
            if temporal_attention_weights is not None:
                for frame_idx in range(num_frames):
                    if all_attention_2d_maps[frame_idx] is not None:
                        absolute_frame_num = frame_start + frame_idx
                        if absolute_frame_num < len(temporal_attention_weights):
                            # Scale attention map by temporal weight
                            all_attention_2d_maps[frame_idx] = all_attention_2d_maps[frame_idx] * temporal_attention_weights[absolute_frame_num]
            else:
                print(f"Warning: temporal attention weights not available for video {video_id}, ignoring temporal scaling")
        
        # Step 2: Apply temporal scaling for temporal_per_frame (normalize per frame, scale by temporal, find global max)
        if colour_scale == "temporal_per_frame":
            if temporal_attention_weights is not None:
                # Process all frames: normalize per frame, scale by temporal weight
                for frame_idx in range(num_frames):
                    if all_attention_2d_maps[frame_idx] is not None:
                        absolute_frame_num = frame_start + frame_idx
                        if absolute_frame_num < len(temporal_attention_weights):
                            attn_2d = all_attention_2d_maps[frame_idx]
                            # Step 1: Normalize per frame to [0, 1]
                            attn_min_frame = attn_2d.min()
                            attn_max_frame = attn_2d.max()
                            if attn_max_frame > attn_min_frame:
                                attn_2d_norm = (attn_2d - attn_min_frame) / (attn_max_frame - attn_min_frame)
                            else:
                                attn_2d_norm = attn_2d - attn_min_frame  # All zeros or constant
                            
                            # Step 2: Scale by temporal weight
                            temporal_weight = temporal_attention_weights[absolute_frame_num]
                            attn_2d_scaled = attn_2d_norm * temporal_weight
                            
                            # Store the scaled version
                            all_attention_2d_maps[frame_idx] = attn_2d_scaled
                
                # Step 3: Find global max across all scaled frames
                valid_maps = [m for m in all_attention_2d_maps if m is not None]
                if valid_maps:
                    global_max_temporal = max(m.max() for m in valid_maps)
                else:
                    global_max_temporal = 1.0
            else:
                print(f"Warning: temporal attention weights not available for video {video_id}, ignoring temporal scaling")
                global_max_temporal = None
        else:
            global_max_temporal = None
        
        # Step 3: Compute global min/max if needed for across_frames modes
        if colour_scale in ["across_frames", "temporal_across_frames"]:
            # Compute global min/max across all valid frames
            valid_maps = [m for m in all_attention_2d_maps if m is not None]
            if valid_maps:
                global_min = min(m.min() for m in valid_maps)
                global_max = max(m.max() for m in valid_maps)
            else:
                global_min = 0.0
                global_max = 1.0
        else:
            global_min = None
            global_max = None
        
        # Plot for each frame using pre-computed attention maps
        for frame_idx in range(num_frames):
            # Calculate the absolute frame number (frame_start + frame_idx)
            absolute_frame_num = frame_start + frame_idx
            
            # Load frame if available
            if frames and frame_idx < len(frames) and frames[frame_idx] is not None:
                frame = frames[frame_idx]
                # Resize to original dimensions first if needed
                if frame.shape[:2] != (original_img_height, original_img_width):
                    frame = cv2.resize(frame, (original_img_width, original_img_height))
            else:
                # Create dummy frame if not available
                frame = np.zeros((original_img_height, original_img_width, 3), dtype=np.uint8)
                # Add grid for reference
                for i in range(0, original_img_height, 50):
                    frame[i, :] = [128, 128, 128]
                for j in range(0, original_img_width, 50):
                    frame[:, j] = [128, 128, 128]
            
            # Get pre-computed attention map for this frame
            attn_2d = all_attention_2d_maps[frame_idx]
            
            if attn_2d is not None:
                # Handle temporal_per_frame: use global max across all scaled frames
                if colour_scale == "temporal_per_frame":
                    if global_max_temporal is not None:
                        # Use global max as peak value, min is 0 (since we normalized to [0,1] before scaling)
                        attn_min = 0.0
                        attn_max = global_max_temporal
                    else:
                        # Fallback to per_frame if temporal weights not loaded
                        attn_min = None
                        attn_max = None
                elif colour_scale in ["across_frames", "temporal_across_frames"]:
                    # Use global min/max computed earlier
                    attn_min = global_min
                    attn_max = global_max
                else:
                    # per_frame: normalize per frame
                    attn_min = None  # Will use attn_2d.min() in overlay function
                    attn_max = None  # Will use attn_2d.max() in overlay function
                
                # Overlay attention with appropriate normalization
                overlaid = overlay_attention_heatmap(
                    frame, attn_2d, alpha=0.5,
                    attn_min=attn_min, attn_max=attn_max
                )
            else:
                # For frame 0 or frames without query IDs, just use the original frame
                overlaid = frame.copy()
            
            # Resize to output dimensions (video dimensions from val.json, same as masks)
            # Use output dimensions if provided, otherwise keep original size
            final_height = output_img_height if output_img_height is not None else original_img_height
            final_width = output_img_width if output_img_width is not None else original_img_width
            if overlaid.shape[:2] != (final_height, final_width):
                overlaid = cv2.resize(overlaid, (final_width, final_height), interpolation=cv2.INTER_LINEAR)
            
            # Save plot in layer-specific directory
            # Use the query ID in the filename if available, otherwise use a placeholder
            query_id_for_frame = query_ids_by_frame_idx[frame_idx]
            if query_choice == "weighted_sum":
                query_label = "weighted_sum"
            elif query_id_for_frame is not None:
                query_label = f"query_{query_id_for_frame}"
            else:
                query_label = "no_query"
            
            output_path = os.path.join(
                layer_output_dir,
                f"video_{video_id}_{query_label}_layer_{layer_idx}_frame_{absolute_frame_num}.png"
            )
            
            # Save image directly without matplotlib padding/titles to preserve exact dimensions
            # Convert RGB to BGR for cv2
            overlaid_bgr = cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, overlaid_bgr)
    
    print(f"Saved attention plots for video {video_id} to {output_dir}/{colour_scale}/video_{video_id}/")


def load_image_dimensions_from_eval(directory, video_id):
    """
    Load image and patch dimensions from image_dimensions.json file in inference directory.
    
    Args:
        directory: Path to directory containing 'attention_maps' folder (e.g., /path/to/attn_extract_3231)
        video_id: Video ID number
    
    Returns:
        dict with keys:
        - 'padded_image_height', 'padded_image_width' (padded dimensions used by model)
        - 'original_image_height', 'original_image_width' (original image dimensions)
        - 'H_patches', 'W_patches' (patch grid dimensions)
        - 'pad_height', 'pad_width', 'pad_top', 'pad_left', 'pad_bottom', 'pad_right' (padding info)
        Raises FileNotFoundError if not found
    """
    # image_dimensions.json is in inference/ directory relative to the specified directory
    inference_dir = os.path.join(directory, "inference")
    dims_path = os.path.join(inference_dir, "image_dimensions.json")
    
    if not os.path.exists(dims_path):
        raise FileNotFoundError(f"Could not find image_dimensions.json at {dims_path}")
    
    with open(dims_path, 'r') as f:
        dims_dict = json.load(f)
        video_key = str(video_id)
        if video_key not in dims_dict:
            raise KeyError(f"Video {video_id} not found in image_dimensions.json")
        return dims_dict[video_key]


def main():
    parser = argparse.ArgumentParser(
        description="Plot cross-attention maps for top-scoring prediction per video_id"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to directory containing 'attention_maps' and 'inference' folders (e.g., /path/to/attn_extract_3231)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[6, 7, 8],
        help="Layer indices to visualize (default: [6, 7, 8] - last 3 layers, matching Mask2Former paper)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: directory/inference/decoder_attention_plots)"
    )
    parser.add_argument(
        "--val-json",
        type=str,
        default=None,
        help="Path to val.json file to use for finding original images. If not provided, will search automatically."
    )
    parser.add_argument(
        "--query-choice",
        type=str,
        choices=["top", "weighted_sum"],
        default="weighted_sum",
        help="Method to choose decoder query ID: 'top' uses layer 5's top index from tracker_attention_top5.json, 'weighted_sum' uses weighted combination (default: top)"
    )
    parser.add_argument(
        "--colour-scale",
        type=str,
        choices=["per_frame", "across_frames", "temporal_per_frame", "temporal_across_frames"],
        default="per_frame",
        help="Colour scaling method: 'per_frame' normalizes each frame independently (default), "
             "'across_frames' normalizes across all frames, "
             "'temporal_per_frame' normalizes per frame then scales by temporal weights, "
             "'temporal_across_frames' normalizes across frames then scales by temporal weights"
    )
    
    args = parser.parse_args()
    
    directory = args.directory
    attention_maps_dir = os.path.join(directory, "attention_maps")
    inference_dir = os.path.join(directory, "inference")
    results_path = os.path.join(inference_dir, "results_temporal.json")
    
    if not os.path.exists(attention_maps_dir):
        raise FileNotFoundError(f"Could not find 'attention_maps' folder in {directory}")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Could not find results_temporal.json at {results_path}")
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(inference_dir, "decoder_attention_plots")
    else:
        output_dir = args.output_dir
    
    # Find and load validation JSON to get video dimensions (same as masks use)
    try:
        if args.val_json is not None:
            val_json_path = args.val_json
            print(f"Using provided validation JSON: {val_json_path}")
        else:
            val_json_path = find_val_json_in_model_dir(directory)
            print(f"Found validation JSON: {val_json_path}")
        with open(val_json_path, 'r') as f:
            val_data = json.load(f)
        video_info_dict = {v['id']: v for v in val_data['videos']}
    except Exception as e:
        print(f"Warning: Could not load validation JSON: {e}")
        video_info_dict = {}
    
    # Load results
    print(f"Loading results from {results_path}")
    results = load_results(results_path)
    
    # Find top prediction for each video_id
    top_per_video = find_top_prediction_per_video(results)
    if not top_per_video:
        print("No predictions found in results")
        return
    
    print(f"Found top predictions for {len(top_per_video)} video(s): {sorted(top_per_video.keys())}")
    
    # Process each video_id
    for video_id, top_pred in sorted(top_per_video.items()):
        print(f"\n{'='*80}")
        print(f"Processing video_id={video_id}")
        print(f"Top prediction: score={top_pred['score']:.6f}, category_id={top_pred['category_id']}, "
              f"refiner_id={top_pred.get('refiner_id', 'N/A')}")
        
        # Get video dimensions from validation JSON (same as masks use)
        if video_id in video_info_dict:
            video_height = video_info_dict[video_id]['height']
            video_width = video_info_dict[video_id]['width']
            print(f"Video dimensions (from val.json): {video_width}x{video_height}")
        else:
            print(f"Warning: Video {video_id} not found in validation JSON, will use image_dimensions.json")
            video_height = None
            video_width = None
        
        # Load query IDs from tracker_attention_top5.json
        tracker_attention_json_path = os.path.join(directory, "inference", "tracker_attention_top5.json")
        try:
            query_ids_by_frame = load_query_ids_from_tracker_attention(tracker_attention_json_path, video_id, query_choice=args.query_choice)
            print(f"Loaded {len(query_ids_by_frame)} query ID mappings from tracker_attention_top5.json for video {video_id} using method '{args.query_choice}'")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading tracker attention data: {e}")
            print(f"Falling back to predictor_query_ids from results_temporal.json")
            
            # Fallback to original method
            predictor_query_ids = top_pred.get('predictor_query_ids', [])
            if not predictor_query_ids:
                print(f"No predictor_query_ids found for video {video_id}, skipping")
                continue
            
            print(f"Found {len(predictor_query_ids)} predictor_query_id mappings for video {video_id}")
            
            # Process each predictor_query_id mapping
            for query_mapping in predictor_query_ids:
                frame_start = query_mapping['frame_start']
                frame_end = query_mapping['frame_end']
                
                # Get per-frame query IDs dictionary (new format)
                predictor_query_ids_by_frame = query_mapping.get('predictor_query_ids_by_frame')
                
                # For backward compatibility, also check for old format
                if predictor_query_ids_by_frame is None:
                    # Old format: single query ID for the whole window
                    predictor_query_id = query_mapping.get('predictor_query_id')
                    if predictor_query_id is not None:
                        print(f"\nProcessing query {predictor_query_id} for frames {frame_start}-{frame_end} (old format - single query ID)")
                        # Convert to new format: create a dictionary with query ID for all frames except frame 0
                        predictor_query_ids_by_frame = {}
                        for frame_num in range(frame_start + 1, frame_end + 1):
                            predictor_query_ids_by_frame[str(frame_num)] = predictor_query_id
                    else:
                        print(f"\nWarning: No predictor_query_ids_by_frame or predictor_query_id found for frames {frame_start}-{frame_end}, skipping")
                        continue
                else:
                    print(f"\nProcessing frames {frame_start}-{frame_end} with per-frame query IDs (found {len(predictor_query_ids_by_frame)} frame mappings)")
                
                # Process this window (existing code continues below)
                # ... (rest of the processing code)
                continue
        
        # Determine frame range from query_ids_by_frame
        if not query_ids_by_frame:
            print(f"No query IDs found for video {video_id}, skipping")
            continue
        
        # Find available attention map windows
        available_windows = find_available_attention_windows(attention_maps_dir, video_id)
        if not available_windows:
            print(f"No attention map windows found for video {video_id}, skipping")
            continue
        
        print(f"Found {len(available_windows)} attention map window(s): {available_windows}")
        
        # Load image dimensions from image_dimensions.json for attention mapping (once for all windows)
        try:
            eval_dims = load_image_dimensions_from_eval(directory, video_id)
            original_img_height = eval_dims['original_image_height']
            original_img_width = eval_dims['original_image_width']
            pad_info = {
                'pad_top': eval_dims.get('pad_top', 0),
                'pad_left': eval_dims.get('pad_left', 0),
                'pad_bottom': eval_dims.get('pad_bottom', 0),
                'pad_right': eval_dims.get('pad_right', 0),
            }
            print(f"Original image dimensions: {original_img_width}x{original_img_height}")
            print(f"Padding: top={pad_info['pad_top']}, left={pad_info['pad_left']}, "
                  f"bottom={pad_info['pad_bottom']}, right={pad_info['pad_right']}")
        except Exception as e:
            print(f"Error loading image dimensions: {e}")
            continue
        
        # Use video dimensions from val.json if available, otherwise fall back to original
        if video_height is not None and video_width is not None:
            output_img_height = video_height
            output_img_width = video_width
        else:
            output_img_height = original_img_height
            output_img_width = original_img_width
        
        print(f"Output image dimensions: {output_img_width}x{output_img_height}")
        
        # Set default output dimensions if not provided
        if output_img_height is None:
            output_img_height = original_img_height
        if output_img_width is None:
            output_img_width = original_img_width
        
        # Load temporal attention weights (activation_proj) if needed for temporal scaling modes
        temporal_attention_weights = None
        if args.colour_scale in ["temporal_per_frame", "temporal_across_frames"]:
            temporal_attention_weights = load_temporal_attention_weights(directory, video_id)
            if temporal_attention_weights is not None:
                print(f"Loaded activation_proj weights (temporal importance): shape {temporal_attention_weights.shape}")
            else:
                print(f"Warning: Could not load activation_proj weights for video {video_id}")
                print(f"Falling back to non-temporal scaling mode")
                # Fallback to non-temporal version
                if args.colour_scale == "temporal_per_frame":
                    args.colour_scale = "per_frame"
                elif args.colour_scale == "temporal_across_frames":
                    args.colour_scale = "across_frames"
        
        # Process each window separately
        for window_start, window_end in available_windows:
            print(f"\nProcessing window: frames {window_start}-{window_end}")
            
            # Filter query_ids_by_frame to only include frames in this window
            window_query_ids = {
                frame_str: query_id 
                for frame_str, query_id in query_ids_by_frame.items()
                if window_start <= int(frame_str) <= window_end
            }
            
            if not window_query_ids:
                print(f"No query IDs found for frames {window_start}-{window_end}, skipping window")
                continue
            
            # Use window_query_ids as predictor_query_ids_by_frame for this window
            predictor_query_ids_by_frame = window_query_ids
            
            # Try to get size_list from metadata (if available from predictor hook)
            # size_list contains (H_feat, W_feat) for each feature level [level0, level1, level2]
            size_list = None
            try:
                # Try to load metadata from first layer to see if we have size_list info
                feature_level = args.layers[0] % 3
                try:
                    _, metadata = load_attention_map(
                        attention_maps_dir, video_id, window_start, window_end,
                        args.layers[0], feature_level if feature_level > 0 else None
                    )
                except FileNotFoundError:
                    _, metadata = load_attention_map(
                        attention_maps_dir, video_id, window_start, window_end,
                        args.layers[0], None
                    )
                
                # Check if we have spatial_shape in metadata (for this specific layer/level)
                # But we need size_list for all levels, so we'll need to load metadata from multiple layers
                # or infer from the available data
                
                # Try to build size_list from available metadata
                # We'll need to check multiple layers to get all feature levels
                size_list = [None, None, None]  # [level0, level1, level2]
                
                # Try to get spatial_shape from each feature level by checking different layers
                for test_layer in [0, 1, 2]:  # These use levels 0, 1, 2 respectively
                    test_level = test_layer % 3
                    try:
                        _, test_metadata = load_attention_map(
                            attention_maps_dir, video_id, window_start, window_end,
                            test_layer, test_level if test_level > 0 else None
                        )
                        test_spatial_shape = test_metadata.get('spatial_shape')
                        if test_spatial_shape and isinstance(test_spatial_shape, list) and len(test_spatial_shape) == 2:
                            size_list[test_level] = tuple(test_spatial_shape)
                    except (FileNotFoundError, KeyError):
                        pass
                
                # If we got any valid dimensions, use them
                if any(s is not None for s in size_list):
                    print(f"Found size_list from metadata: {size_list}")
                else:
                    print(f"Could not find size_list in metadata, will infer per layer")
                    size_list = None
                
            except FileNotFoundError:
                # Metadata files don't exist for this query/window - this is expected in some cases
                # Dimensions will be inferred when loading the actual attention maps
                print(f"Note: Metadata not found for frames {window_start}-{window_end}, will infer dimensions from attention maps")
                size_list = None
            except Exception as e:
                print(f"Warning: Error loading metadata: {e}, will infer dimensions from attention maps")
                size_list = None
            
            # Load frames
            frames = load_video_frames(
                directory, video_id, window_start, window_end,
                original_img_height, original_img_width,
                pad_top=pad_info['pad_top'],
                pad_left=pad_info['pad_left'],
                pad_bottom=pad_info['pad_bottom'],
                pad_right=pad_info['pad_right'],
                val_json_path=args.val_json
            )
            
            # Plot attention (output_dir structure will be created inside plot_attention_for_query)
            plot_attention_for_query(
                attention_maps_dir, output_dir, video_id,
                predictor_query_ids_by_frame, window_start, window_end,
                args.layers, original_img_height, original_img_width,
                frames, size_list=size_list,
                output_img_height=output_img_height,
                output_img_width=output_img_width,
                query_choice=args.query_choice,
                colour_scale=args.colour_scale,
                temporal_attention_weights=temporal_attention_weights
            )
    
    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()

