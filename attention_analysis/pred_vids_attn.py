import os
import json
import cv2
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from pycocotools import mask as mask_utils

def decode_rle(rle_obj, height, width):
    if isinstance(rle_obj['counts'], list):
        rle = mask_utils.frPyObjects(rle_obj, height, width)
    else:
        rle = rle_obj
    return mask_utils.decode(rle)

def draw_mask_outline(frame, mask, color):
    """Draw only the outline of the mask"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, thickness=2)

def draw_truth_and_prediction(frame, true_species_name, pred_species_name, score):
    """Draw truth, prediction species, and score stacked in bottom left corner"""
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)     # Black background
    
    # Prepare text lines
    truth_text = f"True: {true_species_name}"
    pred_text = f"Pred: {pred_species_name}"
    score_text = f"Score: {score:.2f}"
    
    # Get text sizes
    (truth_width, truth_height), baseline = cv2.getTextSize(truth_text, font, font_scale, thickness)
    (pred_width, pred_height), baseline = cv2.getTextSize(pred_text, font, font_scale, thickness)
    (score_width, score_height), baseline = cv2.getTextSize(score_text, font, font_scale, thickness)
    
    # Use the maximum width
    max_width = max(truth_width, pred_width, score_width)
    line_height = truth_height  # All lines should have same height
    total_height = line_height * 3 + 5 * 2  # 3 lines with 5px spacing between each
    
    # Position in bottom left corner with padding
    padding = 10
    x = padding
    y = height - padding
    
    # Draw background rectangle for all lines
    cv2.rectangle(frame, (x - 5, y - total_height - 5), (x + max_width + 5, y + 5), bg_color, -1)
    
    # Draw truth text (top line)
    cv2.putText(frame, truth_text, (x, y - line_height * 2 - 5 * 2), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    
    # Draw prediction text (middle line)
    cv2.putText(frame, pred_text, (x, y - line_height - 5), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    
    # Draw score text (bottom line)
    cv2.putText(frame, score_text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

def draw_frame_refiner_and_importance(frame, frame_num, total_frames, refiner_id=None, importance_value=None):
    """Draw frame number, refiner ID, and per-frame importance stacked in bottom right corner.
    importance_value should be in [0,1] if provided.
    """
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)     # Black background
    
    # Prepare lines and measure sizes
    lines = []
    frame_text = f"Frame {frame_num}/{total_frames}"
    lines.append(frame_text)
    if refiner_id is not None:
        lines.append(f"Refiner ID: {refiner_id}")
    if importance_value is not None:
        lines.append(f"Importance: {importance_value:.2f}")

    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_heights = [cv2.getTextSize(line, font, font_scale, thickness)[0][1] for line in lines]
    max_text_width = max([w for (w, h) in line_sizes]) if line_sizes else 0
    line_height = line_heights[0] if line_heights else 0
    total_height = sum(line_heights) + (len(lines) - 1) * 5
    
    # Position in bottom right corner with padding
    padding = 10
    max_width = max_text_width

    x = width - max_width - padding
    y = height - padding
    
    # Draw background rectangle for all lines
    cv2.rectangle(frame, (x - 5, y - total_height - 5), (x + max_width + 5, y + 5), bg_color, -1)
    
    # Draw lines from bottom to top
    current_y = y
    for i, line in enumerate(reversed(lines)):
        cv2.putText(frame, line, (x, current_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
        if i < len(lines) - 1:
            current_y -= (line_height + 5)

def hex_to_bgr(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def draw_small_top_right_label(frame, text):
    """Draw a small label in the top right corner (smaller than main labels)."""
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 8
    x = width - text_width - padding
    y = padding + text_height
    # background box
    cv2.rectangle(frame, (x - 4, y - text_height - 4), (x + text_width + 4, y + 4), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

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

def find_attention_images(decoder_attention_plots_dir: str, video_id: int, layer_idx: int) -> dict:
    """
    Find all attention images for a specific layer across all queries.
    Prioritizes weighted_sum images (from plot_decoder_attn.py with query-choice=weighted_sum).
    
    Args:
        decoder_attention_plots_dir: Output directory from plot_decoder_attn.py
        video_id: Video ID
        layer_idx: Layer index to find images for
    
    Returns:
        Dictionary mapping frame_idx to image path
    """
    frame_to_image = {}
    
    # Directory structure: decoder_attention_plots_dir/video_{video_id}/layer_{layer_idx}/
    layer_dir = os.path.join(decoder_attention_plots_dir, f"video_{video_id}", f"layer_{layer_idx}")
    
    if not os.path.exists(layer_dir):
        return frame_to_image
    
    # First, try to find weighted_sum images (default from plot_decoder_attn.py)
    # Pattern: video_{video_id}_weighted_sum_layer_{layer_idx}_frame_{frame_idx}.png
    weighted_sum_pattern = re.compile(rf'video_{video_id}_weighted_sum_layer_{layer_idx}_frame_(\d+)\.png')
    
    # Search for weighted_sum images first
    for filename in os.listdir(layer_dir):
        match = weighted_sum_pattern.match(filename)
        if match:
            frame_idx = int(match.group(1))
            frame_to_image[frame_idx] = os.path.join(layer_dir, filename)
    
    # If we found weighted_sum images, return them (they're the default)
    if frame_to_image:
        return frame_to_image
    
    # Fallback to query-based pattern if no weighted_sum images found
    # Pattern: video_{video_id}_query_{query_id}_layer_{layer_idx}_frame_{frame_idx}.png
    query_pattern = re.compile(rf'video_{video_id}_query_(\d+)_layer_{layer_idx}_frame_(\d+)\.png')
    
    # Search in the layer-specific directory
    for filename in os.listdir(layer_dir):
        match = query_pattern.match(filename)
        if match:
            frame_idx = int(match.group(2))
            # If multiple queries have the same frame, keep the first one found
            # (or we could prefer a specific query, but for now just use first)
            if frame_idx not in frame_to_image:
                frame_to_image[frame_idx] = os.path.join(layer_dir, filename)
    
    return frame_to_image

def find_available_layers(decoder_attention_plots_dir: str, video_id: int) -> list:
    """
    Find all layers that have attention images for a specific video.
    
    Args:
        decoder_attention_plots_dir: Output directory from plot_decoder_attn.py
        video_id: Video ID
    
    Returns:
        List of layer indices that have attention images
    """
    video_dir = os.path.join(decoder_attention_plots_dir, f"video_{video_id}")
    if not os.path.exists(video_dir):
        return []
    
    layers = []
    layer_pattern = re.compile(r'layer_(\d+)')
    for item in os.listdir(video_dir):
        item_path = os.path.join(video_dir, item)
        if os.path.isdir(item_path):
            match = layer_pattern.match(item)
            if match:
                layer_idx = int(match.group(1))
                # Check if this layer has any images
                if os.listdir(item_path):
                    layers.append(layer_idx)
    
    return sorted(layers)

def visualize_predictions_from_attention_dir(attention_dir, val_json_path=None):
    """
    Create prediction videos with attention overlays from attention directory.
    
    Args:
        attention_dir: Path to attention extraction directory (e.g., /path/to/eval_attn_6059)
                      Should contain inference/decoder_attention_plots/ subdirectory
        val_json_path: Optional path to val.json file. If not provided, will search automatically.
    """
    attention_dir = os.path.abspath(attention_dir)
    
    # Auto-detect paths
    inference_dir = os.path.join(attention_dir, "inference")
    decoder_attention_plots_dir = os.path.join(inference_dir, "decoder_attention_plots")
    results_json = os.path.join(inference_dir, "results_temporal.json")
    
    if not os.path.exists(decoder_attention_plots_dir):
        raise FileNotFoundError(f"Decoder attention plots directory not found: {decoder_attention_plots_dir}")
    if not os.path.exists(results_json):
        raise FileNotFoundError(f"Results JSON not found: {results_json}")
    
    # Find val.json
    try:
        if val_json_path is not None:
            valid_json = val_json_path
            print(f"Using provided validation JSON: {valid_json}")
        else:
            valid_json = find_val_json_in_model_dir(attention_dir)
            print(f"Found validation JSON: {valid_json}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find val.json. {e}")
    
    # Determine image root from model directory
    model_dir = os.path.dirname(valid_json)
    try:
        image_root = get_image_root_directory(model_dir)
        print(f"Using image root: {image_root}")
    except ValueError as e:
        print(f"Warning: {e}")
        image_root = None  # Will only use attention images
    
    # Output directory structure: attention_dir/inference/attn_pred_vids/video_X/layer_X/
    output_base_dir = os.path.join(inference_dir, "attn_pred_vids")
    
    # Load results and validation data
    with open(results_json) as f:
        predictions = json.load(f)
    
    with open(valid_json) as f:
        valid_data = json.load(f)
    
    categories = {cat["id"]: cat["name"] for cat in valid_data["categories"]}
    video_info = {v["id"]: v for v in valid_data["videos"]}
    
    # Extract true species information from annotations
    true_species_by_video = {}
    if "annotations" in valid_data:
        for ann in valid_data["annotations"]:
            video_id = ann["video_id"]
            category_id = ann["category_id"]
            if category_id in categories:
                true_species_name = categories[category_id]
                if video_id not in true_species_by_video:
                    true_species_by_video[video_id] = []
                true_species_by_video[video_id].append(true_species_name)
    
    # Remove duplicates and join multiple species if present
    for video_id in true_species_by_video:
        true_species_by_video[video_id] = ", ".join(list(set(true_species_by_video[video_id])))
    
    # Choose color per category
    np.random.seed(42)
    category_colors = {cat_id: tuple(np.random.randint(0, 256, 3).tolist()) for cat_id in categories}
    
    # Group ALL predictions by video_id
    all_preds_by_video = {}
    for pred in predictions:
        all_preds_by_video.setdefault(pred["video_id"], []).append(pred)
    
    print(f"Found {len(all_preds_by_video)} videos with predictions")
    
    # Derive run_dir to locate attention maps
    attn_maps_dir = os.path.join(inference_dir, "attention_maps")
    
    # Track videos that don't meet criteria
    skipped_videos = []
    processed_layers = 0
    
    # Process each video that has predictions
    for video_id, video_preds in tqdm(all_preds_by_video.items(), desc="Processing videos"):
        if video_id not in video_info:
            folder_name = f"video_id_{video_id}"
            skipped_videos.append((folder_name, "Video ID not found in validation JSON"))
            print(f"Warning: Video id {video_id} not found in valid_json, skipping.")
            continue
        
        # Find available layers for this video
        available_layers = find_available_layers(decoder_attention_plots_dir, video_id)
        if not available_layers:
            skipped_videos.append((f"video_{video_id}", "No attention images found"))
            print(f"Warning: No attention images found for video {video_id}, skipping.")
            continue
        
        video = video_info[video_id]
        file_names = video["file_names"]
        height = video["height"]
        width = video["width"]
        folder_name = os.path.dirname(file_names[0])
        total_frames = len(file_names)
        
        # Get true species for this video
        true_species = true_species_by_video.get(video_id, "Unknown")
        
        # Select the prediction with the highest score for this video
        best_pred = max(video_preds, key=lambda x: x["score"])
        
        # Extract refiner_id if available
        refiner_id = best_pred.get("refiner_id", None)
        
        # Get prediction species name
        category_id = best_pred["category_id"]
        pred_species_name = categories.get(category_id, f"Unknown (id={category_id})")
        
        # Print predictions for this video
        print(f"\nVideo {video_id} ({folder_name}):")
        print(f"  True species: {true_species}")
        print(f"  Best prediction: {pred_species_name} (score: {best_pred['score']:.4f})")
        if refiner_id is not None:
            print(f"  Refiner ID: {refiner_id}")
        print(f"  Available layers: {available_layers}")
        
        # Create frame masks from the best prediction only
        frame_masks = {fn: [] for fn in file_names}
        segmentations = best_pred["segmentations"]
        color = category_colors.get(category_id, (255, 255, 255))
        
        for idx, rle in enumerate(segmentations):
            if idx >= len(file_names):
                continue
            fn = file_names[idx]
            mask = decode_rle(rle, height, width)
            frame_masks[fn].append((mask, color))
        
        # Load per-frame importance if available (using nolayer0 version)
        importance_arr = None
        if refiner_id is not None:
            try:
                # Try nolayer0 version first (rolled_out is at attention_dir/attention_maps/rolled_out/)
                rolled_out_dir = os.path.join(attention_dir, "attention_maps", "rolled_out")
                colavg_norm_npz = os.path.join(rolled_out_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_nolayer0_col_avg_norm.npz")
                if os.path.exists(colavg_norm_npz):
                    data = np.load(colavg_norm_npz)
                    if 'col_avg_norm' in data:
                        importance_arr = data['col_avg_norm']
                        print(f"  Loaded importance array (nolayer0): {len(importance_arr)} values")
                    else:
                        print(f"  Warning: 'col_avg_norm' key not found in {colavg_norm_npz}")
                        print(f"    Available keys: {list(data.keys())}")
                else:
                    # Fallback to old location/format for backward compatibility
                    colavg_norm_npz_old = os.path.join(attn_maps_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_rownorm_colavg_norm.npz")
                    if os.path.exists(colavg_norm_npz_old):
                        data = np.load(colavg_norm_npz_old)
                        if 'rollout_rownorm_colavg_norm' in data:
                            importance_arr = data['rollout_rownorm_colavg_norm']
                            print(f"  Loaded importance array (legacy format): {len(importance_arr)} values")
                        else:
                            print(f"  Warning: 'rollout_rownorm_colavg_norm' key not found in {colavg_norm_npz_old}")
                            print(f"    Available keys: {list(data.keys())}")
                    else:
                        print(f"  Warning: Importance file not found: {colavg_norm_npz}")
                        print(f"    Also tried legacy location: {colavg_norm_npz_old}")
            except Exception as e:
                print(f"  Warning: Error loading importance array: {e}")
                importance_arr = None
        else:
            print(f"  Warning: No refiner_id available, cannot load importance array")
        
        # Attention images are now saved at video dimensions (same as masks), so no resizing needed
        print(f"  Video dimensions (masks and attention images): {width}x{height}")
        
        # Process each available layer
        for layer_idx in available_layers:
            print(f"  Processing layer {layer_idx}...")
            
            # Load attention images for this layer
            attention_frame_map = find_attention_images(decoder_attention_plots_dir, video_id, layer_idx)
            if not attention_frame_map:
                print(f"    Warning: No attention images found for layer {layer_idx}, skipping")
                continue
            
            # Create output directory: output_base_dir/video_{video_id}/
            output_dir = os.path.join(output_base_dir, f"video_{video_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Output video path
            output_video_path = os.path.join(output_dir, f"video_{video_id}_layer_{layer_idx}_attention.mp4")
            fps = 10
            out = None
            
            for frame_idx, fn in enumerate(file_names, start=1):
                # Use attention image if available, otherwise fall back to original frame
                if (frame_idx - 1) in attention_frame_map:
                    img_path = attention_frame_map[frame_idx - 1]
                elif image_root:
                    img_path = os.path.join(image_root, fn)
                else:
                    print(f"    Warning: No image available for frame {frame_idx}, skipping")
                    continue
                
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"    Warning: could not read image {img_path}")
                    continue
                
                # Resize frame to match video dimensions if needed (attention images should already match)
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                # Draw truth, prediction, and score stacked in bottom left corner
                draw_truth_and_prediction(frame, true_species, pred_species_name, best_pred["score"])
                
                # Determine importance value for this frame index (0-based)
                imp_val = None
                if importance_arr is not None and (frame_idx - 1) < len(importance_arr):
                    imp_val = float(importance_arr[frame_idx - 1])
                
                # Draw frame number, refiner ID, and importance in bottom right corner
                draw_frame_refiner_and_importance(frame, frame_idx, total_frames, refiner_id, imp_val)
                
                # Draw original name (folder_name) in top right corner, small font
                draw_small_top_right_label(frame, folder_name)
                
                # Draw mask outline (masks are already at correct dimensions)
                for mask, color in frame_masks.get(fn, []):
                    draw_mask_outline(frame, mask, color)
                
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
                
                out.write(frame)
            
            if out:
                out.release()
                print(f"    Saved video to {output_video_path}")
                processed_layers += 1
    
    print(f"\nProcessing complete. Processed {processed_layers} layer videos.")
    print(f"Videos saved to: {output_base_dir}")
    
    # Print summary
    if skipped_videos:
        print(f"\n{len(skipped_videos)} videos were skipped:")
        for video_name, reason in skipped_videos[:10]:  # Show first 10
            print(f"  {video_name}: {reason}")
        if len(skipped_videos) > 10:
            print(f"  ... and {len(skipped_videos) - 10} more")

def visualize_predictions(results_json, valid_json, image_root, output_dir, 
                          decoder_attention_plots_dir=None, layer_idx=6):
    os.makedirs(output_dir, exist_ok=True)

    with open(results_json) as f:
        predictions = json.load(f)

    with open(valid_json) as f:
        valid_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in valid_data["categories"]}
    video_info = {v["id"]: v for v in valid_data["videos"]}
    
    # Extract true species information from annotations
    true_species_by_video = {}
    if "annotations" in valid_data:
        for ann in valid_data["annotations"]:
            video_id = ann["video_id"]
            category_id = ann["category_id"]
            if category_id in categories:
                true_species_name = categories[category_id]
                if video_id not in true_species_by_video:
                    true_species_by_video[video_id] = []
                true_species_by_video[video_id].append(true_species_name)
    
    # Remove duplicates and join multiple species if present
    for video_id in true_species_by_video:
        true_species_by_video[video_id] = ", ".join(list(set(true_species_by_video[video_id])))

    # Choose color per category
    np.random.seed(42)
    category_colors = {cat_id: tuple(np.random.randint(0, 256, 3).tolist()) for cat_id in categories}

    # Group ALL predictions by video_id (no threshold filtering)
    all_preds_by_video = {}
    for pred in predictions:
        all_preds_by_video.setdefault(pred["video_id"], []).append(pred)

    print(f"Found {len(all_preds_by_video)} videos with predictions")

    # Derive run_dir to locate attention maps (two levels up from results_json)
    run_dir = os.path.dirname(os.path.dirname(results_json))
    attn_maps_dir = os.path.join(run_dir, "inference", "attention_maps")
    
    # If decoder_attention_plots_dir is provided, use attention images instead of original frames
    use_attention_images = decoder_attention_plots_dir is not None and os.path.exists(decoder_attention_plots_dir)
    if use_attention_images:
        print(f"Using decoder attention images from: {decoder_attention_plots_dir}")
        print(f"Using layer {layer_idx} for attention visualization")
    
    # Track videos that don't meet criteria
    skipped_videos = []
    
    # Process each video that has predictions
    for video_id, video_preds in tqdm(all_preds_by_video.items(), desc="Processing videos"):
        if video_id not in video_info:
            folder_name = f"video_id_{video_id}"  # Fallback name
            skipped_videos.append((folder_name, "Video ID not found in validation JSON"))
            print(f"Warning: Video id {video_id} not found in valid_json, skipping.")
            continue

        video = video_info[video_id]
        file_names = video["file_names"]
        height = video["height"]
        width = video["width"]
        folder_name = os.path.dirname(file_names[0])
        total_frames = len(file_names)

        # Get true species for this video
        true_species = true_species_by_video.get(video_id, "Unknown")
        
        # Select the prediction with the highest score for this video (no threshold)
        best_pred = max(video_preds, key=lambda x: x["score"])
        
        # Extract refiner_id if available
        refiner_id = best_pred.get("refiner_id", None)
        
        # Get prediction species name
        category_id = best_pred["category_id"]
        pred_species_name = categories.get(category_id, f"Unknown (id={category_id})")
        
        # Print predictions for this video
        print(f"\nVideo {video_id} ({folder_name}):")
        print(f"  True species: {true_species}")
        print(f"  Best prediction: {pred_species_name} (score: {best_pred['score']:.4f})")
        if refiner_id is not None:
            print(f"  Refiner ID: {refiner_id}")
        
        # Create frame masks from the best prediction only
        frame_masks = {fn: [] for fn in file_names}
        segmentations = best_pred["segmentations"]
        color = category_colors.get(category_id, (255, 255, 255))  # Default to white if category not found
        
        for idx, rle in enumerate(segmentations):
            if idx >= len(file_names):
                continue
            fn = file_names[idx]
            mask = decode_rle(rle, height, width)
            frame_masks[fn].append((mask, color))

        # Load per-frame importance (row-normalized column averages) if available
        importance_arr = None
        try:
            colavg_norm_npz = os.path.join(attn_maps_dir, f"video_{video_id}_refiner_{refiner_id}_rollout_rownorm_colavg_norm.npz")
            if os.path.exists(colavg_norm_npz):
                data = np.load(colavg_norm_npz)
                # Key name per temporal_rollout.py
                if 'rollout_rownorm_colavg_norm' in data:
                    importance_arr = data['rollout_rownorm_colavg_norm']
        except Exception:
            importance_arr = None

        # Load attention images if using decoder attention plots
        attention_frame_map = {}
        if use_attention_images:
            attention_frame_map = find_attention_images(decoder_attention_plots_dir, video_id, layer_idx)
            if not attention_frame_map:
                print(f"  Warning: No attention images found for video {video_id}, layer {layer_idx}, using original frames")
                use_attention_images = False

        # Check if any frames have masks (only save videos with predictions)
        has_masks = any(len(masks) > 0 for masks in frame_masks.values())
        if not has_masks:
            skipped_videos.append((folder_name, "No masks to visualize after processing"))
            print(f"  Skipping video {video_id} - no masks to visualize")
            continue

        # Save file as top_pred_video_<video_id>.mp4 (or with layer suffix if using attention images)
        if use_attention_images:
            output_video_path = os.path.join(output_dir, f"top_pred_video_{video_id}_layer_{layer_idx}_attention.mp4")
        else:
            output_video_path = os.path.join(output_dir, f"top_pred_video_{video_id}.mp4")
        fps = 10
        out = None

        for frame_idx, fn in enumerate(file_names, start=1):
            # Use attention image if available, otherwise use original frame
            if use_attention_images and (frame_idx - 1) in attention_frame_map:
                img_path = attention_frame_map[frame_idx - 1]
            else:
                img_path = os.path.join(image_root, fn)
            
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: could not read image {img_path}")
                continue

            # Draw truth, prediction, and score stacked in bottom left corner
            draw_truth_and_prediction(frame, true_species, pred_species_name, best_pred["score"])
            
            # Determine importance value for this frame index (0-based)
            imp_val = None
            if importance_arr is not None and (frame_idx - 1) < len(importance_arr):
                imp_val = float(importance_arr[frame_idx - 1])

            # Draw frame number, refiner ID, and importance in bottom right corner
            draw_frame_refiner_and_importance(frame, frame_idx, total_frames, refiner_id, imp_val)

            # Draw original name (folder_name) in top right corner, small font
            draw_small_top_right_label(frame, folder_name)

            # Draw mask outline only (no label/score text)
            for mask, color in frame_masks.get(fn, []):
                draw_mask_outline(frame, mask, color)

            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

            out.write(frame)

        if out:
            out.release()
            print(f"  Saved video to {output_video_path}")

    # Track videos with no predictions
    videos_with_no_predictions = []
    
    all_video_ids = set(video_info.keys())
    videos_with_any_predictions = set(all_preds_by_video.keys())
    
    # Videos with no predictions at all
    videos_without_any_predictions = all_video_ids - videos_with_any_predictions
    
    for video_id in videos_without_any_predictions:
        if video_id in video_info:
            folder_name = os.path.dirname(video_info[video_id]["file_names"][0])
            videos_with_no_predictions.append((folder_name, "No predictions at all"))
    
    print(f"\nProcessing complete. Videos saved to: {output_dir}")
    
    # Print summary of skipped videos
    all_skipped = skipped_videos + videos_with_no_predictions
    if all_skipped:
        print(f"\n{len(all_skipped)} videos were not saved:")
        print("-" * 80)
        for video_name, reason in all_skipped:
            print(f"  {video_name}: {reason}")
        print("-" * 80)
        
        # Print breakdown
        print(f"\nBreakdown:")
        print(f"  Videos with no predictions at all: {len(videos_with_no_predictions)}")
        print(f"  Videos with processing issues: {len(skipped_videos)}")
        print(f"  Total videos in validation set: {len(all_video_ids)}")
        print(f"  Videos with predictions: {len(videos_with_any_predictions)}")
        print(f"  Videos successfully saved: {len(videos_with_any_predictions) - len([v for v in skipped_videos if 'No masks' in v[1]])}")
    else:
        print("\nAll videos with predictions were successfully processed and saved.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create prediction videos with attention overlays"
    )
    parser.add_argument(
        "attention_dir",
        type=str,
        help="Path to attention extraction directory (e.g., /path/to/eval_attn_6059). "
             "Should contain inference/decoder_attention_plots/ subdirectory"
    )
    parser.add_argument(
        "--val-json",
        type=str,
        default=None,
        help="Path to val.json file to use for finding original images. If not provided, will search automatically."
    )
    
    args = parser.parse_args()
    
    visualize_predictions_from_attention_dir(args.attention_dir, val_json_path=args.val_json)
