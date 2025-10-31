import os
import json
import cv2
import numpy as np
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

def draw_frame_and_refiner(frame, frame_num, total_frames, refiner_id=None):
    """Draw frame number and refiner ID stacked in bottom right corner"""
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)     # Black background
    
    # Prepare text lines
    frame_text = f"Frame {frame_num}/{total_frames}"
    lines = [frame_text]
    
    if refiner_id is not None:
        refiner_text = f"Refiner ID: {refiner_id}"
        lines.append(refiner_text)
    
    # Get text sizes for all lines
    line_widths = []
    line_heights = []
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        line_widths.append(w)
        line_heights.append(h)
    
    # Use the maximum width
    max_width = max(line_widths)
    line_height = line_heights[0]  # All lines should have same height
    total_height = sum(line_heights) + (len(lines) - 1) * 5  # 5px spacing between lines
    
    # Position in bottom right corner with padding
    padding = 10
    x = width - max_width - padding
    y = height - padding
    
    # Draw background rectangle for all lines
    cv2.rectangle(frame, (x - 5, y - total_height - 5), (x + max_width + 5, y + 5), bg_color, -1)
    
    # Draw lines from bottom to top
    current_y = y
    for i, line in enumerate(reversed(lines)):
        cv2.putText(frame, line, (x, current_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
        if i < len(lines) - 1:  # Not the last line
            current_y -= (line_height + 5)

def hex_to_bgr(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def visualize_predictions(results_json, valid_json, image_root, output_dir):
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

        # Check if any frames have masks (only save videos with predictions)
        has_masks = any(len(masks) > 0 for masks in frame_masks.values())
        if not has_masks:
            skipped_videos.append((folder_name, "No masks to visualize after processing"))
            print(f"  Skipping video {video_id} - no masks to visualize")
            continue

        output_video_path = os.path.join(output_dir, f"{folder_name}.mp4")
        fps = 10
        out = None

        for frame_idx, fn in enumerate(file_names, start=1):
            img_path = os.path.join(image_root, fn)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: could not read image {img_path}")
                continue

            # Draw truth, prediction, and score stacked in bottom left corner
            draw_truth_and_prediction(frame, true_species, pred_species_name, best_pred["score"])
            
            # Draw frame number and refiner ID stacked in bottom right corner
            draw_frame_and_refiner(frame, frame_idx, total_frames, refiner_id)

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
    results_json = "/home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_lr5e-4_redo/oct30_attn_0003635_eval/inference/results_temporal.json"
    valid_json = "/data/fishway_ytvis/val_5fish.json"
    image_root = "/data/fishway_ytvis/all_videos"
    output_dir = "/home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_lr5e-4_redo/oct30_attn_0003635_eval/inference/video_predictions"

    visualize_predictions(results_json, valid_json, image_root, output_dir)
