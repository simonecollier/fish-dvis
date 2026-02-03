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

def draw_label_and_mask(frame, mask, label, score, color):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, thickness=4)  # Increased thickness for better visibility
    
    # Get label position from the first contour
    if contours:
        x, y, _, _ = cv2.boundingRect(contours[0])
        # Draw text with background for better visibility
        text = f"{label} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Slightly larger font
        thickness = 3  # Increased thickness
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (x - 3, y - text_height - 8), (x + text_width + 3, y + 3), (0, 0, 0), -1)
        
        cv2.putText(
            frame,
            text,
            (x, y - 5),
            font,
            font_scale,
            color,
            thickness,
            lineType=cv2.LINE_AA
        )

def draw_true_species(frame, true_species_name):
    """Draw true species name in top right corner"""
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)     # Black background
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(f"True: {true_species_name}", font, font_scale, thickness)
    
    # Position in top right corner with padding
    padding = 10
    x = width - text_width - padding
    y = text_height + padding
    
    # Draw background rectangle
    cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), bg_color, -1)
    
    # Draw text
    cv2.putText(frame, f"True: {true_species_name}", (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

def hex_to_bgr(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def visualize_predictions(results_json, valid_json, image_root, output_dir, score_thresh=0.5):
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
            true_species_name = categories[category_id]
            if video_id not in true_species_by_video:
                true_species_by_video[video_id] = []
            true_species_by_video[video_id].append(true_species_name)
    
    # Remove duplicates and join multiple species if present
    for video_id in true_species_by_video:
        true_species_by_video[video_id] = ", ".join(list(set(true_species_by_video[video_id])))

    # Choose color per category - using vibrant, standout colors
    # Bright colors that stand out: cyan, magenta, yellow, lime green, orange, bright blue, pink, etc.
    vibrant_colors = [
        (0, 255, 255),    # Cyan
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Yellow
        (0, 255, 0),      # Lime Green
        (255, 165, 0),    # Orange
        (0, 0, 255),      # Bright Blue
        (255, 0, 0),      # Red
        (255, 20, 147),   # Deep Pink
        (0, 191, 255),    # Deep Sky Blue
        (255, 140, 0),    # Dark Orange
        (50, 205, 50),    # Lime Green
        (255, 215, 0),    # Gold
        (138, 43, 226),   # Blue Violet
        (255, 69, 0),     # Red Orange
        (0, 250, 154),    # Medium Spring Green
        (255, 105, 180),  # Hot Pink
        (30, 144, 255),   # Dodger Blue
        (255, 20, 147),   # Deep Pink
        (124, 252, 0),    # Lawn Green
        (255, 165, 0),    # Orange
    ]
    
    # Assign colors to categories, cycling through the vibrant colors if there are more categories
    np.random.seed(42)
    category_ids = sorted(categories.keys())
    category_colors = {}
    for idx, cat_id in enumerate(category_ids):
        color_idx = idx % len(vibrant_colors)
        category_colors[cat_id] = vibrant_colors[color_idx]

    # Group ALL predictions by video_id (before filtering by threshold)
    all_preds_by_video = {}
    for pred in predictions:
        all_preds_by_video.setdefault(pred["video_id"], []).append(pred)
    
    # Group predictions by video_id and filter by score threshold
    preds_by_video = {}
    for pred in predictions:
        if pred["score"] >= score_thresh:
            preds_by_video.setdefault(pred["video_id"], []).append(pred)

    print(f"Found {len(preds_by_video)} videos with predictions above threshold {score_thresh}")
    
    # Track videos that don't meet criteria
    skipped_videos = []
    
    # Process each video that has predictions
    for video_id, video_preds in tqdm(preds_by_video.items(), desc="Processing videos"):
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

        # Get true species for this video
        true_species = true_species_by_video.get(video_id, "Unknown")
        
        # Select the prediction with the highest score for this video
        best_pred = max(video_preds, key=lambda x: x["score"])
        
        # Print predictions for this video
        print(f"\nVideo {video_id} ({folder_name}):")
        print(f"  True species: {true_species}")
        print(f"  Best prediction: {categories[best_pred['category_id']]} (score: {best_pred['score']:.4f})")
        
        # Create frame masks from the best prediction only
        frame_masks = {fn: [] for fn in file_names}
        category_id = best_pred["category_id"]
        segmentations = best_pred["segmentations"]
        score = best_pred["score"]
        
        for idx, rle in enumerate(segmentations):
            if idx >= len(file_names):
                continue
            fn = file_names[idx]
            mask = decode_rle(rle, height, width)
            frame_masks[fn].append((mask, categories[category_id], score, category_colors[category_id]))

        # Check if any frames have masks (only save videos with predictions)
        has_masks = any(len(masks) > 0 for masks in frame_masks.values())
        if not has_masks:
            skipped_videos.append((folder_name, "No masks to visualize after processing"))
            print(f"  Skipping video {video_id} - no masks to visualize")
            continue

        output_video_path = os.path.join(output_dir, f"{folder_name}.mp4")
        fps = 10
        out = None

        for fn in file_names:
            img_path = os.path.join(image_root, fn)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: could not read image {img_path}")
                continue

            # Draw true species in top right corner
            draw_true_species(frame, true_species)

            for mask, label, scr, color in frame_masks.get(fn, []):
                draw_label_and_mask(frame, mask, label, scr, color)

            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

            out.write(frame)

        if out:
            out.release()
            print(f"  Saved video to {output_video_path}")

    # Track videos with different types of prediction issues
    videos_with_no_predictions = []
    videos_below_threshold = []
    
    all_video_ids = set(video_info.keys())
    videos_with_any_predictions = set(all_preds_by_video.keys())
    videos_with_predictions_above_thresh = set(preds_by_video.keys())
    
    # Videos with no predictions at all
    videos_without_any_predictions = all_video_ids - videos_with_any_predictions
    
    # Videos with predictions but none above threshold
    videos_with_predictions_below_thresh = videos_with_any_predictions - videos_with_predictions_above_thresh
    
    for video_id in videos_without_any_predictions:
        if video_id in video_info:
            folder_name = os.path.dirname(video_info[video_id]["file_names"][0])
            videos_with_no_predictions.append((folder_name, "No predictions at all"))
    
    for video_id in videos_with_predictions_below_thresh:
        if video_id in video_info:
            folder_name = os.path.dirname(video_info[video_id]["file_names"][0])
            max_score = max(pred["score"] for pred in all_preds_by_video[video_id])
            videos_below_threshold.append((folder_name, f"No predictions above threshold {score_thresh} (max score: {max_score:.4f})"))
    
    print(f"\nProcessing complete. Videos saved to: {output_dir}")
    
    # Print summary of skipped videos
    all_skipped = skipped_videos + videos_with_no_predictions + videos_below_threshold
    if all_skipped:
        print(f"\n{len(all_skipped)} videos were not saved:")
        print("-" * 80)
        for video_name, reason in all_skipped:
            print(f"  {video_name}: {reason}")
        print("-" * 80)
        
        # Print breakdown
        print(f"\nBreakdown:")
        print(f"  Videos with no predictions at all: {len(videos_with_no_predictions)}")
        print(f"  Videos with predictions below threshold: {len(videos_below_threshold)}")
        print(f"  Videos with processing issues: {len(skipped_videos)}")
        print(f"  Total videos in validation set: {len(all_video_ids)}")
        print(f"  Videos with any predictions: {len(videos_with_any_predictions)}")
        print(f"  Videos successfully saved: {len(videos_with_any_predictions) - len(videos_below_threshold) - len([v for v in skipped_videos if 'No masks' in v[1]])}")
    else:
        print("\nAll videos with predictions were successfully processed and saved.")

if __name__ == "__main__":
    results_json = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/attention/fold6_4443_attn_extra/inference/results.json"
    valid_json = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/attention/fold6_4443_attn_extra/val_fold6_all_frames.json"
    image_root = "/data/fishway_ytvis/all_videos_mask"
    output_dir = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/attention/fold6_4443_attn_extra/inference/video_predictions"

    visualize_predictions(results_json, valid_json, image_root, output_dir, score_thresh=0)
