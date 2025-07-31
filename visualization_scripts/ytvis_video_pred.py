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
    cv2.drawContours(frame, contours, -1, color, thickness=2)
    
    # Get label position from the first contour
    if contours:
        x, y, _, _ = cv2.boundingRect(contours[0])
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
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

    # Choose color per category
    np.random.seed(42)
    category_colors = {cat_id: tuple(np.random.randint(0, 256, 3).tolist()) for cat_id in categories}

    # Group predictions by video_id and filter by score threshold
    preds_by_video = {}
    for pred in predictions:
        if pred["score"] >= score_thresh:
            preds_by_video.setdefault(pred["video_id"], []).append(pred)

    print(f"Found {len(preds_by_video)} videos with predictions above threshold {score_thresh}")
    
    # Process each video that has predictions
    for video_id, video_preds in tqdm(preds_by_video.items(), desc="Processing videos"):
        if video_id not in video_info:
            print(f"Warning: Video id {video_id} not found in valid_json, skipping.")
            continue

        video = video_info[video_id]
        file_names = video["file_names"]
        height = video["height"]
        width = video["width"]
        folder_name = os.path.dirname(file_names[0])

        # Get true species for this video
        true_species = true_species_by_video.get(video_id, "Unknown")
        
        # Print predictions for this video
        print(f"\nVideo {video_id} ({folder_name}):")
        print(f"  True species: {true_species}")
        for pred in video_preds:
            category_name = categories[pred["category_id"]]
            score = pred["score"]
            print(f"  - {category_name}: {score:.4f}")

        # For each frame, collect all masks for all objects
        frame_masks = {fn: [] for fn in file_names}
        for pred in video_preds:
            category_id = pred["category_id"]
            segmentations = pred["segmentations"]
            score = pred["score"]
            for idx, rle in enumerate(segmentations):
                if idx >= len(file_names):
                    continue
                fn = file_names[idx]
                mask = decode_rle(rle, height, width)
                frame_masks[fn].append((mask, categories[category_id], score, category_colors[category_id]))

        # Check if any frames have masks (only save videos with predictions)
        has_masks = any(len(masks) > 0 for masks in frame_masks.values())
        if not has_masks:
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

    print(f"\nProcessing complete. Videos saved to: {output_dir}")

if __name__ == "__main__":
    results_json = "/store/simone/dvis-model-outputs/trained_models/enhanced_augmentations_0.0001_15f/inference/results.json"
    valid_json = "/data/fishway_ytvis/val.json"
    image_root = "/data/fishway_ytvis/all_videos"
    output_dir = "/store/simone/dvis-model-outputs/trained_models/enhanced_augmentations_0.0001_15f/inference/video_predictions"

    visualize_predictions(results_json, valid_json, image_root, output_dir, score_thresh=0.01)
