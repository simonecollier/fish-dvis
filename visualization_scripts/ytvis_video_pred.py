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

def hex_to_bgr(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def visualize_predictions(results_json, valid_json, image_root, output_dir, score_thresh=0.5, selected_video_id=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(results_json) as f:
        predictions = json.load(f)

    with open(valid_json) as f:
        valid_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in valid_data["categories"]}
    video_info = {v["id"]: v for v in valid_data["videos"]}

    # Choose color per category
    np.random.seed(42)
    category_colors = {cat_id: tuple(np.random.randint(0, 256, 3).tolist()) for cat_id in categories}

    # Group predictions by video_id
    preds_by_video = {}
    for pred in predictions:
        if pred["score"] < score_thresh:
            continue
        preds_by_video.setdefault(pred["video_id"], []).append(pred)

    # If no video_id specified, just pick the first one
    if selected_video_id is None:
        selected_video_id = next(iter(preds_by_video.keys()))

    if selected_video_id not in video_info:
        print(f"Video id {selected_video_id} not found in valid_json.")
        return

    video = video_info[selected_video_id]
    file_names = video["file_names"]
    height = video["height"]
    width = video["width"]
    folder_name = os.path.dirname(file_names[0])

    # For each frame, collect all masks for all objects
    frame_masks = {fn: [] for fn in file_names}
    for pred in preds_by_video[selected_video_id]:
        category_id = pred["category_id"]
        segmentations = pred["segmentations"]
        score = pred["score"]
        for idx, rle in enumerate(segmentations):
            if idx >= len(file_names):
                continue
            fn = file_names[idx]
            mask = decode_rle(rle, height, width)
            frame_masks[fn].append((mask, categories[category_id], score, category_colors[category_id]))

    output_video_path = os.path.join(output_dir, f"{folder_name}.mp4")
    fps = 10
    out = None

    for fn in file_names:
        img_path = os.path.join(image_root, fn)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: could not read image {img_path}")
            continue

        for mask, label, scr, color in frame_masks.get(fn, []):
            draw_label_and_mask(frame, mask, label, scr, color)

        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

        out.write(frame)

    if out:
        out.release()
    print(f"Saved video to {output_video_path}")

# Usage example:
# if __name__ == "__main__":
#     results_json = "/home/simone/dvis-daq-outputs/LOMU_evalfull/inference/results.json"
#     valid_json = "/data/ytvis_2021/valid.json"
#     image_root = "/data/ytvis_2021/valid/JPEGImages"
#     output_dir = "/home/simone/dvis-daq-outputs/LOMU_eval_full/inference/video_predictions"

#     #selected_video_id = "fd2e11b168"  # <-- set your desired video ID here
#     visualize_predictions(results_json, valid_json, image_root, output_dir, score_thresh=0.5, selected_video_id=1)

if __name__ == "__main__":
    results_json = "//home/simone/dvis-daq-outputs/Fishway_coho_chinook_eval/inference/results.json"
    valid_json = "/data/fishway_ytvis/val.json"
    image_root = "/data/fishway_ytvis/val"
    output_dir = "/home/simone/dvis-daq-outputs/Fishway_coho_chinook_eval/inference/video_predictions"

    #selected_video_id = "fd2e11b168"  # <-- set your desired video ID here
    visualize_predictions(results_json, valid_json, image_root, output_dir, score_thresh=0.5, selected_video_id=2)
