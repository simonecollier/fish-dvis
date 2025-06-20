import os
import json
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import random

def decode_rle(rle):
    """
    Decode RLE mask encoding, handling empty masks.
    """
    if not isinstance(rle, dict) or not rle:
        return np.array([])
    
    # Check for empty counts
    if not rle.get('counts', []):
        return np.array([])
    
    decoded_mask = mask_utils.decode({
        'size': rle['size'],
        'counts': bytes(rle['counts']) if isinstance(rle['counts'][0], int) else rle['counts']
    })
    return decoded_mask

def convert_all_segmentations_to_compressed_rle(ytvis_json):
    for ann in ytvis_json["annotations"]:
        segs = ann.get("segmentations", [])
        for i, seg in enumerate(segs):
            if isinstance(seg, dict) and isinstance(seg.get("counts", None), list):
                # try:
                #     # Try to decode the mask (works if counts is valid uncompressed RLE)
                #     mask = mask_utils.decode(seg)
                # except Exception as e:
                #     print(f"Failed to decode RLE: {e}")
                #     continue
                # Re-encode as compressed RLE
                mask = decode_rle(seg)
                compressed = mask_utils.encode(np.asfortranarray(mask))
                compressed["counts"] = compressed["counts"].decode("ascii")
                segs[i] = compressed
        ann["segmentations"] = segs
    return ytvis_json

def convert_coco_to_ytvis(
    df,
    base_data_dir,
    output_dir,
    final_json_path
):
    # Step 2: Create output directories
    os.makedirs(output_dir, exist_ok=True)
    image_out_dir = os.path.join(output_dir)
    os.makedirs(image_out_dir, exist_ok=True)

    # Step 3: Initialize YTVIS json components
    ytvis = {
        "videos": [],
        "annotations": [],
        "categories": [],
        "info": {},
        "licenses": []
    }

    video_id_counter = 1
    annotation_id_counter = 1
    video_name_to_id = {}
    track_to_anns = defaultdict(list)

    first_categories_loaded = False
    used_cat_ids = set()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        annotator = row["Mask Annotator Name"].strip().lower()
        video_folder = row["Fish Computer Folder"]
        rel_path = os.path.join(annotator, video_folder)
        video_path = os.path.join(base_data_dir, rel_path)
        ann_path = os.path.join(video_path, "annotations", "edited_instances_default.json")
        img_path = os.path.join(video_path, "images")

        if not (os.path.exists(ann_path) and os.path.exists(img_path)):
            print(f"Skipping inaccessible: {ann_path}")
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        if not first_categories_loaded:
            ytvis["categories"] = coco["categories"]
            first_categories_loaded = True

        images = sorted(coco["images"], key=lambda x: x["file_name"])
        image_filenames = []
        for img in images:
            src_img = os.path.join(img_path, img["file_name"])
            dst_img = os.path.join(image_out_dir, f"{video_folder}/{img['file_name']}")
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.copy2(src_img, dst_img)
            image_filenames.append(f"{video_folder}/{img['file_name']}")

        video_id = video_id_counter
        video_id_counter += 1
        video_name_to_id[video_folder] = video_id

        # --- TRIMMING LOGIC STARTS HERE ---
        # Group annotations by track_id
        num_frames = len(images)
        image_id_to_idx = {img["id"]: i for i, img in enumerate(images)}
        frame_has_ann = [False] * num_frames
        for ann in coco["annotations"]:
            frame_idx = image_id_to_idx[ann["image_id"]]
            frame_has_ann[frame_idx] = True

        # Find first and last frame with annotation
        try:
            first_anno = frame_has_ann.index(True)
            last_anno = len(frame_has_ann) - 1 - frame_has_ann[::-1].index(True)
        except ValueError:
            # No annotation in this video, skip it
            print(f"Skipping video {video_folder}: no annotated frames.")
            continue

        # Trim images and filenames
        images = images[first_anno:last_anno+1]
        image_filenames = image_filenames[first_anno:last_anno+1]
        num_frames_trimmed = len(images)
        image_id_to_idx_trimmed = {img["id"]: i for i, img in enumerate(images)}

        # --- TRIMMING LOGIC ENDS HERE ---

        ytvis["videos"].append({
            "id": video_id,
            "file_names": image_filenames,
            "width": images[0]["width"],
            "height": images[0]["height"],
            "length": num_frames_trimmed,
            "coco_url": "",
            "flickr_url": "",
            "date_captured": ""
        })

        # Group annotations by track_id for trimmed frames
        track_anns = defaultdict(lambda: {"segmentations": [None]*num_frames_trimmed, "bboxes": [None]*num_frames_trimmed, "areas": [None]*num_frames_trimmed, "category_id": None})
        for ann in coco["annotations"]:
            tid = ann["attributes"]["track_id"]
            orig_frame_idx = image_id_to_idx[ann["image_id"]]
            trimmed_frame_idx = orig_frame_idx - first_anno
            if 0 <= trimmed_frame_idx < num_frames_trimmed:
                track_anns[tid]["segmentations"][trimmed_frame_idx] = ann["segmentation"]
                track_anns[tid]["bboxes"][trimmed_frame_idx] = ann.get("bbox", None)
                track_anns[tid]["areas"][trimmed_frame_idx] = ann.get("area", None)
                track_anns[tid]["category_id"] = ann["category_id"]
                track_anns[tid]["iscrowd"] = ann["iscrowd"]

        for track_id, info in track_anns.items():
            ytvis["annotations"].append({
                "id": annotation_id_counter,
                "video_id": video_id,
                "category_id": info["category_id"],
                "height": images[0]["height"],  # required
                "width": images[0]["width"],    # required
                "length": num_frames_trimmed,    # required
                "score": 1.0,  # GT annotations
                "iscrowd": info["iscrowd"],
                "segmentations": info["segmentations"],
                "bboxes": info["bboxes"],
                "areas": info["areas"],        
            })
            used_cat_ids.add(info["category_id"])
            annotation_id_counter += 1

    ytvis = convert_all_segmentations_to_compressed_rle(ytvis)
    # Filter categories to only those present in the annotations
    ytvis["categories"] = [cat for cat in ytvis["categories"] if cat["id"] in used_cat_ids]
    # Save final YTVIS JSON
    with open(final_json_path, "w") as f:
        json.dump(ytvis, f, indent=2)
    print(f"Saved YTVIS JSON to {final_json_path}")


def validate_ytvis(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Map video_id to its length
    video_lengths = {video["id"]: video["length"] for video in data["videos"]}
    errors_found = False

    for ann in data["annotations"]:
        video_id = ann["video_id"]
        expected_len = video_lengths.get(video_id)

        for field in ["segmentations", "bboxes", "areas"]:
            values = ann.get(field, [])
            if len(values) != expected_len:
                print(f"Error: Annotation ID {ann['id']} has mismatched {field} length: {len(values)} vs expected {expected_len}")
                errors_found = True

        # Optional: Check segmentation format
        for idx, seg in enumerate(ann.get("segmentations", [])):
            if seg is not None:
                if isinstance(seg, dict):
                    if "counts" not in seg or "size" not in seg:
                        print(f"Warning: Annotation ID {ann['id']} frame {idx} has invalid RLE segmentation.")
                        errors_found = True
                elif not isinstance(seg, list):
                    print(f"Warning: Annotation ID {ann['id']} frame {idx} has segmentation of unexpected type: {type(seg)}")
                    errors_found = True

    if not errors_found:
        print("✅ JSON passed validation.")
    else:
        print("❌ Errors found in JSON.")


csv_path = "/home/simone/fish-dvis/data_scripts/fishway_metadata.csv"

# Step 1: Load and filter
df = pd.read_csv(csv_path)
df = df[
    (df["mask annotated"] == 1) &
    (df["Mask Annotator Name"].isin(["Simone", "Bushra"])) &
    (df["train/val/test"] == "train")
]

# --- New logic for val split: two random videos per present category ---
video_to_categories = defaultdict(set)
present_categories = set()

for _, row in tqdm(df.iterrows(), total=len(df)):
    annotator = row["Mask Annotator Name"].strip().lower()
    video_folder = row["Fish Computer Folder"]
    rel_path = os.path.join(annotator, video_folder)
    video_path = os.path.join("/data/labeled", rel_path)
    ann_path = os.path.join(video_path, "annotations", "edited_instances_default.json")
    if not os.path.exists(ann_path):
        continue
    with open(ann_path) as f:
        coco = json.load(f)
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        present_categories.add(cat_id)
        video_to_categories[video_folder].add(cat_id)

val_videos = set()
for cat_id in present_categories:
    videos_with_cat = [v for v, cats in video_to_categories.items() if cat_id in cats]
    num_to_select = min(2, len(videos_with_cat))
    if num_to_select > 0:
        val_videos.update(random.sample(videos_with_cat, num_to_select))

val_df = df[df["Fish Computer Folder"].isin(val_videos)]
train_df = df[~df["Fish Computer Folder"].isin(val_videos)]

# Step 3: Write to separate YTVIS JSONs
convert_coco_to_ytvis(
    train_df,
    base_data_dir="/data/labeled",
    output_dir="/data/fishway_ytvis/train",
    final_json_path="/data/fishway_ytvis/train.json"
)

convert_coco_to_ytvis(
    val_df,
    base_data_dir="/data/labeled",
    output_dir="/data/fishway_ytvis/val",
    final_json_path="/data/fishway_ytvis/val.json"
)

# Run this on your train/val json
validate_ytvis("/data/fishway_ytvis/train.json")
validate_ytvis("/data/fishway_ytvis/val.json")

def create_tiny_overfit_jsons(train_json_path, val_json_path, out_train_json, out_val_json, category_id=2):
    """
    Create tiny YTVIS JSONs for overfitting:
    - 2 videos with category_id == 2 for train
    - 1 video with category_id == 2 for val
    - Only category 2 in categories
    """
    import copy
    for src_path, out_path, num_videos in [
        (train_json_path, out_train_json, 2),
        (val_json_path, out_val_json, 1)
    ]:
        with open(src_path, 'r') as f:
            data = json.load(f)
        # Find all videos with at least one annotation of category_id
        vids_with_cat = set(
            ann['video_id'] for ann in data['annotations'] if ann['category_id'] == category_id
        )
        vids_with_cat = list(vids_with_cat)
        vids_with_cat = vids_with_cat[:num_videos]
        # Filter videos
        videos = [v for v in data['videos'] if v['id'] in vids_with_cat]
        # Filter annotations
        anns = [ann for ann in data['annotations'] if ann['video_id'] in vids_with_cat and ann['category_id'] == category_id]
        # Find the category info for category_id
        cat_info = [cat for cat in data['categories'] if cat['id'] == category_id]
        # Build new json
        new_json = {
            'videos': videos,
            'annotations': anns,
            'categories': cat_info,
            'info': data.get('info', {}),
            'licenses': data.get('licenses', [])
        }
        with open(out_path, 'w') as f:
            json.dump(new_json, f, indent=2)
        print(f"Wrote tiny overfit json to {out_path}")

# create_tiny_overfit_jsons(train_json_path="/data/fishway_ytvis/train.json", 
#                           val_json_path="/data/fishway_ytvis/val.json", 
#                           out_train_json="/data/fishway_ytvis/train_tiny.json", 
#                           out_val_json="/data/fishway_ytvis/val_tiny.json", 
#                           category_id=2)