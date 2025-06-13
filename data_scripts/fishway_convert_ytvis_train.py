import os
import json
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def convert_coco_to_ytvis(
    train_or_val_df,
    base_data_dir,
    output_dir,
    final_json_path
):
    df = train_or_val_df

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

        for track_id, info in track_anns.items():
            ytvis["annotations"].append({
                "id": annotation_id_counter,
                "video_id": video_id,
                "category_id": info["category_id"],
                "segmentations": info["segmentations"],
                "bboxes": info["bboxes"],
                "areas": info["areas"],
                "score": 1.0,  # GT annotations
                "height": images[0]["height"],  # required
                "width": images[0]["width"],    # required
                "length": num_frames_trimmed    # required
            })
            annotation_id_counter += 1

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


csv_path = "/home/simone/LO_Fishway_Labelling/fishway_metadata.csv"

# Step 1: Load and filter
df = pd.read_csv(csv_path)
df = df[
    (df["mask annotated"] == 1) &
    (df["Mask Annotator Name"].isin(["Simone", "Bushra"])) &
    (df["train/val/test"] == "train")
]

# Step 2: Split into train/val (e.g., 80/20 split by video)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 3: Write to separate YTVIS JSONs
convert_coco_to_ytvis(
    train_df,
    base_data_dir="/data/labeled",
    output_dir="/home/simone/shared-data/fishway_ytvis/train",
    final_json_path="/home/simone/shared-data/fishway_ytvis/train.json"
)

convert_coco_to_ytvis(
    val_df,
    base_data_dir="/data/labeled",
    output_dir="/home/simone/shared-data/fishway_ytvis/val",
    final_json_path="/home/simone/shared-data/fishway_ytvis/val.json"
)

# Run this on your train/val json
validate_ytvis("/data/fishway_ytvis/train.json")
validate_ytvis("/data/fishway_ytvis/val.json")