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
    metadata_csv_path,
    base_data_dir,
    output_dir,
    final_json_path
):
    # Step 1: Load and filter metadata
    df = pd.read_csv(metadata_csv_path)
    df = df[
        (df["mask annotated"] == 1) &
        (df["Mask Annotator Name"].isin(["Simone", "Bushra"]))
    ]

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

        num_frames = len(images)
        image_id_to_idx = {img["id"]: i for i, img in enumerate(images)}

        ytvis["videos"].append({
            "id": video_id,
            "file_names": image_filenames,
            "width": images[0]["width"],
            "height": images[0]["height"],
            "length": num_frames,
            "coco_url": "",
            "flickr_url": "",
            "date_captured": ""
        })

        # Group annotations by track_id for all frames (no trimming)
        track_anns = defaultdict(lambda: {"segmentations": [None]*num_frames, "bboxes": [None]*num_frames, "areas": [None]*num_frames, "category_id": None})
        for ann in coco["annotations"]:
            tid = ann["attributes"]["track_id"]
            frame_idx = image_id_to_idx[ann["image_id"]]
            track_anns[tid]["segmentations"][frame_idx] = ann["segmentation"]
            track_anns[tid]["bboxes"][frame_idx] = ann.get("bbox", None)
            track_anns[tid]["areas"][frame_idx] = ann.get("area", None)
            track_anns[tid]["category_id"] = ann["category_id"]
            track_anns[tid]["iscrowd"] = ann["iscrowd"]

        for track_id, info in track_anns.items():
            ytvis["annotations"].append({
                "id": annotation_id_counter,
                "video_id": video_id,
                "category_id": info["category_id"],
                "height": images[0]["height"],  # required
                "width": images[0]["width"],    # required
                "length": num_frames,    # required
                "score": 1.0,  # GT annotations
                "iscrowd": info["iscrowd"],
                "occ_score": 0.0,
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


def create_train_val_jsons(
    all_vids_json,
    output_train_json,
    output_val_json,
    categories_to_include=["Chinook", "Coho", "Brown Trout", "Atlantic"],
    exclude_empty_train=True,
    balance=True,
    num_vids_val=2,
    num_vids_train=None,
    frame_skip=0,
    random_seed=None
):
    """
    Create train and val YTVIS JSONs from a big all-videos YTVIS JSON.
    - categories_to_include: list of category names to include
    - exclude_empty_train: if True, trim empty frames from train set
    - balance: if True, balance number of videos per class
    - num_vids_val: number of videos per class for val set
    - num_vids_train: number of videos per class for train set (None = as many as possible)
    - frame_skip: 0 = include every frame, 1 = every 2nd frame, 2 = every 3rd frame, etc.
    - random_seed: if set, ensures reproducible splits
    """
    import copy
    if random_seed is not None:
        random.seed(random_seed)
    with open(all_vids_json, 'r') as f:
        data = json.load(f)
    # Map category names to ids
    cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    cat_ids = [cat_name_to_id[name] for name in categories_to_include if name in cat_name_to_id]
    # Find videos for each category
    video_to_cats = defaultdict(set)
    for ann in data['annotations']:
        if ann['category_id'] in cat_ids:
            video_to_cats[ann['video_id']].add(ann['category_id'])
    # Build val set
    val_videos = set()
    for cat_id in cat_ids:
        vids = [vid for vid, cats in video_to_cats.items() if cat_id in cats]
        vids = list(set(vids) - val_videos)  # avoid overlap
        num = min(num_vids_val, len(vids))
        if balance:
            vids = random.sample(vids, num) if num > 0 else []
        else:
            vids = vids[:num]
        val_videos.update(vids)
    # Build train set
    train_videos = set([vid for vid in video_to_cats if vid not in val_videos])
    if balance:
        # For each class, sample up to num_vids_train (or min count if None)
        vids_per_cat = {cat_id: [vid for vid in train_videos if cat_id in video_to_cats[vid]] for cat_id in cat_ids}
        if num_vids_train is None:
            min_count = min(len(v) for v in vids_per_cat.values())
        else:
            min_count = num_vids_train
        balanced_train_videos = set()
        for cat_id, vids in vids_per_cat.items():
            n = min(min_count, len(vids))
            if n > 0:
                balanced_train_videos.update(random.sample(vids, n))
        train_videos = balanced_train_videos
    # Filter videos/annotations for train/val
    def filter_json(videoset, trim_empty):
        vids = [v for v in data['videos'] if v['id'] in videoset]
        anns = [a for a in data['annotations'] if a['video_id'] in videoset and a['category_id'] in cat_ids]
        # Optionally trim empty frames (for train)
        if trim_empty:
            # For each video, find frames with at least one annotation
            vid_to_frames_with_ann = defaultdict(set)
            for ann in anns:
                for idx, seg in enumerate(ann['segmentations']):
                    if seg is not None:
                        vid_to_frames_with_ann[ann['video_id']].add(idx)
            # Trim frames for each video
            new_vids = []
            new_anns = []
            for v in vids:
                frames_with_ann = vid_to_frames_with_ann[v['id']]
                if not frames_with_ann:
                    continue
                first = min(frames_with_ann)
                last = max(frames_with_ann)
                # Trim file_names and update length
                v_new = copy.deepcopy(v)
                v_new['file_names'] = v_new['file_names'][first:last+1]
                v_new['length'] = len(v_new['file_names'])
                new_vids.append(v_new)
                # Trim segmentations/bboxes/areas for each annotation
                for ann in [a for a in anns if a['video_id'] == v['id']]:
                    ann_new = copy.deepcopy(ann)
                    ann_new['segmentations'] = ann_new['segmentations'][first:last+1]
                    ann_new['bboxes'] = ann_new['bboxes'][first:last+1]
                    ann_new['areas'] = ann_new['areas'][first:last+1]
                    ann_new['length'] = len(ann_new['segmentations'])
                    new_anns.append(ann_new)
            vids = new_vids
            anns = new_anns
        # Apply frame skipping
        if frame_skip > 0:
            def skip_frames(seq):
                return seq[::frame_skip+1]
            # For videos
            for v in vids:
                v['file_names'] = skip_frames(v['file_names'])
                v['length'] = len(v['file_names'])
            # For annotations
            for ann in anns:
                ann['segmentations'] = skip_frames(ann['segmentations'])
                ann['bboxes'] = skip_frames(ann['bboxes'])
                ann['areas'] = skip_frames(ann['areas'])
                ann['length'] = len(ann['segmentations'])
        return vids, anns
    train_vids, train_anns = filter_json(train_videos, trim_empty=exclude_empty_train)
    val_vids, val_anns = filter_json(val_videos, trim_empty=False)
    # Build output jsons
    def build_json(vids, anns):
        used_cat_ids = set(a['category_id'] for a in anns)
        cats = [cat for cat in data['categories'] if cat['id'] in used_cat_ids]
        return {
            'videos': vids,
            'annotations': anns,
            'categories': cats,
            'info': data.get('info', {}),
            'licenses': data.get('licenses', [])
        }
    with open(output_train_json, 'w') as f:
        json.dump(build_json(train_vids, train_anns), f, indent=2)
    with open(output_val_json, 'w') as f:
        json.dump(build_json(val_vids, val_anns), f, indent=2)
    print(f"Wrote train json to {output_train_json}")
    print(f"Wrote val json to {output_val_json}")


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


if __name__ == "__main__":
    convert_coco_to_ytvis(
        metadata_csv_path="/home/simone/fish-dvis/data_scripts/fishway_metadata.csv",
        base_data_dir="/data/labeled",
        output_dir="/data/fishway_ytvis/all_videos",
        final_json_path="/data/fishway_ytvis/all_videos.json"
    )

    create_train_val_jsons(
        all_vids_json="/data/fishway_ytvis/all_videos.json",
        output_train_json="/data/fishway_ytvis/train.json",
        output_val_json="/data/fishway_ytvis/val.json",
        categories_to_include=["Chinook", "Coho", "Brown Trout", "Atlantic"],
        exclude_empty_train=True,
        balance=True,
        num_vids_val=2,
        num_vids_train=None,
        frame_skip=3
    )

    validate_ytvis("/data/fishway_ytvis/all_videos.json")
    validate_ytvis("/data/fishway_ytvis/train.json")
    validate_ytvis("/data/fishway_ytvis/val.json")
