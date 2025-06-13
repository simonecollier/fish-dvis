import json
import pandas as pd
from pathlib import Path

# with open('/home/simone/shared-data/YTVIS2021/train/instances.json', 'r') as f:
#     data = json.load(f)

# with open('/home/simone/shared-data/YTVIS2021/train/instances_indented.json', 'w') as f:
#     json.dump(data, f, indent=4)

# try:
#     json.load(open('/home/simone/shared-data/YTVIS2021/train/instances.json'))
# except json.JSONDecodeError as e:
#     print(e)

# input_path = '/home/simone/dvis-daq-outputs/ytvis21_eval/inference/results.json'
# output_path = '/home/simone/dvis-daq-outputs/ytvis21_eval/inference/results_summary.json'

# with open(input_path, "r") as f:
#     data = json.load(f)

# summary = {}

# for key, value in data.items():
#     if key in ["categories", "info", "licenses"]:
#         summary[key] = value  # keep all entries
#     elif isinstance(value, list):
#         summary[key] = value[:5]  # first 3 items in list
#     elif isinstance(value, dict):
#         summary[key] = dict(list(value.items())[:5])  # first 3 key-value pairs
#     else:
#         summary[key] = value  # scalar values

# with open(output_path, "w") as f:
#     json.dump(summary, f, indent=2)

# print(f"Summary saved to {output_path}")

import json
import pandas as pd
from pathlib import Path

def get_annotated_video_paths(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    df = df[df["mask annotated"] == 1]
    df = df[df["Mask Annotator Name"].isin(["Simone", "Bushra"])]

    selected_paths = []
    for _, row in df.iterrows():
        annotator = row["Mask Annotator Name"].lower()
        folder = row["Fish Computer Folder"]
        full_path = Path(base_dir) / annotator / folder
        if full_path.exists():
            selected_paths.append((annotator, folder, full_path))
        else:
            print(f"Warning: Missing folder {full_path}")
    return selected_paths

labelling_spreadsheet = "/home/simone/LO_Fishway_Labelling/Fishway Data Labelling Spreadsheet.csv"
data_dir = "/data/labeled"
labeled_video_paths = get_annotated_video_paths(csv_path=labelling_spreadsheet, base_dir=data_dir)
# print(labeled_video_paths)
# labeled_video_paths[1][2].exists()

import json
import shutil
import os

def convert_to_ytvis_format(video_entries, output_dir):
    ytvis = {
        "videos": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    video_id = 0
    annotation_id = 0
    image_id = 0
    categories_set = set()

    for annotator, folder, full_path in video_entries:
        ann_path = full_path / "annotations" / "edited_instances_default.json"
        img_dir = full_path / "images"
        if not ann_path.exists():
            print(f"Missing annotation file for {folder}")
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        # Copy images to a flat directory: {output_dir}/images/{video_name}/
        target_img_dir = Path(output_dir) / "images" / folder
        target_img_dir.mkdir(parents=True, exist_ok=True)

        frame_filenames = {}
        for img in coco["images"]:
            frame_file = img["file_name"]
            src = img_dir / frame_file
            dst = target_img_dir / frame_file
            shutil.copy2(src, dst)

            ytvis["images"].append({
                "file_name": f"{folder}/{frame_file}",
                "height": img["height"],
                "width": img["width"],
                "id": image_id,
                "video_id": video_id,
                "frame_id": int(Path(frame_file).stem),
            })
            frame_filenames[img["id"]] = image_id
            image_id += 1

        for ann in coco["annotations"]:
            track_id = ann.get("track_id", ann["id"])
            ytvis["annotations"].append({
                "id": annotation_id,
                "category_id": ann["category_id"],
                "video_id": video_id,
                "segmentations": [ann["segmentation"]],
                "iscrowd": ann.get("iscrowd", 0),
                "area": ann.get("area", 0),
                "track_id": track_id,
                "image_ids": [frame_filenames[ann["image_id"]]],
            })
            annotation_id += 1

        ytvis["videos"].append({
            "id": video_id,
            "file_names": [f"{folder}/{img['file_name']}" for img in coco["images"]],
            "length": len(coco["images"]),
            "width": coco["images"][0]["width"],
            "height": coco["images"][0]["height"],
        })
        video_id += 1

        # Collect categories
        for cat in coco["categories"]:
            categories_set.add((cat["id"], cat["name"]))

    ytvis["categories"] = [{"id": cid, "name": name} for cid, name in sorted(categories_set)]

    out_ann_file = Path(output_dir) / "annotations" / "instances_train.json"
    out_ann_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ann_file, "w") as f:
        json.dump(ytvis, f)
    
    print(f"Saved YTVIS JSON to {out_ann_file}")
    return str(out_ann_file)
