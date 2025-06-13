import os
import json
from PIL import Image, ImageOps
from datetime import datetime

# input_folder = "/home/simone/shared-data/ytvis_2021/valid/JPEGImages/BruceFish"
# output_folder = "/home/simone/shared-data/ytvis_2021/valid/JPEGImages/BruceFish_downsampled"
# target_size = (1280, 720)  # 16:9 resolution

# os.makedirs(output_folder, exist_ok=True)

# for filename in sorted(os.listdir(input_folder)):
#     if filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)

#         with Image.open(input_path) as img:
#             img = ImageOps.pad(img, target_size, method=Image.BILINEAR, color=(0, 0, 0))
#             img.save(output_path)

# print("✅ All frames resized and padded to 1280×720.")

def generate_valid_json(frames_root_dir, output_json):
    info = {
        "description": "YouTube-VOS", 
        "url": "https://youtube-vos.org/home", 
        "version": "2.0", 
        "year": 2021, 
        "contributor": "ychfan", 
        "date_created": "2021-02-12 19:17:02.937750"
        }, 
    licenses = [{
        "url": "https://creativecommons.org/licenses/by/4.0/", 
        "id": 1, 
        "name": "Creative Commons Attribution 4.0 License"}]
    categories = [
        {"supercategory": "object", "id": 1, "name": "airplane"}, 
        {"supercategory": "object", "id": 2, "name": "bear"}, 
        {"supercategory": "object", "id": 3, "name": "bird"}, 
        {"supercategory": "object", "id": 4, "name": "boat"}, 
        {"supercategory": "object", "id": 5, "name": "car"}, 
        {"supercategory": "object", "id": 6, "name": "cat"}, 
        {"supercategory": "object", "id": 7, "name": "cow"}, 
        {"supercategory": "object", "id": 8, "name": "deer"}, 
        {"supercategory": "object", "id": 9, "name": "dog"}, 
        {"supercategory": "object", "id": 10, "name": "duck"},
        {"supercategory": "object", "id": 11, "name": "earless_seal"}, 
        {"supercategory": "object", "id": 12, "name": "elephant"}, 
        {"supercategory": "object", "id": 13, "name": "fish"}, 
        {"supercategory": "object", "id": 14, "name": "flying_disc"}, 
        {"supercategory": "object", "id": 15, "name": "fox"}, 
        {"supercategory": "object", "id": 16, "name": "frog"}, 
        {"supercategory": "object", "id": 17, "name": "giant_panda"}, 
        {"supercategory": "object", "id": 18, "name": "giraffe"}, 
        {"supercategory": "object", "id": 19, "name": "horse"}, 
        {"supercategory": "object", "id": 20, "name": "leopard"}, 
        {"supercategory": "object", "id": 21, "name": "lizard"}, 
        {"supercategory": "object", "id": 22, "name": "monkey"}, 
        {"supercategory": "object", "id": 23, "name": "motorbike"}, 
        {"supercategory": "object", "id": 24, "name": "mouse"}, 
        {"supercategory": "object", "id": 25, "name": "parrot"}, 
        {"supercategory": "object", "id": 26, "name": "person"}, 
        {"supercategory": "object", "id": 27, "name": "rabbit"}, 
        {"supercategory": "object", "id": 28, "name": "shark"}, 
        {"supercategory": "object", "id": 29, "name": "skateboard"}, 
        {"supercategory": "object", "id": 30, "name": "snake"}, 
        {"supercategory": "object", "id": 31, "name": "snowboard"}, 
        {"supercategory": "object", "id": 32, "name": "squirrel"}, 
        {"supercategory": "object", "id": 33, "name": "surfboard"}, 
        {"supercategory": "object", "id": 34, "name": "tennis_racket"}, 
        {"supercategory": "object", "id": 35, "name": "tiger"}, 
        {"supercategory": "object", "id": 36, "name": "train"}, 
        {"supercategory": "object", "id": 37, "name": "truck"}, 
        {"supercategory": "object", "id": 38, "name": "turtle"}, 
        {"supercategory": "object", "id": 39, "name": "whale"}, 
        {"supercategory": "object", "id": 40, "name": "zebra"}
        ] 
    
    json_dict = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "videos": []
    }

    video_folders = sorted([
        d for d in os.listdir(frames_root_dir)
        if os.path.isdir(os.path.join(frames_root_dir, d))
    ])

    video_id = 1

    for video_folder in video_folders:
        folder_path = os.path.join(frames_root_dir, video_folder)
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not image_files:
            continue

        # Use first image to get resolution
        first_image_path = os.path.join(folder_path, image_files[0])
        with Image.open(first_image_path) as img:
            width, height = img.size

        file_names = [os.path.join(video_folder, f) for f in image_files]

        video_entry = {
            "license": 1,
            "coco_url": "",
            "height": height,
            "width": width,
            "length": len(file_names),
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "flickr_url": "",
            "file_names": file_names,
            "id": video_id
        }
        json_dict["videos"].append(video_entry)
        video_id += 1

    with open(output_json, 'w') as f:
        json.dump(json_dict, f, indent=2)

    print(f"Saved valid.json with {len(json_dict['videos'])} videos at: {output_json}")

# Usage example:
if __name__ == "__main__":
    frames_root_dir = "/data/ytvis_2021/valid/JPEGImages"  # directory containing video folders
    output_json = "/data/ytvis_2021/valid.json"
    generate_valid_json(frames_root_dir=frames_root_dir, output_json=output_json)

def ytvis_results_to_coco_per_video(results_json_path, valid_json_path, output_dir):
    with open(results_json_path) as f:
        predictions = json.load(f)
    with open(valid_json_path) as f:
        valid = json.load(f)

    categories = valid["categories"]
    videos = valid["videos"]
    video_id_to_info = {str(v["id"]): v for v in videos}

    # Group predictions by video_id
    preds_by_video = {}
    for pred in predictions:
        vid = str(pred["video_id"])
        preds_by_video.setdefault(vid, []).append(pred)

    os.makedirs(output_dir, exist_ok=True)

    for video_id, video in video_id_to_info.items():
        file_names = video["file_names"]
        width = video["width"]
        height = video["height"]

        # Build images list
        images = []
        for idx, fname in enumerate(file_names):
            images.append({
                "id": idx + 1,
                "file_name": fname,
                "width": width,
                "height": height,
                "frame_id": idx
            })

        # Build annotations list
        coco_anns = []
        ann_id = 1
        for pred in preds_by_video.get(video_id, []):
            category_id = pred["category_id"]
            segmentations = pred["segmentations"]
            score = pred.get("score", 1.0)
            for frame_idx, seg in enumerate(segmentations):
                if seg is None:
                    continue
                coco_anns.append({
                    "id": ann_id,
                    "image_id": frame_idx + 1,
                    "category_id": category_id,
                    "segmentation": seg,
                    "score": score,
                    "iscrowd": 0,
                    "attributes": {
                        "track_id": pred.get("id", ann_id)
                    }
                })
                ann_id += 1

        coco_dict = {
            "info": valid.get("info", {}),
            "licenses": valid.get("licenses", []),
            "categories": categories,
            "images": images,
            "annotations": coco_anns
        }

        out_path = os.path.join(output_dir, f"video_{video_id}_coco.json")
        with open(out_path, "w") as f:
            json.dump(coco_dict, f, indent=2)
        print(f"Saved: {out_path}")

# Example usage:
# if __name__ == "__main__":
#     results_json = "/home/simone/dvis-daq-outputs/ytvis21_eval/inference/results.json"
#     valid_json = "/data/ytvis_2021/valid.json"  # Needed for categories if not in results.json
#     output_dir = "/home/simone/dvis-daq-outputs/ytvis21_eval/inference/coco_per_video"
#     ytvis_results_to_coco_per_video(results_json, valid_json, output_dir)