import json

VAL_JSON_PATH = "/data/fishway_ytvis/val_all.json"  # Change this if your file is elsewhere
OUTPUT_JSON_PATH = "/data/fishway_ytvis/val.json"  # Output file

MAX_FRAMES = 30

def get_middle_indices(total, n):
    """Get the start and end indices for the middle n elements of a list of length total."""
    if total <= n:
        return 0, total
    start = (total - n) // 2
    end = start + n
    return start, end

def main():
    with open(VAL_JSON_PATH, "r") as f:
        data = json.load(f)

    # Map video_id to (start, end) indices for slicing
    video_id_to_indices = {}
    for video in data["videos"]:
        total_frames = len(video["file_names"])
        start, end = get_middle_indices(total_frames, MAX_FRAMES)
        video["file_names"] = video["file_names"][start:end]
        video["length"] = len(video["file_names"])
        video_id_to_indices[video["id"]] = (start, end)

    # Truncate annotations
    for ann in data["annotations"]:
        vid = ann["video_id"]
        start, end = video_id_to_indices[vid]
        # Truncate segmentations, bboxes, areas if present and are lists
        for key in ["segmentations", "bboxes", "areas"]:
            if key in ann and isinstance(ann[key], list):
                ann[key] = ann[key][start:end]
        # Update length if present
        if "length" in ann:
            ann["length"] = min(ann.get("length", MAX_FRAMES), MAX_FRAMES)

    # Save new json
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved truncated val.json to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()