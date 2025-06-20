import subprocess
from pathlib import Path
import numpy as np
import json
import os
from matplotlib import pyplot as plt
import cv2
import ipywidgets as widgets
from pycocotools import mask as mask_utils
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from ipywidgets import (
    Button, Dropdown, HBox, VBox, IntSlider, Output, ToggleButtons, ToggleButton, Label
)
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath
from copy import deepcopy
from PIL import Image
from IPython.display import Video, display
from ipywidgets import FloatSlider
from collections import defaultdict
from tqdm import tqdm
import shutil
import random
import warnings
from filelock import FileLock, Timeout
from datetime import datetime, timedelta
import time
import threading

LABELLED_DATA_DIR = "/data/labeled/simone"
BASE_DATA_DIR = "/data/remora"

def hold_lock(lock_path):
    print("Thread 1: Acquiring lock...")
    with FileLock(lock_path, timeout=1):
        print("Thread 1: Lock acquired, holding for 20 seconds...")
        time.sleep(10)
        print("Thread 1: Lock released.")

def try_lock(lock_path):
    print("Thread 2: Trying to acquire lock...")
    try:
        with FileLock(lock_path, timeout=2):
            print("Thread 2: Lock acquired!")
    except Timeout:
        print("Thread 2: Could not acquire lock (as expected).")

def test_filelock_system(tracking_json_path):
    lock_path = str(tracking_json_path) + ".lock"
    t1 = threading.Thread(target=hold_lock, args=(lock_path,))
    t2 = threading.Thread(target=try_lock, args=(lock_path,))

    t1.start()
    time.sleep(2)  # Ensure t1 acquires the lock first
    t2.start()

    t1.join()
    t2.join()
    print("Test complete.")

def initialize_tracking_json(tracking_file_path):
    """Initialize or load the tracking JSON file."""
    tracking_file = Path(tracking_file_path)
    if not tracking_file.exists():
        tracking_data = {
            "videos": {},
            "last_updated": datetime.now().isoformat()
        }
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
    
    with open(tracking_file) as f:
        return json.load(f)

def select_unlabeled_video(base_dir, tracking_json_path, specific_path=None, random_select=True):
    """
    Get a video that hasn't been labeled yet.
    
    Args:
        base_dir (str/Path): Base directory containing videos
        tracking_json_path (str/Path): Path to JSON tracking file
        specific_path (str/Path, optional): Specific video or directory to check
        random_select (bool): If True, select random video; if False, get first available
    
    Returns:
        Path: Path to selected video, or None if no unlabeled videos found
        dict: Current tracking data
    """
    base_dir = Path(base_dir)
    lock_path = str(tracking_json_path) + ".lock"
    try:
        with FileLock(lock_path, timeout=0.1):
            with open(tracking_json_path, "r") as f:
                tracking_data = json.load(f)
    except Timeout:
        print(f"⚠️ The file {tracking_json_path} is currently locked by another process. Try again")
    
    # Handle specific path logic
    if specific_path:
        specific_path = Path(specific_path)
        if specific_path.is_file():
            video_files = [specific_path]
        elif specific_path.is_dir():
            if specific_path.name in ["Credit", "Ganaraska"]:
                # If "Credit" or "Ganaraska" folder is selected, follow the same logic as below
                if specific_path.name == "Credit":
                    specific_path = specific_path / "2024"
                elif specific_path.name == "Ganaraska":
                    specific_path = specific_path / "Ganaraska 2024"
            video_files = list(specific_path.rglob("*.mp4"))
        else:
            print(f"Invalid specific path: {specific_path}")
            return None
    else:
        # Randomly select "Credit" or "Ganaraska" if no specific path is provided
        selected_folder = random.choice(["Credit", "Ganaraska"])
        if selected_folder == "Credit":
            specific_path = base_dir / "Credit" / "2024"
        elif selected_folder == "Ganaraska":
            specific_path = base_dir / "Ganaraska" / "Ganaraska 2024"
        video_files = list(specific_path.rglob("*.mp4"))

    # Filter out already labeled or in-progress videos
    unlabeled_videos = [
        video for video in video_files
        if str(Path(video).relative_to(Path(base_dir))) not in tracking_data["videos"]
    ]
    
    if not unlabeled_videos:
        print("All videos in this directory have been labelled")
        return None
    
    # Select video
    if random_select:
        selected_video = random.choice(unlabeled_videos)
    else:
        selected_video = unlabeled_videos[0]
    
    print(f"Selected video: {selected_video}")
    return selected_video

def update_video_status(video_path, tracking_file_path, status, base_dir=BASE_DATA_DIR, labeler=None, comments=None, 
                        coco_json_path=None, images_path=None, labelled_data_dir=LABELLED_DATA_DIR):
    """
    Update the status of a video in the tracking JSON.
    
    Args:
        video_path (str/Path): Path to the video
        tracking_file_path (str/Path): Path to tracking JSON
        status (str): One of 'mask_generation_in_progress', 'masks_generated', 'mask_editing_in_progress', 'complete', 'skipped'
        labeler (str, optional): Name of person labeling
        comments (str, optional): Any comments about the video
    """
    lock_path = str(tracking_file_path) + ".lock"
    try:
        with FileLock(lock_path, timeout=0.1):
            with open(tracking_file_path, "r") as f:
                tracking_data = json.load(f)
    
            video_path = Path(video_path)
            base_dir = Path(base_dir)

            # Convert video_path to a relative path if it is within base_dir
            if video_path.is_relative_to(base_dir):
                video_path = str(video_path.relative_to(base_dir))
            else:
                raise ValueError(f"Video path '{video_path}' is not within the base directory '{base_dir}'.")

            # Ensure labelled_data_dir is a Path object
            if labelled_data_dir:
                labelled_data_dir = Path(labelled_data_dir)
            
            # if "simone" in video_path:
            #     video_path = convert_video_filename(local_path=video_path, data_dir=labelled_data_dir)

            if status == 'mask_generation_in_progress' or status == 'skipped':
                tracking_data["videos"][video_path] = {
                    "status": status,
                    "labeler": labeler,
                    "last_updated": datetime.now().isoformat(),
                    "comments": comments
                }

            elif status == 'masks_generated':
                # Ensure coco_json_path is relative to labelled_data_dir
                if labelled_data_dir:
                    if coco_json_path:
                        coco_json_path = Path(coco_json_path)
                        relative_coco_path = coco_json_path.relative_to(labelled_data_dir)
                    if images_path and labelled_data_dir:
                        images_path = Path(images_path)
                        relative_images_path = images_path.relative_to(labelled_data_dir)
                else:
                    relative_coco_path = coco_json_path
                    relative_images_path = images_path

                # Load the COCO JSON file
                with open(coco_json_path, "r") as f:
                    coco_data = json.load(f)

                # Create a mapping of track_id to category_id
                track_to_category = {}
                for annotation in coco_data["annotations"]:
                    track_id = annotation["attributes"]["track_id"]
                    category_id = annotation["category_id"]
                    if track_id not in track_to_category:
                        track_to_category[track_id] = category_id

                # Create a mapping of category_id to category_name
                category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

                # Get category names in the order of track IDs
                category_names = [
                    category_id_to_name[track_to_category[track_id]]
                    for track_id in sorted(track_to_category.keys())
                ]

                tracking_data["videos"][video_path]["status"] = status
                tracking_data["videos"][video_path]["last_updated"] = datetime.now().isoformat()
                tracking_data["videos"][video_path]["frames_path"] = str(relative_images_path)
                tracking_data["videos"][video_path]["generated_mask_annotations"] = str(relative_coco_path)
                tracking_data["videos"][video_path]["species"] = category_names
            
            elif status == "mask_editing_in_progress":
                gen_anns_path = tracking_data["videos"][video_path]["generated_mask_annotations"]
                images_path = tracking_data["videos"][video_path]["frames_path"]
                # Ensure gen_anns_path is relative to labelled_data_dir if possible
                if labelled_data_dir:
                    if gen_anns_path:
                        gen_anns_path = Path(gen_anns_path)
                        if not gen_anns_path.is_absolute():
                            readable_gen_anns_path = labelled_data_dir / gen_anns_path
                        else:
                            readable_gen_anns_path = gen_anns_path  # Already relative
                    else:
                        readable_gen_anns_path = gen_anns_path

                    if images_path:
                        images_path = Path(images_path)
                        if not images_path.is_absolute():
                            readable_images_path = labelled_data_dir / images_path
                        else:
                            readable_images_path = images_path
                    else:
                        readable_images_path = images_path

                # Create a new JSON file for edited masks if it doesn't already exist
                edited_json_path = readable_gen_anns_path.parent / "edited_instances_default.json"
                if not edited_json_path.exists():
                    with open(readable_gen_anns_path, "r") as f:
                        coco_data = json.load(f)

                    # Save the replica JSON file
                    with open(edited_json_path, "w") as f:
                        json.dump(coco_data, f, indent=2)
                    print(f"✅ Created edited JSON file: {edited_json_path}")

                # Add the relative path of the edited JSON file to the tracking data
                if labelled_data_dir and edited_json_path.is_relative_to(labelled_data_dir):
                    relative_edited_path = edited_json_path.relative_to(labelled_data_dir)
                else:
                    relative_edited_path = edited_json_path

                tracking_data["videos"][video_path]["status"] = status
                tracking_data["videos"][video_path]["last_updated"] = datetime.now().isoformat()
                tracking_data["videos"][video_path]["edited_mask_annotations"] = str(relative_edited_path)
                
                tracking_data["last_updated"] = datetime.now().isoformat()

                with open(tracking_file_path, 'w') as f:
                    json.dump(tracking_data, f, indent=2)

                return edited_json_path, readable_images_path
            
            elif status == "requires_review" or status == "complete":
                tracking_data["videos"][video_path]["status"] = status
                tracking_data["videos"][video_path]["last_updated"] = datetime.now().isoformat()
            
            else:
                print("❌ Invalid status entry. Status must be one of the following: " \
                "skipped, mask_generation_in_progress, masks_generated, mask_editing_in_progress, complete.")
                return
            
            tracking_data["last_updated"] = datetime.now().isoformat()

            with open(tracking_file_path, 'w') as f:
                json.dump(tracking_data, f, indent=2)
    
    except Timeout:
        print(f"⚠️ The file {tracking_file_path} is currently locked by another process. Try again.")

def confirm_video_for_labeling(video_path, tracking_json_path, base_dir, labeler_name=None):
    """
    Confirm whether to label the selected video and update tracking status.
    
    Args:
        video_path (str/Path): Path to the video
        tracking_json_path (str/Path): Path to tracking JSON file
        labeler_name (str, optional): Name of person doing the labeling
    
    Returns:
        bool: True if video accepted for labeling, False otherwise
    """
    while True:
        decision = input("\nDo you want to label this video? (y/n/s): ").lower()
        if decision in ['y', 'n', 's']:
            break
        print("Invalid input. Please enter 'y' for yes, 'n' for no, or 's' for skip (too difficult)")

    if decision == 'y':
        if not labeler_name:
            labeler_name = input("Enter your name: ")
        update_video_status(
            video_path, 
            tracking_json_path,
            status='mask_generation_in_progress',
            base_dir=base_dir,
            labeler=labeler_name
        )
        return True
    elif decision == 's':
        reason = input("Enter reason for skipping: ")
        update_video_status(
            video_path, 
            tracking_json_path,
            status='skipped',
            base_dir=base_dir,
            comments=str(reason)
        )
    return False

def select_video_for_editing(tracking_file_path, labeler_name, base_dir=BASE_DATA_DIR, labelled_data_dir=LABELLED_DATA_DIR):
    """
    Check the tracking_data JSON for videos with the status "masks_generated" or 
    "mask_editing_in_progress" and the specified labeler name. Prompt the user to select a video.
    """
    # Load the tracking_data JSON
    lock_path = str(tracking_file_path) + ".lock"
    try:
        with FileLock(lock_path, timeout=0.1):
            with open(tracking_file_path, "r") as f:
                tracking_data = json.load(f)
    except Timeout:
        print(f"⚠️ The file {tracking_file_path} is currently locked by another process. Try again")

    # Filter videos by status and labeler name
    matching_videos = [
        (video_path, info)
        for video_path, info in tracking_data["videos"].items()
        if info["status"] in ["masks_generated", "mask_editing_in_progress", "requires_review"] and info["labeler"] == labeler_name
    ]

    # Check if there are any matching videos
    if not matching_videos:
        print(f"No videos found with status 'masks_generated', 'mask_editing_in_progress', or 'requires_review' for labeler '{labeler_name}'.")
        return None

    # Display the matching videos with a number
    print("\nMatching videos:")
    for i, (video_path, info) in enumerate(matching_videos, start=1):
        print(f"{i}. Video: {video_path}")
        for key, value in info.items():
            print(f"   {key.capitalize()}: {value}")
        print()

    # Prompt the user to select a video
    while True:
        try:
            selection = int(input("For which video do you want to edit the generated masks? (Enter the number): "))
            if 1 <= selection <= len(matching_videos):
                selected_video = matching_videos[selection - 1][0]
                selected_video = str(Path(base_dir) / Path(selected_video))
                print(f"\nYou selected: {selected_video}")
                edited_json_path, readable_images_path = update_video_status(video_path=selected_video, 
                                                                             tracking_file_path=tracking_file_path,
                                                                             status="mask_editing_in_progress",
                                                                             base_dir=base_dir,
                                                                             labelled_data_dir=labelled_data_dir)
                return str(selected_video), str(edited_json_path), str(readable_images_path)
            else:
                print(f"Please enter a number between 1 and {len(matching_videos)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def create_labelled_video_dir(label_dir, video_path, base_dir):
    # Extract parts from the path
    parts = Path(video_path).relative_to(base_dir).parts[:-1]  # ['Credit', '08012024-08122024', '24  08  01  11  58']
    filename_stem = video_path.stem  # '28'

    # Construct new folder name
    new_folder_name = "__".join(parts + (filename_stem,))

    base_dir = Path(label_dir) # Change this to the directory you will store the images and annotations for all videos

    # Full path to new folder
    new_folder_path = base_dir / new_folder_name

    # Create the directory (if it doesn't already exist)
    new_folder_path.mkdir(parents=True, exist_ok=True)

    print(f"Created folder: {new_folder_path}")
    return new_folder_path

def extract_frames_opencv(video_path, output_dir, start_number=0, quiet=True, jpeg_quality=100):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not quiet:
        print(f"Video FPS: {fps}")
        print(f"Total frames reported: {total_frames}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filename = output_dir / f"{frame_count + start_number:05d}.jpg"
        cv2.imwrite(str(filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        frame_count += 1

        if not quiet and frame_count % 100 == 0:
            print(f"Saved {frame_count} frames...")

    cap.release()
    print(f"Extracted {frame_count} frames to: {output_dir}")


def create_coco_json(frames_path, output_path):
    categories = [
    {
      "id": 1,
      "name": "Chinook",
      "supercategory": ""
    },
    {
      "id": 2,
      "name": "Coho",
      "supercategory": ""
    },
    {
      "id": 3,
      "name": "Atlantic",
      "supercategory": ""
    },
    {
      "id": 4,
      "name": "Rainbow Trout",
      "supercategory": ""
    },
    {
      "id": 5,
      "name": "Brown Trout",
      "supercategory": ""
    },
    {
      "id": 6,
      "name": "Unknown",
      "supercategory": ""
    }
    ]
    images = []
    annotations = []
    coco_json = {}

    # Fill out the license information
    coco_json["licenses"] = [{"name": "", "id": 0, "url": ""}]
    coco_json["info"] = {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": "",
    }

    # Image entries
    output_pattern = "%05d.jpg"

    # Loop through all frames
    for frame_idx in range(len(os.listdir(frames_path))):
        file_name = output_pattern % frame_idx
        # Add the image information
        images.append(
            {
                "id": frame_idx + 1,
                "license": 0,
                "file_name": file_name,
                "height": 960,
                "width": 1280,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

    # Update the coco_json dictionary
    coco_json.update(
        {"categories": categories, "images": images, "annotations": annotations}
    )

    # Save the json file in the frames directory and the json directory
    with open(output_path, "w") as json_file:
        json.dump(coco_json, json_file, indent=2)

    print(f"COCO JSON saved to: {output_path}")
    return coco_json

STATUS_FILE = "gpu_status.json"
LOCK_FILE = STATUS_FILE + ".lock"

def log_on(user_name, est_end_time):
    # If the input is in "HH:MM" format, prepend the current date
    if ":" in est_end_time and len(est_end_time) == 5:
        today = datetime.now().strftime("%Y-%m-%d")
        est_end_time = f"{today}T{est_end_time}:00"  # Add seconds to match ISO format

    # Parse the estimated end time
    try:
        est_end_datetime = datetime.fromisoformat(est_end_time)
    except ValueError:
        print("❌ Invalid time format. Please use 'HH:MM' or 'YYYY-MM-DDTHH:MM:SS'.")
        return

    # Get the current time
    now = datetime.now()

    # Ensure the stop time is in the future
    if est_end_datetime <= now:
        print("❌ Stop time must be in the future.")
        return

    # Ensure the stop time is no more than 12 hours from now
    if est_end_datetime > now + timedelta(hours=12):
        print("❌ Cannot request GPU for more than 12 hours.")
        return

    with FileLock(LOCK_FILE):
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)

        # Find first free GPU
        for gpu_id, info in status.items():
            if info is None:
                # Assign GPU
                status[gpu_id] = {
                    "user": user_name,
                    "start_time": now.isoformat(),
                    "est_end": est_end_time
                }
                with open(STATUS_FILE, "w") as f:
                    json.dump(status, f, indent=2)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                print(f"✅ Assigned GPU {gpu_id} to {user_name}")
                return

        print("❌ All GPUs are currently in use:")
        for gpu_id, info in status.items():
            print(f"  GPU {gpu_id} → {info['user']} until {info['est_end']}")

def log_off(user_name):
    with FileLock(LOCK_FILE):
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)

        for gpu_id, info in status.items():
            if info and info["user"] == user_name:
                status[gpu_id] = None
                with open(STATUS_FILE, "w") as f:
                    json.dump(status, f, indent=2)
                print(f"✅ {user_name} has logged off GPU {gpu_id}")
                return

        print(f"⚠️ No active GPU found for {user_name}")

# How you use it.....
#log_on("alice", "2025-05-02T15:00:00")
# later...
#log_off("alice")

## Here’s a function you could run at the start of any cell or add to a watchdog:
def check_expiration(user_name, warn_minutes=10, expire_minutes=15):
    with FileLock(LOCK_FILE):
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)

        for gpu_id, info in status.items():
            if info and info["user"] == user_name:
                est_end = datetime.fromisoformat(info["est_end"])
                now = datetime.now()
                delta = est_end - now
                if delta.total_seconds() < 60 * warn_minutes and delta.total_seconds() > 0:
                    print(f"⚠️ GPU session for {user_name} on GPU {gpu_id} is expiring in {int(delta.total_seconds() // 60)} minutes.")
                elif now > est_end + timedelta(minutes=expire_minutes):
                    print(f"⛔ GPU session for {user_name} on GPU {gpu_id} has expired and will be logged off.")
                    status[gpu_id] = None
                    with open(STATUS_FILE, "w") as f:
                        json.dump(status, f, indent=2)
                    print(f"✅ GPU {gpu_id} is now free.")
                return

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

def clean_clicks(clicks):
    """
    Remove empty objects (no pos or neg clicks), and clean up categories and frames
    if they become empty as a result.
    
    Args:
        clicks (dict): Nested dict {frame_id: {category: {track_id: {"pos": [...], "neg": [...]}}}}

    Returns:
        dict: Cleaned version of the same dictionary.
    """
    cleaned_clicks = {}

    for frame_id, frame_data in clicks.items():
        new_frame = {}
        for category, obj_dict in frame_data.items():
            new_cat = {}
            for obj_id, clicks_dict in obj_dict.items():
                pos = clicks_dict.get("pos", [])
                neg = clicks_dict.get("neg", [])
                if pos or neg:
                    new_cat[obj_id] = {"pos": pos, "neg": neg}
            if new_cat:
                new_frame[category] = new_cat
        if new_frame:
            cleaned_clicks[frame_id] = new_frame

    return cleaned_clicks


def sam_mask_to_uncompressed_rle(mask_tensor, is_binary=False):
    """
    Converts a SAM2 mask tensor (shape [1, H, W]) into COCO uncompressed RLE.
    """
    # Step 1: Convert to binary mask (uint8, 0/1)
    if is_binary:
      binary_mask = mask_tensor.astype(np.uint8)
    else:
      binary_mask = (mask_tensor > 0).astype(np.uint8)  # Threshold at 0

    # Step 2: Fortran-contiguous layout (required by pycocotools)
    binary_mask_fortran = np.asfortranarray(binary_mask)

    # Step 3: Encode to RLE (compressed by default)
    rle = mask_utils.encode(binary_mask_fortran)

    # Calculate area of the mask
    area = float(mask_utils.area(rle))

    # Calculate bounding box
    bbox = mask_utils.toBbox(rle).tolist()

    # Step 4: Convert counts from bytes to list (uncompressed-style for COCO/YTVIS)
    rle["counts"] = list(rle["counts"])
    
    return rle, area, bbox

def create_annotation_id_map(coco_dict):
    """
    Create a mapping of (image_id, track_id) to annotation index.
    """
    ann_index_map = {}
    for idx, ann in enumerate(coco_dict["annotations"]):
        ann_index_map[(ann["image_id"], ann["attributes"]["track_id"])] = idx
    return ann_index_map

def create_data_maps(coco_dict):
    """
    Create a mapping of image_id to file_name and a mapping of category_id to name.
    """
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_dict["images"]}
    image_id_to_data = {img["id"]: img for img in coco_dict["images"]}
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_dict["categories"]}
    categories = list(category_id_to_name.values())
    category_name_to_id = {v: k for k, v in category_id_to_name.items()}
    
    return image_id_to_filename, image_id_to_data, categories, category_id_to_name, category_name_to_id

def remove_floating_mask_parts(coco, max_size=800):
        """
        Remove floating mask components smaller than max_size for all annotations in the COCO dict.
        If the largest component is <= max_size, nothing is removed.
        """
        for ann in coco["annotations"]:
            rle = ann.get("segmentation")
            if not rle or not isinstance(rle, dict) or not rle.get("counts"):
                continue
            mask = decode_rle(rle)
            if mask.size == 0:
                continue

            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            # Find the largest component (excluding background)
            max_area = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area

            # Remove small components (except the largest if > max_size)
            cleaned_mask = np.zeros_like(mask, dtype=np.uint8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_size or area == max_area:
                    cleaned_mask[labels == i] = 1

            # If the largest component is <= max_size, keep all
            if max_area <= max_size:
                continue

            # Update annotation if changed
            if not np.array_equal(mask, cleaned_mask):
                new_rle, area, bbox = sam_mask_to_uncompressed_rle(cleaned_mask, is_binary=True)
                ann["segmentation"] = new_rle
                ann["area"] = int(area)
                ann["bbox"] = bbox
        
        print(f"Removed floating mask components smaller than {max_size}.")

# A widget for adding click prompts for object annotations
class ImageAnnotationWidget:
    def __init__(self, coco_dict, image_dir, start_frame=0, predictor=None, inference_state=None, output_json_path=None):
        # Close any existing figures to prevent memory issues
        plt.close('all')
        # add arguments to the class output
        self.image_dir = image_dir
        self.predictor = predictor
        self.inference_state = inference_state
        self.coco = coco_dict
        self.coco_json_path = output_json_path
        # Create data maps of coco info
        self.image_id_to_filename, self.image_id_to_data, self.categories, self.category_id_to_name, self.category_name_to_id = create_data_maps(self.coco)
        self.image_ids = sorted(self.image_id_to_data.keys())
        self.annotations_by_image = self._group_annotations(self.coco["annotations"])
        self.cat_to_color = self._assign_colors()
        # Initialize some variables
        self.clicks = {}
        self.current_frame_idx = start_frame
        self.current_xlim = None
        self.current_ylim = None
        self.mask_history = {} # Dictionary to store mask history per frame
        self.active_category = self.categories[0]
        self.active_track_id = 1
        self.show_clicks = True  # Default to showing clicks

        # Create UI elements
        self.category_selector = widgets.Dropdown(options=self.categories, 
                                                  value = self.active_category,
                                                  description="Species")
        self.track_id_selector = widgets.BoundedIntText(value=self.active_track_id, 
                                                        min=1, 
                                                        max=100, 
                                                        step=1, 
                                                        description="Track ID")
        self.prev_button = widgets.Button(description="Previous Frame")
        self.next_button = widgets.Button(description="Next Frame")
        self.frame_slider = widgets.IntSlider(
            value=self.current_frame_idx,
            min=0,
            max=len(self.image_ids) - 1,
            step=1,
            description="Frame",
            continuous_update=False  # Disable continuous updates to reduce lag
        )
        self.generate_mask_button = widgets.Button(description="Generate Mask")
        self.undo_mask_button = widgets.Button(description="Undo Mask")
        self.delete_button = widgets.Button(description="Delete Annotation")
        self.show_clicks_toggle = widgets.ToggleButton(
            value=True,
            description='Show Clicks',
            tooltip='Toggle click visibility'
        )
        # Add button and target frame box for forward and backward mask propagation
        self.propagate_button = widgets.Button(description="Propagate Mask")
        self.target_frame = widgets.BoundedIntText(
            value=len(self.image_ids)-1,
            min=0,
            max=len(self.image_ids)-1,
            description="Target Frame"
        )
        self.save_button = widgets.Button(description="Save JSON")

        self.output = widgets.Output()

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.canvas.header_visible = False  # This hides the "Figure #" text
        
        self._connect_events()
        self._display_ui()

    def _group_annotations(self, annotations):
        """
        Groups annotations by image_id and then by track_id for faster lookup.
        
        Returns:
            dict: {
                image_id: {
                    track_id: annotation,
                    ...
                },
                ...
            }
        """
        grouped = {}
        for ann in annotations:
            image_id = ann["image_id"]
            track_id = ann["attributes"]["track_id"]
            
            # Initialize nested dictionaries if they don't exist
            if image_id not in grouped:
                grouped[image_id] = {}
                
            # Store annotation by track_id
            grouped[image_id][track_id] = ann
            
        return grouped

    def _assign_colors(self):
        cmap = plt.get_cmap("Set1")
        num_colors = cmap.N
        return {track_id: cmap(i % num_colors) for i, track_id in enumerate(range(1, 10))}

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.prev_button.on_click(lambda b: self.update_frame(-1))
        self.next_button.on_click(lambda b: self.update_frame(+1))
        self.generate_mask_button.on_click(self.generate_mask_for_current_frame)
        self.undo_mask_button.on_click(self.undo_mask_for_current_frame)
        self.delete_button.on_click(self.delete_annotation_for_current_track)  
        self.propagate_button.on_click(self._propagate_mask)
        self.save_button.on_click(self._save_annotations)

        # Dropdown/selector events using observe
        self.category_selector.observe(self._update_category, names='value')
        self.track_id_selector.observe(self._update_track_id, names='value')
        self.show_clicks_toggle.observe(self._update_click_visibility, names='value')
        self.frame_slider.observe(self._on_slider_change, names="value")


    def _update_category(self, change):
        """Handler for category selection changes"""
        self.active_category = change['new']
        
    def _update_track_id(self, change):
        """Handler for track ID selection changes"""
        self.active_track_id = change['new']
    
    def _update_click_visibility(self, change):
        """Handler for click visibility toggle"""
        self.show_clicks = change['new']
        self.plot_frame()  # Refresh display
    
    def _on_slider_change(self, change):
        """Handle changes to the frame slider."""
        new_frame_idx = change["new"]
        if new_frame_idx != self.current_frame_idx:
            self.current_frame_idx = new_frame_idx
            self.plot_frame()

    def _display_ui(self):
        controls = VBox([
            HBox([self.prev_button, self.next_button, self.frame_slider]),
            HBox([self.category_selector, self.track_id_selector]),
            HBox([self.generate_mask_button, self.undo_mask_button, 
                  self.delete_button, self.show_clicks_toggle]),
            HBox([self.propagate_button, self.target_frame, self.save_button])
        ])
        
        #display(widgets.VBox([controls, self.output]))
        self.ui = widgets.VBox([controls, self.output])
        self.plot_frame()

    def plot_frame(self):
        image_id = self.image_ids[self.current_frame_idx]
        image_info = self.image_id_to_data[image_id]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_anns = self.annotations_by_image.get(image_id, {})

        with self.output:
            self.output.clear_output(wait=True)
            self.ax.clear()
            self.ax.imshow(image)
            self.ax.set_title(f"{image_info['file_name']}")

            # Draw annotations
            for track_id, ann in image_anns.items():
                cat_id = ann["category_id"]
                cat_name = self.category_id_to_name[cat_id]
                track_id = ann["attributes"]["track_id"]
                color = self.cat_to_color[track_id] # Use track-based color

                x, y, w, h = ann["bbox"]

                if "segmentation" in ann:
                    rle = ann["segmentation"]
                    mask = decode_rle(rle)
                    if mask.size > 0:  # Only plot if mask is not empty
                        self.ax.text(int(x + w / 2), y - 20, f"{cat_name} T{track_id}", color=color, fontsize=10, weight='bold')
                        self.ax.imshow(np.ma.masked_where(mask == 0, mask),
                                    cmap=mcolors.ListedColormap([color]), alpha=0.2)

            # Draw clicks
            if self.show_clicks:
                frame_clicks = self.clicks.get(image_id, {})
                for cat, tracks in frame_clicks.items():
                    for track_id, data in tracks.items():
                        pos = np.array(data.get("pos", []))
                        neg = np.array(data.get("neg", []))
                        if len(pos):
                            self.ax.scatter(pos[:, 0], pos[:, 1], c="green", marker="o")
                            for (x, y) in pos:
                                self.ax.text(x + 5, y, f"{cat} T{track_id}", color="green", fontsize=10)
                        if len(neg):
                            self.ax.scatter(neg[:, 0], neg[:, 1], c="red", marker="x")
                            for (x, y) in neg:
                                self.ax.text(x + 5, y, f"{cat} T{track_id}", color="red", fontsize=10)

            if self.current_xlim and self.current_ylim:
                self.ax.set_xlim(self.current_xlim)
                self.ax.set_ylim(self.current_ylim)

            self.fig.tight_layout()

    def update_frame(self, direction):
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()
        new_idx = self.current_frame_idx + direction
        if 0 <= new_idx < len(self.image_ids):
            self.current_frame_idx = new_idx
            self.plot_frame()
    
    def generate_mask_for_current_frame(self, b):
        # Filter out specific SAM2 warning about _C module
        warnings.filterwarnings('ignore', message='cannot import name.*_C.*from.*sam2')
        warnings.filterwarnings('ignore', message='Skipping the post-processing step.*')
    
        image_id = self.image_ids[self.current_frame_idx]
        frame_clicks = self.clicks.get(image_id, {})

        # Save current state before generating new masks
        if image_id not in self.mask_history:
            self.mask_history[image_id] = {}
        
        # Remove empty clicks
        frame_clicks = clean_clicks({image_id: frame_clicks})[image_id]

        
        with self.output:
            self.output.clear_output(wait=False)  # Clear previous output
            if not frame_clicks:
                print("No valid clicks on this frame.")
                return

            for category, track in frame_clicks.items():
                category_id = self.category_name_to_id[category]
                for track_id, data in track.items():
                    # Store history per track
                    if track_id not in self.mask_history[image_id]:
                        self.mask_history[image_id][track_id] = []

                    # Store current state before changes
                    current_ann = next((ann.copy() for ann in self.coco["annotations"] 
                                    if ann["image_id"] == image_id and 
                                    ann["attributes"]["track_id"] == track_id), None)
                    if current_ann:
                        self.mask_history[image_id][track_id].append(current_ann)

                    # Prepare points
                    pos = np.array(data.get("pos", []), dtype=np.float32)
                    neg = np.array(data.get("neg", []), dtype=np.float32)
                    if len(pos) == 0 and len(neg) == 0:
                        continue
                    labels = [1] * len(pos) + [0] * len(neg)
                    points = np.concatenate([pos, neg], axis=0) if len(neg) else pos
                    labels = np.array(labels, dtype=np.int32)

                    # Generate mask
                    _, out_ids, masks = self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=self.current_frame_idx,
                        obj_id=track_id,
                        points=points,
                        labels=labels,
                    )

                    if track_id not in out_ids:
                        print(f"Failed to generate mask for {category} (track {track_id})")
                        continue

                    mask_idx = out_ids.index(track_id)
                    mask_tensor = masks[mask_idx].cpu().numpy()[0]
                    rle, area, bbox = sam_mask_to_uncompressed_rle(mask_tensor)

                    found = False
                    for ann in self.coco["annotations"]:
                        if ann["image_id"] == image_id and ann["attributes"]["track_id"] == track_id:
                            ann.update({"segmentation": rle, "area": area, "bbox": bbox})
                            found = True
                            break
                    if not found:
                        new_ann = {
                            "id": max([a["id"] for a in self.coco["annotations"]] + [0]) + 1,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": rle,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                            "attributes": {
                                "occluded": False,
                                "rotation": 0.0,
                                "track_id": track_id,
                                "keyframe": True,
                            },
                        }
                        self.coco["annotations"].append(new_ann)

        self.annotations_by_image = self._group_annotations(self.coco["annotations"])
        self.plot_frame()

    def undo_mask_for_current_frame(self, b):
        image_id = self.image_ids[self.current_frame_idx]

        # Check if there's any annotation for this frame and track_id
        current_ann = next((ann for ann in self.coco["annotations"] 
                        if ann["image_id"] == image_id and 
                        ann["attributes"]["track_id"] == self.active_track_id), None)
        
        # If no annotation exists for this frame/track_id, do nothing
        if not current_ann:
            print(f"No mask to undo for track {self.active_track_id} in frame {image_id}")
            return
        
        # Check if we have history for this track
        if (image_id not in self.mask_history or 
            self.active_track_id not in self.mask_history[image_id] or 
            not self.mask_history[image_id][self.active_track_id]):
            # If no history, set empty segmentation but keep annotation entry
            current_ann["segmentation"]["counts"] = []
            current_ann["area"] = 0
            current_ann["bbox"] = [0, 0, 0, 0]

            # Update annotations_by_image
            if image_id in self.annotations_by_image:
                # for i, ann in enumerate(self.annotations_by_image[image_id]):
                #     if ann["attributes"]["track_id"] == self.active_track_id:
                #         self.annotations_by_image[image_id][i] = current_ann
                #         break
                self.annotations_by_image[image_id][self.active_track_id] = current_ann
        else:
            # Restore previous state for this track only
            previous_ann = self.mask_history[image_id][self.active_track_id].pop()
            
            # Update current annotation in coco["annotations"]
            for i, ann in enumerate(self.coco["annotations"]):
                if (ann["image_id"] == image_id and 
                    ann["attributes"]["track_id"] == self.active_track_id):
                    self.coco["annotations"][i] = previous_ann
                    break
            
            # Update annotations_by_image
            if image_id in self.annotations_by_image:
                self.annotations_by_image[image_id][self.active_track_id] = previous_ann

        # Update display
        self.plot_frame()

    def delete_annotation_for_current_track(self, b):
        """Delete annotation for current track in current frame."""
        image_id = self.image_ids[self.current_frame_idx]
        
        # Check if there's any annotation for this frame and track_id
        current_ann = next((ann for ann in self.coco["annotations"] 
                        if ann["image_id"] == image_id and 
                        ann["attributes"]["track_id"] == self.active_track_id), None)
        
        if not current_ann:
            print(f"No annotation to delete for track {self.active_track_id} in frame {image_id}")
            return
        
        # Remove from coco annotations
        self.coco["annotations"] = [ann for ann in self.coco["annotations"]
                                if not (ann["image_id"] == image_id and 
                                        ann["attributes"]["track_id"] == self.active_track_id)]
        
        # Remove from annotations_by_image
        if image_id in self.annotations_by_image:
            if self.active_track_id in self.annotations_by_image[image_id]:
                del self.annotations_by_image[image_id][self.active_track_id]
        
        # Clear history for this track in this frame
        if image_id in self.mask_history and self.active_track_id in self.mask_history[image_id]:
            del self.mask_history[image_id][self.active_track_id]
        
        # Update display
        self.plot_frame()
    
    def _propagate_mask(self, b):
        """Propagate masks forward from the current frame, one track at a time."""
        image_id = self.image_ids[self.current_frame_idx]
        target_frame = self.target_frame.value
        
        with self.output:
            self.output.clear_output(wait=True)  # Clear previous output

            if target_frame == self.current_frame_idx:
                print("❌ Target frame must be different from current frame")
                return
                
            # Get all tracks in current frame
            current_anns = self.annotations_by_image.get(image_id, {})
            if not current_anns:
                print("❌ No masks to propagate in current frame")
                return
            
            # Propagate each track separately
            for track_id, cur_ann in current_anns.items():
                # Reset the inference state before propagating each track
                self.predictor.reset_state(self.inference_state)

                # Add clicks for the current track back to the inference state
                frame_clicks = self.clicks.get(image_id, {}).get(self.category_id_to_name[cur_ann["category_id"]], {})
                track_clicks = frame_clicks.get(track_id, {})
                pos = np.array(track_clicks.get("pos", []), dtype=np.float32)
                neg = np.array(track_clicks.get("neg", []), dtype=np.float32)
                labels = [1] * len(pos) + [0] * len(neg)
                points = np.concatenate([pos, neg], axis=0) if len(neg) else pos
                labels = np.array(labels, dtype=np.int32)

                if len(points) == 0:
                    print(f"❌ No valid clicks for track {track_id}. Skipping propagation.")
                    continue
                
                self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.current_frame_idx,
                    obj_id=track_id,
                    points=points,
                    labels=labels,
                )

                print(f"🔄 Propagating {self.category_id_to_name[cur_ann['category_id']]} track {track_id}...")

                # Propagate masks
                self.video_segments = track_masks(
                    cur_frame_idx=self.current_frame_idx,
                    predictor=self.predictor,
                    inference_state=self.inference_state,
                    max_frame2propagate=target_frame,
                )

                # Update annotations with propagated masks
                print(f"Updating annotations for {self.category_id_to_name[cur_ann['category_id']]} track {track_id}...")
                for ann_frame_idx, masks in self.video_segments.items():
                    for obj_id, mask_tensor in masks.items():
                        # Convert to uncompressed RLE
                        rle, area, bbox = sam_mask_to_uncompressed_rle(mask_tensor, is_binary=True)

                        if isinstance(rle, dict) and "counts" in rle:
                            counts = rle["counts"]
                            if isinstance(counts, list) and len(counts) < 10:
                                break

                        found = False
                        for ann in self.coco["annotations"]:
                            if ann["image_id"] == ann_frame_idx + 1 and ann["attributes"]["track_id"] == obj_id:
                                ann.update({"segmentation": rle, "area": area, "bbox": bbox})
                                found = True
                                break

                        if not found:
                            new_ann = {
                                "id": max([a["id"] for a in self.coco["annotations"]] + [0]) + 1,
                                "image_id": ann_frame_idx + 1,
                                "category_id": cur_ann["category_id"],
                                "segmentation": rle,
                                "area": area,
                                "bbox": bbox,
                                "iscrowd": 0,
                                "attributes": {
                                    "occluded": False,
                                    "rotation": 0.0,
                                    "track_id": obj_id,
                                    "keyframe": True,
                                },
                            }
                            self.coco["annotations"].append(new_ann)
            
                print(f"✅ Propagation complete for {self.category_id_to_name[cur_ann['category_id']]} track {track_id}.")

            # Update internal state
            self.annotations_by_image = self._group_annotations(self.coco["annotations"])
            print("✅ All tracks propagated.")

        self.plot_frame()

    def _on_click(self, event):
        if not event.inaxes:
            return

        xdata, ydata = round(event.xdata), round(event.ydata)
        image_id = self.image_ids[self.current_frame_idx]
        
        with self.output:
            self.output.clear_output(wait=False)  # Clear previous output
            # Check if this track ID is already used for a different species
            for cat, tracks in self.clicks.get(image_id, {}).items():
                if (cat != self.active_category and  # Different category
                    self.active_track_id in tracks):  # Same track ID
                    print(f"❌ Error: Track ID {self.active_track_id} is already used for species {cat}.")
                    print("Each fish should have a unique Track ID number!")
                    return

        self.clicks.setdefault(image_id, {}).setdefault(self.active_category, {}).setdefault(self.active_track_id, {"pos": [], "neg": []})
        click_type = "pos" if event.button == 1 else "neg" if event.button == 3 else None
        if click_type is None:
            return

        points = self.clicks[image_id][self.active_category][self.active_track_id][click_type]
        for i, (px, py) in enumerate(points):
            if abs(px - xdata) <= 10 and abs(py - ydata) <= 10:
                points.pop(i)
                self.plot_frame()
                return

        points.append([xdata, ydata])
        self.plot_frame()

    def _on_key(self, event):
        if event.key == 'right':
            self.update_frame(+1)
        elif event.key == 'left':
            self.update_frame(-1)
        elif event.key == 'p':
            self.copy_clicks_from_previous()
            self.plot_frame()

    def copy_clicks_from_previous(self):
        idx = self.current_frame_idx
        if idx == 0:
            return
        prev_id = self.image_ids[idx - 1]
        curr_id = self.image_ids[idx]
        if prev_id not in self.clicks:
            return
        prev_clicks = self.clicks[prev_id]
        curr_clicks = self.clicks.setdefault(curr_id, {})
        for cat, tracks in prev_clicks.items():
            for track_id, data in tracks.items():
                curr_clicks.setdefault(cat, {}).setdefault(track_id, {"pos": [], "neg": []})
                curr_clicks[cat][track_id]["pos"].extend([pt[:] for pt in data.get("pos", [])])
                curr_clicks[cat][track_id]["neg"].extend([pt[:] for pt in data.get("neg", [])])
    
    def _save_annotations(self, b):
        """Save current annotations to JSON file"""
        with self.output:
            self.output.clear_output(wait=False)  # Clear previous output
            try:
                output_path = self.coco_json_path
                # Remove floating mask parts before saving
                remove_floating_mask_parts(self.coco, max_size=800)
                # Save the COCO JSON
                with open(output_path, "w") as f:
                    json.dump(self.coco, f, indent=2)
                
                print(f"✅ Annotations saved to: {output_path}")
            except Exception as e:
                print(f"❌ Error saving annotations: {str(e)}")

    def __del__(self):
        """Cleanup when widget is destroyed"""
        plt.close(self.fig)

# A function to track masks over multiple frames
def track_masks(
    cur_frame_idx,
    predictor,
    inference_state,
    max_frame2propagate=None,
):
    """
    Track masks either forward or backward through video frames.
    
    Args:
        cur_frame_idx (int): Current frame index (where mask exists)
        predictor: SAM2 predictor instance
        inference_state: SAM2 inference state
        max_frame2propagate (int): Target frame to propagate to
        reverse (bool): If True, propagate backward; if False, propagate forward
    """
    # Dictionary to save the generated masks for each frame
    video_segments = {}

    if max_frame2propagate < cur_frame_idx:
        # For backward propagation
        num_frames_to_track = cur_frame_idx - max_frame2propagate + 1
        reverse = True
        print(f"Starting from frame {cur_frame_idx}, tracking backwards {num_frames_to_track} frames")
    else:
        # For forward propagation
        num_frames_to_track = max_frame2propagate - cur_frame_idx + 1
        reverse = False
        print(f"Starting from frame {cur_frame_idx}, tracking forwards {num_frames_to_track} frames")

    # Propage the video and save the masks
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=cur_frame_idx,
        reverse=reverse,
        max_frame_num_to_track=num_frames_to_track,
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    return video_segments

def adjust_brightness_contrast(img, brightness=1.0, contrast=1.0):
        img = img.astype(np.float32)
        # Apply contrast (centered at 128)
        img = (img - 128) * contrast + 128
        # Apply brightness
        img = img * brightness
        # Clip and convert back
        return np.clip(img, 0, 255).astype(np.uint8)

## Drawing widget -------
class MaskEditor:
    def __init__(self, coco_json_path, frames_dir, start_frame, mask_transparancy=0.2):
        self.frames_dir = Path(frames_dir)
        # Load the COCO JSON file
        self.coco_json_path = Path(coco_json_path)
        with open(coco_json_path) as f:
            self.coco = json.load(f)
        # Create name maps
        self.image_id_to_filename, self.image_id_to_data, self.categories, self.category_id_to_name, self.category_name_to_id = create_data_maps(self.coco)

        self.annotations_by_image = {}
        self.track_to_category = {}
        for ann in self.coco["annotations"]:
            # Store original annotations
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)
            # Create mapping of track_id to category_name
            track_id = ann["attributes"]["track_id"]
            category_name = self.category_id_to_name[ann["category_id"]]
            self.track_to_category[track_id] = category_name

        # Assign colors to track_ids
        self.track_colors = self._assign_colors()

        # Create dropdown options in "Object track_id: category_name" format
        self.track_options = [f"Object {track_id}: {cat_name}" 
                            for track_id, cat_name in sorted(self.track_to_category.items())]
        
        if self.track_options:
            self.active_track_id = int(self.track_options[0].split(':')[0].split()[-1])
            category_name = self.track_to_category[self.active_track_id]
            self.active_category_id = self.category_name_to_id[category_name]
        
        self.image_ids = sorted(self.image_id_to_filename.keys())
        self.current_index = start_frame
        self.mode = "draw"
        self.brush_size = 10
        self.mask_alpha = mask_transparancy
        self.show_mask = True

        self.drawing_mode = "polygon"  # Add this to track drawing tool
        self.polygon_vertices = []    # Store vertices while drawing polygon
        self.temp_line = None        # Store temporary line while drawing

        self.mask_history = {}  # Dictionary to hold history for each image_id
        self.last_click_pos = None
        self.zoom_mode = False  # Add this to track if we're waiting for a zoom click
        self.img = None  # Initialize img attribute
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.canvas.header_visible = False

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # Connect both click types
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        self._setup_ui()
        self._update_canvas()

        # Return the widget for display
        display(self.ui)
        
    def _setup_ui(self):
        self.output = Output()

        self.prev_btn = Button(description="Previous")
        self.next_btn = Button(description="Next")
        self.save_btn = Button(description="Save JSON")
        self.undo_btn = Button(description="Undo")
        self.smooth_btn = Button(description="Smooth Mask")
        self.zoom_in_btn = Button(description="Zoom In")
        self.reset_zoom_btn = Button(description="Reset Zoom")

        self.mode_toggle = ToggleButtons(options=["draw", "erase"], value="draw")
        self.show_mask_toggle = ToggleButton(value=True, description="Show Mask")  # Toggle button for showing/hiding mask
        #self.brush_slider = IntSlider(description="Brush Size", min=1, max=50, value=self.brush_size)

        # Track-based dropdown
        self.object_dropdown = Dropdown(
            options=self.track_options,
            value=self.track_options[0] if self.track_options else None,
            description="Track"
        )

        # Add drawing tool dropdown
        self.drawing_tool = Dropdown(
            options=['polygon', 'lasso'],
            value='polygon',
            description='Tool:'
        )

        self.brightness_slider = FloatSlider(description="Brightness", min=0.0, max=2.0, value=1.0, step=0.01)
        self.contrast_slider = FloatSlider(description="Contrast", min=0.0, max=2.0, value=1.0, step=0.01)

        self.prev_btn.on_click(lambda _: self._change_frame(-1))
        self.next_btn.on_click(lambda _: self._change_frame(1))
        self.save_btn.on_click(lambda _: self._save_json())
        self.undo_btn.on_click(lambda _: self._undo_last_action())
        self.smooth_btn.on_click(lambda _: self._smooth_mask())
        self.zoom_in_btn.on_click(self._zoom_in)
        self.reset_zoom_btn.on_click(self._reset_zoom)

        self.drawing_tool.observe(self._update_drawing_tool, names='value')
        self.mode_toggle.observe(self._update_mode, names="value")
        self.object_dropdown.observe(self._update_object, names="value")
        self.show_mask_toggle.observe(self._toggle_mask_visibility, names="value")  # Observe toggle button changes

        #self.brush_slider.observe(self._update_brush, names="value")

        self.brightness_slider.observe(self._update_canvas_from_slider, names="value")
        self.contrast_slider.observe(self._update_canvas_from_slider, names="value")

        controls = VBox([
            HBox([self.prev_btn, self.next_btn, self.object_dropdown, self.drawing_tool]),
            HBox([Label("Mode:"), self.mode_toggle, self.smooth_btn, self.undo_btn, self.save_btn]),
            HBox([self.zoom_in_btn, self.reset_zoom_btn,
                  self.brightness_slider, self.contrast_slider, self.show_mask_toggle]),
        ])

        self.output = Output()
        self.ui = VBox([controls, self.output])

        #display(VBox([controls, self.output]))
    
    def _assign_colors(self):
        cmap = plt.get_cmap("Set1")  # Use a colormap (e.g., Set1, viridis)
        num_colors = cmap.N
        return {track_id: cmap(i % num_colors) for i, track_id in enumerate(range(1, 10))}
    
    def _toggle_mask_visibility(self, change):
        """Toggle the visibility of the mask."""
        self.show_mask = change["new"]
        self._update_canvas()
    
    def _update_drawing_tool(self, change):
        self.drawing_mode = change['new']
        if self.drawing_mode == 'lasso':
            self.lasso.set_active(True)
            # Clear any in-progress polygon
            self.polygon_vertices = []
            if self.temp_line:
                self.temp_line.remove()
                self.temp_line = None
                self.fig.canvas.draw_idle()
        else:
            self.lasso.set_active(False)

    def _update_mode(self, change):
        self.mode = change["new"]

    def _update_object(self, change):
        # Parse track ID and category from selected option
        track_id = int(change["new"].split(':')[0].split()[-1])
        category_name = change["new"].split(': ')[1]
        
        self.active_track_id = track_id
        self.active_category_id = self.category_name_to_id[category_name]

    #def _update_brush(self, change):
        #self.brush_size = change["new"]
    
    def _update_canvas_from_slider(self, change):
        self._update_canvas()

    def _change_frame(self, direction):
        self.current_index = np.clip(self.current_index + direction, 0, len(self.image_ids) - 1)
        self._update_canvas()
    
    def _undo_last_action(self):
        image_id = self.image_ids[self.current_index]
        if image_id in self.mask_history and self.mask_history[image_id]:
            # Restore mask + annotations
            last_mask, last_anns = self.mask_history[image_id].pop()
            self.mask = last_mask
            self.annotations_by_image[image_id] = deepcopy(last_anns)

            # Update the global coco annotations list
            self.coco["annotations"] = [
                ann for ann in self.coco["annotations"]
                if ann["image_id"] != image_id
            ]
            self.coco["annotations"].extend(deepcopy(last_anns))

            # Refresh display
            masked_display = np.ma.masked_where(self.mask == 0, self.mask * 40)
            self.img_plot.set_data(masked_display)
            self.img_plot.figure.canvas.draw_idle()
    
    def _zoom_in(self, b):
        self.zoom_mode = "in"
        print("Click where you want to zoom in")

    def _reset_zoom(self, b):
        # Reset view to full image
        self.ax.set_xlim(0, self.img.shape[1])
        self.ax.set_ylim(self.img.shape[0], 0)
        self.fig.canvas.draw_idle()
    
    def _on_click(self, event):
        if not event.inaxes == self.ax:
            return
            
        if self.zoom_mode:
            # Handle zoom functionality
            self.last_click_pos = (event.xdata, event.ydata)
            if self.zoom_mode == "in":
                self._perform_zoom(zoom_in=True)
                self.zoom_mode = False
            return

        if self.drawing_mode == 'polygon':
            if event.dblclick:
                # Complete polygon
                if len(self.polygon_vertices) >= 3:
                    self.polygon_vertices.append(self.polygon_vertices[0])  # Close the polygon
                    verts = np.array(self.polygon_vertices)
                    self._on_select(verts)
                    
                # Clear temporary drawing
                self.polygon_vertices = []
                if self.temp_line:
                    self.temp_line.remove()
                    self.temp_line = None
                self.fig.canvas.draw_idle()
            else:
                # Add vertex
                self.polygon_vertices.append([event.xdata, event.ydata])
                self._update_temp_polygon()
    
    def _on_mouse_move(self, event):
        if not event.inaxes == self.ax:
            return
            
        if self.drawing_mode == 'polygon' and len(self.polygon_vertices) > 0:
            # Update temporary line
            temp_vertices = self.polygon_vertices + [[event.xdata, event.ydata]]
            self._update_temp_polygon(temp_vertices)
    
    def _update_temp_polygon(self, vertices=None):
        if vertices is None:
            vertices = self.polygon_vertices
            
        if self.temp_line:
            self.temp_line.remove()
        
        if len(vertices) > 0:
            verts = np.array(vertices)
            self.temp_line, = self.ax.plot(verts[:, 0], verts[:, 1], 'r-', linewidth=1)
            self.fig.canvas.draw_idle()
    
    def _perform_zoom(self, zoom_in=True):
        if self.last_click_pos is None:
            return

        x, y = self.last_click_pos
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        # Calculate zoom factor
        factor = 0.5 if zoom_in else 2.0
        
        # Calculate new width and height
        new_width = (current_xlim[1] - current_xlim[0]) * factor
        new_height = (current_ylim[0] - current_ylim[1]) * factor
        
        # Calculate initial zoom window centered on click
        xmin = x - new_width/2
        xmax = x + new_width/2
        ymin = y - new_height/2
        ymax = y + new_height/2
        
        # Get image boundaries
        img_width = self.img.shape[1]
        img_height = self.img.shape[0]
            
        # Adjust if zoom window exceeds boundaries
        if xmin < 0:
            xmax = min(new_width, img_width)
            xmin = 0
        elif xmax > img_width:
            xmin = max(img_width - new_width, 0)
            xmax = img_width
            
        if ymin < 0:
            ymax = min(new_height, img_height)
            ymin = 0
        elif ymax > img_height:
            ymin = max(img_height - new_height, 0)
            ymax = img_height
        
        # Set new limits
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymax, ymin)  # Reversed for matplotlib's coordinate system
        self.fig.canvas.draw_idle()

    def _update_canvas(self):
        with self.output:
            self.output.clear_output(wait=True)
            # Clear the current axes instead of creating new ones
            self.ax.clear()
            image_id = self.image_ids[self.current_index]
            image_path = self.frames_dir / self.image_id_to_filename[image_id]
            self.img = np.array(Image.open(image_path))

            # Apply brightness and contrast adjustments
            self.img = adjust_brightness_contrast(self.img, self.brightness_slider.value, self.contrast_slider.value)

            self.ax.imshow(self.img)
            self.ax.set_title(f"Image ID: {image_id}")
            self.ax.axis("off")

            # Initialize mask as zero (no mask yet)
            self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)

            # Apply existing annotations as masks (if any)
            anns = self.annotations_by_image.get(image_id, [])
            for ann in anns:
                rle = ann["segmentation"]
                mask = decode_rle(rle)
                track_id = ann["attributes"]["track_id"]
                color = self.track_colors.get(track_id, (1, 0, 0, 0.3))
                #self.mask = np.ma.masked_where(mask == 0, mask * ann["category_id"])
                self.mask[mask == 1] = ann["category_id"]

                # Draw label above the mask
                x, y, w, h = ann["bbox"]
                cat_name = self.category_id_to_name[ann["category_id"]]

                self.ax.text(
                    int(x + w / 2), y - 20,
                    f"{cat_name} T{track_id}",
                    color=color if isinstance(color, str) else mcolors.to_hex(color),
                    fontsize=10, weight='bold')

            # Only plot the mask if it's not empty (i.e., there are masks for the current frame)
            if np.any(self.mask) and self.show_mask:
                # Display only non-zero regions in the mask
                mask_display = np.ma.masked_where(self.mask == 0, self.mask)
                self.img_plot = self.ax.imshow(mask_display * 40, cmap="jet", alpha=self.mask_alpha)
                

            # Draw the lasso selector to allow drawing/erasing
            self.lasso = LassoSelector(self.ax, onselect=self._on_select)
            if self.drawing_mode != 'lasso':
                self.lasso.set_active(False)

            self.fig.canvas.draw_idle()  # Update the canvas instead of displaying new figure
            self.fig.tight_layout()
            # display(self.fig)
            # plt.close()

    def _on_select(self, verts):
        image_id = self.image_ids[self.current_index]
        anns = self.annotations_by_image.setdefault(image_id, [])
        
        # Save a deepcopy of mask and annotations *before* any changes
        previous_mask = self.mask.copy()
        previous_anns = deepcopy(anns)
        self.mask_history.setdefault(image_id, []).append((previous_mask, previous_anns))

        path = MplPath(verts)
        y, x = np.meshgrid(np.arange(self.mask.shape[0]), np.arange(self.mask.shape[1]), indexing='ij')
        points = np.vstack((x.flatten(), y.flatten())).T
        inside = path.contains_points(points).reshape(self.mask.shape)

        if self.mode == "draw":
            self.mask[inside] = self.active_category_id
        elif self.mode == "erase":
            # Only erase pixels belonging to the active category
            self.mask[inside & (self.mask == self.active_category_id)] = 0

        # Update both internal state and COCO annotations
        self._update_annotations(image_id)
        
        # Force redraw of the mask
        self._refresh_display()

    def _update_annotations(self, image_id):
        # Extract updated binary mask for the current category
        category_mask = (self.mask == self.active_category_id)
        # Get current annotations for this image
        anns = self.annotations_by_image.setdefault(image_id, [])
        # Find existing annotation for this track/category
        existing = next((a for a in anns 
                        if a["category_id"] == self.active_category_id 
                        and a["attributes"]["track_id"] == self.active_track_id), None)

        # Remove the existing annotation from both lists if it exists
        if existing:
            anns.remove(existing)
            self.coco["annotations"] = [a for a in self.coco["annotations"] 
                                    if not (a["image_id"] == image_id and 
                                            a["category_id"] == self.active_category_id and
                                            a["attributes"]["track_id"] == self.active_track_id)]

        # Only create new annotation if there are pixels in the mask
        if np.any(category_mask):
            encoded_rle, area, bbox = sam_mask_to_uncompressed_rle(category_mask, is_binary=True)
            
            new_ann = {
                "id": max([a["id"] for a in self.coco["annotations"]] + [0]) + 1,
                "image_id": image_id,
                "category_id": self.active_category_id,
                "segmentation": encoded_rle,
                "area": int(area),
                "bbox": bbox,
                "iscrowd": 0,
                "attributes": {
                    "occluded": False,
                    "rotation": 0.0,
                    "track_id": self.active_track_id,
                    "keyframe": False
                }
            }
            anns.append(new_ann)
            self.coco["annotations"].append(new_ann)

    def _smooth_mask(self):
        image_id = self.image_ids[self.current_index]

        # Save current state before smoothing — this makes undo work!
        anns = self.annotations_by_image.setdefault(image_id, [])
        previous_mask = self.mask.copy()
        previous_anns = deepcopy(anns)
        self.mask_history.setdefault(image_id, []).append((previous_mask, previous_anns))

        binary_mask = (self.mask == self.active_category_id).astype(np.uint8) * 255

        # Morphological closing to smooth edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


        # Fill holes by flood-filling the background and inverting
        h, w = closed.shape
        flood_fill = closed.copy()
        mask_flood = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(flood_fill, mask_flood, (0, 0), 255)
        holes_filled = cv2.bitwise_or(closed, cv2.bitwise_not(flood_fill))

        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes_filled, connectivity=8)
        cleaned_mask = np.zeros_like(holes_filled)
        min_size = 500
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_mask[labels == i] = 255

        self.mask[cleaned_mask > 0] = self.active_category_id
        self._update_annotations(image_id)
        
        # Force redraw of the mask
        self._refresh_display()
    
    def _refresh_display(self):
        """Helper method to update the display"""
        if hasattr(self, 'img_plot'):
            masked_display = np.ma.masked_where(self.mask == 0, self.mask * 40)
            self.img_plot.set_data(masked_display)
            self.fig.canvas.draw_idle()

    def _save_json(self):
        file_name = self.coco_json_path.name
        if file_name.startswith("edited"):
            output_path = self.coco_json_path.parent / f"{self.coco_json_path.name}"
        else:
            output_path = self.coco_json_path.parent / f"edited_{self.coco_json_path.name}"
        # Save the edited COCO JSON
        with open(output_path, "w") as f:
            json.dump(self.coco, f, indent=2)
        print(f"Saved to {output_path}")

def convert_annotations_to_uncompressed(coco_data):
    """
    Converts all compressed RLE segmentations in the COCO JSON to uncompressed RLE.
    """
    for ann in coco_data["annotations"]:
        seg = ann.get("segmentation")
        if isinstance(seg, dict) and isinstance(seg["counts"], str):
            # Decode compressed RLE to binary mask
            rle = {
                "size": seg["size"],
                "counts": seg["counts"].encode("utf-8")
            }
            binary_mask = mask_utils.decode(rle).astype(np.uint8)

            # Re-encode using your uncompressed RLE function
            uncompressed_rle, area, bbox = sam_mask_to_uncompressed_rle(binary_mask, is_binary=True)

            # Replace in annotation
            ann["segmentation"] = uncompressed_rle
            ann["area"] = int(area)
            ann["bbox"] = bbox

    return coco_data


def convert_all_coco_to_ytvis(coco_root_dir, output_json_path, indentation=2):
    """
    Convert COCO-style annotations (with track_ids) for multiple videos into one
    YTVIS-style JSON.
    
    Parameters:
        coco_root_dir (str): Path to the folders of video annotations.
        output_json_path (str): Path to write the YTVIS-style JSON.
    """

    ytvis_json = {
        "info": {"description": "Converted from COCO to YTVIS"},
        "categories": [],
        "videos": [],
        "annotations": []
    }

    video_id = 0
    annotation_id_offset = 0
    categories_added = False

    for video_folder in sorted(os.listdir(coco_root_dir)):
        video_path = os.path.join(coco_root_dir, video_folder)
        images_path = os.path.join(video_path, "images")
        coco_json_path = os.path.join(video_path, "annotations", "edited_instances_default.json")

        if not os.path.exists(images_path) or not os.path.exists(coco_json_path):
            continue

        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        if not categories_added:
            ytvis_json["categories"] = coco_data["categories"]
            categories_added = True

        # Sort images by filename and build image_id -> frame index mapping
        images_sorted = sorted(coco_data["images"], key=lambda x: x["file_name"])
        image_id_to_frame_index = {img["id"]: idx for idx, img in enumerate(images_sorted)}
        frame_filenames = [os.path.join(video_folder, "images", img["file_name"]) for img in images_sorted]

        width = images_sorted[0]["width"]
        height = images_sorted[0]["height"]
        length = len(images_sorted)

        ytvis_json["videos"].append({
            "id": video_id,
            "width": width,
            "height": height,
            "length": length,
            "file_names": frame_filenames
        })

        # Organize annotations by track_id
        track_segments = {}
        for anno in coco_data["annotations"]:
            track_id = anno["attributes"]["track_id"]
            image_id = anno["image_id"]
            frame_index = image_id_to_frame_index.get(image_id)

            if frame_index is None:
                continue  # skip unmatched image_ids

            if track_id not in track_segments:
                track_segments[track_id] = {
                    "id": track_id + annotation_id_offset,
                    "video_id": video_id,
                    "category_id": anno["category_id"],
                    "segmentations": [None] * length,
                    "areas": [None] * length,
                    "bboxes": [None] * length,
                    "iscrowd": anno.get("iscrowd", 0)
                }

            track_segments[track_id]["segmentations"][frame_index] = anno["segmentation"]
            track_segments[track_id]["areas"][frame_index] = anno["area"]
            track_segments[track_id]["bboxes"][frame_index] = anno["bbox"]

        ytvis_json["annotations"].extend(track_segments.values())

        max_track_id = max(track_segments.keys(), default=0)
        annotation_id_offset += max_track_id + 1
        video_id += 1

    with open(output_json_path, "w") as f:
        json.dump(ytvis_json, f, indent=indentation)

    print(f"YTVIS JSON saved to {output_json_path}")


def find_short_segmentations(json_path, threshold=10):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "images" in data:
        # COCO format
        print("Detected COCO format.")
        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

        for anno in data["annotations"]:
            seg = anno.get("segmentation")
            if isinstance(seg, dict) and "counts" in seg:
                counts = seg["counts"]
                if isinstance(counts, list) and len(counts) < threshold:
                    image_id = anno["image_id"]
                    print(f"[COCO] Short segmentation: image_id={image_id}, file_name={id_to_filename.get(image_id, 'unknown')}, counts_len={len(counts)}")

    elif "videos" in data:
        # YTVIS format
        print("Detected YTVIS format.")
        video_id_to_names = {v["id"]: v["file_names"] for v in data["videos"]}

        for anno in data["annotations"]:
            segs = anno.get("segmentations", [])
            video_id = anno["video_id"]
            file_names = video_id_to_names.get(video_id, [])
            for frame_idx, seg in enumerate(segs):
                if isinstance(seg, dict) and "counts" in seg:
                    counts = seg["counts"]
                    if isinstance(counts, list) and len(counts) < threshold:
                        file_name = file_names[frame_idx] if frame_idx < len(file_names) else "unknown"
                        print(f"[YTVIS] Short segmentation: video_id={video_id}, frame={frame_idx}, file_name={file_name}, counts_len={len(counts)}")

    else:
        print("Unsupported format: JSON must contain 'images' or 'videos' key.")

def fix_video_folders_safe(source_root, target_root):
    """
    Create a safe copy of Fishway_Data as Fishway_Data_NoDup, then:
    - Remove duplicated frames.
    - Renumber frames to be continuous (00000.jpg, 00001.jpg, ...)
    - Update all COCO JSON files inside annotations/ folders accordingly.

    Args:
        source_root (str or Path): Original dataset directory (e.g., Fishway_Data).
        target_root (str or Path): New safe directory (e.g., Fishway_Data_NoDup).
    """
    source_root = Path(source_root)
    target_root = Path(target_root)

    if target_root.exists():
        raise ValueError(f"Target directory {target_root} already exists. Please delete it first to avoid accidental overwrite.")

    print(f"Copying {source_root} -> {target_root} (this might take a few minutes)...")
    shutil.copytree(source_root, target_root)
    print("✅ Copy complete.")

    print("Processing videos in the copied directory...")
    for video_folder in tqdm(list(target_root.glob("*"))):
        if not video_folder.is_dir():
            continue

        images_dir = video_folder / "images"
        annotations_dir = video_folder / "annotations"

        if not images_dir.exists() or not annotations_dir.exists():
            print(f"Skipping {video_folder.name}: missing images/ or annotations/")
            continue

        # Step 1: Detect duplicated frames
        image_files = sorted(images_dir.glob("*.jpg"))
        to_delete = []
        last_img = None

        for img_file in image_files:
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if last_img is not None and np.array_equal(img, last_img):
                to_delete.append(img_file)
            else:
                last_img = img

        # Step 2: Remove duplicated images
        for dup in to_delete:
            os.remove(dup)

        # Step 3: Build renaming dict
        remaining_images = sorted(images_dir.glob("*.jpg"))
        rename_dict = {}
        for idx, img_file in enumerate(remaining_images):
            new_name = f"{idx:05d}.jpg"
            if img_file.name != new_name:
                rename_dict[img_file.name] = new_name
                img_file.rename(images_dir / new_name)

        deleted_filenames = {dup.name for dup in to_delete}

        # Step 4: Update all COCO files
        json_files = list(annotations_dir.glob("*.json"))
        for coco_path in json_files:
            with open(coco_path, "r") as f:
                coco = json.load(f)

            # Build filename -> image entry mapping
            filename_to_image = {img["file_name"]: img for img in coco["images"]}

            deleted_image_ids = set()
            new_images = []
            old_id_to_new_id = {}
            next_image_id = 0

            # Update images
            for img in coco["images"]:
                filename = Path(img["file_name"]).name
                if filename in deleted_filenames:
                    deleted_image_ids.add(img["id"])
                    continue
                # Rename if necessary
                if filename in rename_dict:
                    new_filename = rename_dict[filename]
                    img["file_name"] = str(Path(img["file_name"]).parent / new_filename)

                old_id_to_new_id[img["id"]] = next_image_id
                img["id"] = next_image_id
                new_images.append(img)
                next_image_id += 1

            # Update annotations
            new_annotations = []
            next_ann_id = 0
            for ann in coco["annotations"]:
                if ann["image_id"] in deleted_image_ids:
                    continue

                # Remove annotations with segmentation count lists of less than 10
                segmentation = ann.get("segmentation", {})
                if isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
                    if len(segmentation["counts"]) < 10:
                        continue

                ann["image_id"] = old_id_to_new_id.get(ann["image_id"], ann["image_id"])
                ann["id"] = next_ann_id
                new_annotations.append(ann)
                next_ann_id += 1

            coco["images"] = new_images
            coco["annotations"] = new_annotations

            with open(coco_path, "w") as f:
                json.dump(coco, f, indent=2)

        print(f"✅ {video_folder.name} fixed.")

def reassign_category_id(coco_json_path, track_id, new_category_id, output_path=None):
    """
    Reassigns the category ID for all annotations with a given track ID in a COCO JSON file.

    Parameters:
        coco_json_path (str): Path to the COCO JSON file.
        track_id (int): The track ID for which the category ID should be reassigned.
        new_category_id (int): The new category ID to assign.
        output_path (str, optional): Path to save the updated COCO JSON file. If None, the original file will be overwritten.

    Returns:
        None
    """
    # Load the COCO JSON file
    with open(coco_json_path, 'r') as file:
        coco_data = json.load(file)

    # Iterate through annotations and update the category_id for the given track_id
    for annotation in coco_data.get('annotations', []):
        if annotation.get('attributes', {}).get('track_id') == track_id:
            annotation['category_id'] = new_category_id

    # Save the updated COCO JSON file
    output_path = output_path or coco_json_path
    with open(output_path, 'w') as file:
        json.dump(coco_data, file, indent=4)

    print(f"Category ID for track ID {track_id} has been updated to {new_category_id} in {output_path}.")

def convert_video_filename(local_path, data_dir=LABELLED_DATA_DIR):
    local_path = Path(local_path)
    local_path = str(local_path.relative_to(Path(data_dir)))
    # Split the partial path into components
    parts = local_path.split("__")
    
    # Extract the year from the last folder name (e.g., "24 08 01 11 58" -> "2024")
    last_folder = parts[-2]
    year = "20" + last_folder.split()[0]
    
    # Construct the full path
    hd_path = f"/VERBATIM HD/{parts[0]}/{year}/{parts[1]}/{parts[2]}/{parts[3]}.mp4"
    
    return hd_path

def check_duplicate_annotations(coco_json_path):
    """
    Check for multiple annotations with the same image_id and track_id in a COCO JSON file.

    Args:
        coco_json_path (str): Path to the COCO JSON file.

    Returns:
        list: A list of duplicate entries, each containing (image_id, track_id).
    """
    with open(coco_json_path, 'r') as file:
        coco_data = json.load(file)

    # Dictionary to track (image_id, track_id) pairs
    annotation_tracker = defaultdict(list)

    # Iterate through annotations
    for annotation in coco_data.get('annotations', []):
        image_id = annotation['image_id']
        track_id = annotation['attributes']['track_id']
        annotation_tracker[(image_id, track_id)].append(annotation)

    # Find duplicates
    duplicates = [
        (image_id, track_id, anns)
        for (image_id, track_id), anns in annotation_tracker.items()
        if len(anns) > 1
    ]

    if not duplicates:
        print("No duplicate annotations found.")
        return

    print("Found duplicate annotations:")
    for image_id, track_id, anns in duplicates:
        print(f"\nImage ID: {image_id}, Track ID: {track_id}, Number of duplicates: {len(anns)}")
        for i, ann in enumerate(anns):
            print(f"  {i + 1}. Annotation ID: {ann['id']}, Area: {ann.get('area', 'N/A')}, BBox: {ann.get('bbox', 'N/A')}")

        while len(anns) > 1:
            try:
                choice = int(input(f"Choose which annotation to keep for Image ID {image_id}, Track ID {track_id} (1-{len(anns)}): "))
                if 1 <= choice <= len(anns):
                    # Remove all annotations except the chosen one
                    anns_to_remove = [ann for i, ann in enumerate(anns) if i != choice - 1]
                    for ann in anns_to_remove:
                        coco_data['annotations'].remove(ann)
                    print(f"Removed {len(anns_to_remove)} duplicate annotations.")
                    break
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(anns)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Save the updated JSON file
    with open(coco_json_path, 'w') as file:
        json.dump(coco_data, file, indent=2)

    print("Duplicates resolved and JSON file updated.")

def select_video_to_edit(video_folder):
    """
    Check the tracking_data JSON for videos with the status "masks_generated" or 
    "mask_editing_in_progress" and the specified labeler name. Prompt the user to select a video.
    """
    video_folder = Path(video_folder)
    images_dir = video_folder / "images"
    unedited_json_path = video_folder / "annotations/instances_default.json"
    edited_json_path = video_folder / "annotations/edited_instances_default.json"
    if not edited_json_path.exists():
        with open(unedited_json_path, "r") as f:
            coco_data = json.load(f)

        # Save the replica JSON file
        with open(edited_json_path, "w") as f:
            json.dump(coco_data, f, indent=2)
        print(f"✅ Created edited JSON file: {edited_json_path}")
    
    return images_dir, edited_json_path

def get_color_for_track(track_id):
    # Generate a color based on track_id (repeatable)
    np.random.seed(track_id)
    color = np.random.randint(0, 255, 3)
    return tuple(int(c) for c in color)

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay a single mask on an image."""
    overlay = image.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (np.array(color) * alpha + overlay[mask_bool] * (1 - alpha)).astype(np.uint8)
    return overlay

def create_video_with_masks(frames_dir, coco_json_path, output_video_path, fps=30):
    # Load COCO JSON
    with open(coco_json_path) as f:
        coco = json.load(f)
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    annotations_by_image = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    # Sort frames by image_id
    sorted_image_ids = sorted(image_id_to_filename.keys())
    first_frame = np.array(Image.open(Path(frames_dir) / image_id_to_filename[sorted_image_ids[0]]))
    height, width = first_frame.shape[:2]

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    for image_id in sorted_image_ids:
        frame_path = Path(frames_dir) / image_id_to_filename[image_id]
        frame = np.array(Image.open(frame_path).convert("RGB"))
        anns = annotations_by_image.get(image_id, [])
        for ann in anns:
            mask = decode_rle(ann["segmentation"])
            if mask.size > 0:
                track_id = ann.get("attributes", {}).get("track_id", 0)
                color = get_color_for_track(track_id)
                frame = overlay_mask(frame, mask, color=color, alpha=0.4)
        # Convert RGB to BGR for OpenCV
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"✅ Video saved to {output_video_path}")