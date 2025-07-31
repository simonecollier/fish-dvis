import cv2
import os

# Path to the text file containing video paths
video_list_path = os.path.expanduser('~/fish-keypoints/videos_to_download.txt')

with open(video_list_path, 'r') as f:
    video_paths = [line.strip().strip('"') for line in f if line.strip()]

print(f"Checking frame rates for {len(video_paths)} videos...")

for video_path in video_paths:
    if not os.path.exists(video_path):
        print(f"[MISSING] {video_path}")
        continue
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    print(f"{video_path}\n  FPS: {fps:.2f}  Frames: {frame_count}  Duration: {duration:.2f} sec")
    cap.release() 