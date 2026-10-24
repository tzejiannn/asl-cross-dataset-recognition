"""
Copies the ASL100 clips out of a folder of WLASL clips named <video_id>.mp4 (e.g. a
flattened clone of the Voxel51 HuggingFace mirror). Searches src recursively, so the
original data/data_N/ layout works too. Copies, never moves, and skips existing files.

Run after preprocess.py: clips already in videos/ (from the original sources) are kept, so this
only adds the ones that could not be downloaded there.

Download the full Hugging Face copy first (needs git-lfs):
    git clone https://huggingface.co/datasets/Voxel51/WLASL hf_raw_videos

Usage (from 01_wlasl_download): python scripts/extract_asl100.py [src_dir] [dst_dir]
(defaults: hf_raw_videos and videos, both in 01_wlasl_download)
"""
import json
import os
import shutil
import sys

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 01_wlasl_download
src_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "hf_raw_videos")
dst_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, "videos")
if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
    sys.exit("src_dir and dst_dir must be different folders.")
if not os.path.isdir(src_dir):
    sys.exit(f"Source folder not found: {src_dir}")

json_path = os.path.join(script_dir, "metadata", "WLASL_asl100.json")
with open(json_path) as f:
    asl100_ids = {inst["video_id"] for entry in json.load(f) for inst in entry["instances"]}

os.makedirs(dst_dir, exist_ok=True)
copied, skipped = 0, 0
found = set()
dst_abs = os.path.abspath(dst_dir)
for root, dirs, files in os.walk(src_dir):
    # Don't descend into dst_dir if it lives inside src_dir.
    dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) != dst_abs]
    for fn in files:
        vid, ext = os.path.splitext(fn)
        if ext.lower() != ".mp4" or vid not in asl100_ids or vid in found:
            continue
        found.add(vid)
        dst = os.path.join(dst_dir, fn)
        if os.path.exists(dst):
            skipped += 1
            continue
        shutil.copyfile(os.path.join(root, fn), dst)
        copied += 1

print(f"Found {len(found)}/{len(asl100_ids)} ASL100 clips in {src_dir}: "
      f"{copied} copied, {skipped} already in {dst_dir}.")
