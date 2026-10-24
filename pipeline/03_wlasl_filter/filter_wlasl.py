"""
Keeps only the WLASL clips of the shared words (../02_vocabulary_overlap/class_list.txt): copies them
from ../01_wlasl_download/videos/ into videos/, and lists them with their source website and signer.
This is the WLASL data before cleaning (843 clips of the 67 words).

Usage (from this folder):
    python filter_wlasl.py
Writes videos/{video_id}.mp4 and wlasl_clips.json: word -> [{video_id, source, signer_id}].
"""
import json
import os
import shutil

SRC = os.path.join("..", "01_wlasl_download", "videos")
words = {l.rstrip("\n").split("\t")[2] for l in open(os.path.join("..", "02_vocabulary_overlap", "class_list.txt"), encoding="utf-8")}
have = {os.path.splitext(f)[0] for f in os.listdir(SRC)}

clips = {}
for entry in json.load(open(os.path.join("..", "01_wlasl_download", "metadata", "WLASL_asl100.json"))):
    if entry["gloss"] in words:
        clips[entry["gloss"]] = [{"video_id": i["video_id"], "source": i["source"], "signer_id": i["signer_id"]}
                                 for i in entry["instances"] if i["video_id"] in have]

os.makedirs("videos", exist_ok=True)
for insts in clips.values():
    for i in insts:
        dst = os.path.join("videos", i["video_id"] + ".mp4")
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(SRC, i["video_id"] + ".mp4"), dst)

json.dump(clips, open("wlasl_clips.json", "w"), indent=1)
print(f"{sum(len(v) for v in clips.values())} WLASL clips of {len(clips)} words")
