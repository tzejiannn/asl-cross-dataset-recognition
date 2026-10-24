"""
Writes the split of every clip of the 67 shared words, before cleaning:
    train / val / test   the ASL Citizen clips, in ASL Citizen's own signer-independent sets (973 / 258 / 819)
    wlasl                the WLASL clips (../03_wlasl_filter/wlasl_clips.json), test only (843)
Each clip's class id comes from ../02_vocabulary_overlap/class_list.txt. ASL Citizen ids are the video
file names without .mp4, WLASL ids the 5-digit ids; they never collide.

Usage (from this folder):
    python build_split.py
Writes split_uncleaned.json: {video_id: {"subset": ..., "action": [class_id, 1, -1]}}.
"""
import csv
import json
import os
from collections import Counter

classes = [l.rstrip("\n").split("\t") for l in open(os.path.join("..", "02_vocabulary_overlap", "class_list.txt"), encoding="utf-8")]
id_of_sign = {sign: int(k) for k, sign, _ in classes}
id_of_word = {word: int(k) for k, _, word in classes}

split = {}
for s in ("train", "val", "test"):
    with open(os.path.join("..", "..", "data", "asl_citizen", "ASL_Citizen", "splits", f"{s}.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Gloss"] in id_of_sign:
                split[os.path.splitext(r["Video file"])[0]] = {"subset": s, "action": [id_of_sign[r["Gloss"]], 1, -1]}
for word, insts in json.load(open(os.path.join("..", "03_wlasl_filter", "wlasl_clips.json"))).items():
    for i in insts:
        split[i["video_id"]] = {"subset": "wlasl", "action": [id_of_word[word], 1, -1]}

json.dump(split, open("split_uncleaned.json", "w"), indent=0)
print(dict(Counter(e["subset"] for e in split.values())))
