"""
Builds the final dataset in ../../data from pipeline stages 01-05: the uncleaned split minus the
WLASL clips removed by the cleaning (../08_wlasl_clean/excluded_clips.json).

Writes into ../../data:
    split.json                  ASL Citizen train 973 / val 258 / test 819, WLASL "wlasl" 791
    class_list.txt              the 67 classes
    wlasl/wlasl_clips.json      the 791 WLASL clips by word, with source and signer
    wlasl/videos/               the 791 WLASL clips
    asl_citizen/videos/         the 2,050 ASL Citizen clips of the 67 signs
Clips already there are not copied again. Then compares the WLASL clips with the published list
of the 791 test clips (../../data/wlasl_test_clips.csv) and names any that differ, e.g. clips a fresh
download couldn't retrieve. The licences (wlasl/C-UDA-1.0.pdf, asl_citizen/ASL_Citizen/use.txt)
and ASL Citizen's split files (asl_citizen/ASL_Citizen/splits/) are kept in data/ as downloaded.

Usage (from this folder):
    python build_final_dataset.py
"""
import csv
import json
import os
import shutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WLASL_DIR = os.path.join(THIS_DIR, "..", "03_wlasl_filter")
CIT_DIR = os.path.join(THIS_DIR, "..", "04_asl_citizen_download")
CLASS_LIST = os.path.join(THIS_DIR, "..", "02_vocabulary_overlap", "class_list.txt")
SPLIT = os.path.join(THIS_DIR, "..", "05_split", "split_uncleaned.json")
EXCLUDED = os.path.join(THIS_DIR, "..", "08_wlasl_clean", "excluded_clips.json")
KEEP_LIST = os.path.join(THIS_DIR, "..", "..", "data", "wlasl_test_clips.csv")
OUT = os.path.join(THIS_DIR, "..", "..", "data")


def copy(src, dst):
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


def main():
    # excluded_clips.json: {sign or source: {"reason", "count", "clips": [video_id, ...]}}
    excluded = {v for group in json.load(open(EXCLUDED)).values() for v in group["clips"]}
    split = {v: e for v, e in json.load(open(SPLIT)).items()
             if v not in excluded}
    words = [l.rstrip("\n").split("\t")[2] for l in open(CLASS_LIST, encoding="utf-8")]
    retrieved = json.load(open(os.path.join(WLASL_DIR, "wlasl_clips.json")))
    wlasl = {w: [i for i in retrieved[w] if i["video_id"] not in excluded] for w in words}

    os.makedirs(os.path.join(OUT, "wlasl"), exist_ok=True)  # a fresh clone has only data/README.md
    json.dump(split, open(os.path.join(OUT, "split.json"), "w"), indent=0)
    json.dump({w: [{k: i[k] for k in ("video_id", "source", "signer_id")} for i in insts] for w, insts in wlasl.items()},
              open(os.path.join(OUT, "wlasl", "wlasl_clips.json"), "w"), indent=1)
    copy(CLASS_LIST, os.path.join(OUT, "class_list.txt"))

    for insts in wlasl.values():
        for i in insts:
            copy(os.path.join(WLASL_DIR, "videos", i["video_id"] + ".mp4"),
                 os.path.join(OUT, "wlasl", "videos", i["video_id"] + ".mp4"))
    for v, e in split.items():
        if e["subset"] != "wlasl":
            copy(os.path.join(CIT_DIR, "videos", v + ".mp4"), os.path.join(OUT, "asl_citizen", "videos", v + ".mp4"))

    counts = {}
    for e in split.values():
        counts[e["subset"]] = counts.get(e["subset"], 0) + 1
    print(f"final dataset: {counts} ({len(excluded)} WLASL clips excluded across all WLASL100 words)")

    if os.path.exists(KEEP_LIST):
        expected = {r["video_id"] for r in csv.DictReader(open(KEEP_LIST, encoding="utf-8"))}
        got = {i["video_id"] for insts in wlasl.values() for i in insts}
        if got == expected:
            print(f"WLASL test clips match {os.path.basename(KEEP_LIST)} ({len(expected)} clips)")
        else:
            print(f"WLASL test clips differ from {os.path.basename(KEEP_LIST)}:")
            print(f"  missing ({len(expected - got)}): {sorted(expected - got)}")
            print(f"  not in the list ({len(got - expected)}): {sorted(got - expected)}")


if __name__ == "__main__":
    main()
