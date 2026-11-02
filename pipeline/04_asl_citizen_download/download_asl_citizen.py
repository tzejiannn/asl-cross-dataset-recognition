"""
Downloads the ASL Citizen clips of the shared signs (../02_vocabulary_overlap/class_list.txt), straight
out of ASL_Citizen.zip (listed as 42.8 GB) on the Microsoft Download Center via HTTP range requests
(remotezip), instead of downloading the whole archive. 2,050 clips of 67 signs.

../../data/asl_citizen/ASL_Citizen/ holds ASL Citizen's split files (splits/train.csv, val.csv, test.csv: every clip with its
sign, signer and ASL-LEX code) and licence (use.txt), taken from the same zip. ASL Citizen is under
the Microsoft Research License Terms: non-commercial research only, no redistribution of the data or
of anything derived from it, and the data must be destroyed when the research is complete.

Usage (from this folder):
    python download_asl_citizen.py
"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from remotezip import RemoteZip

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"


def fetch(names, out_dir):
    # one RemoteZip (one central-directory read) per worker batch
    with RemoteZip(URL) as z:
        for name in names:
            dest = os.path.join(out_dir, name)
            if not os.path.exists(dest):
                with z.open(f"ASL_Citizen/videos/{name}") as src, open(dest + ".part", "wb") as dst:
                    dst.write(src.read())
                os.replace(dest + ".part", dest)
    return len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    signs = {l.split("\t")[1] for l in open(os.path.join("..", "02_vocabulary_overlap", "class_list.txt"), encoding="utf-8")}
    files = []
    for s in ("train", "val", "test"):
        with open(os.path.join("..", "..", "data", "asl_citizen", "ASL_Citizen", "splits", f"{s}.csv"), encoding="utf-8") as f:
            files += [r["Video file"] for r in csv.DictReader(f) if r["Gloss"] in signs]
    os.makedirs("videos", exist_ok=True)
    todo = [f for f in files if not os.path.exists(os.path.join("videos", f))]
    print(f"{len(signs)} signs, {len(files)} clips, {len(todo)} still to download", flush=True)

    batches = [todo[i::args.workers * 4] for i in range(args.workers * 4)]
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(fetch, b, "videos") for b in batches if b]):
            done += fut.result()
            print(f"{done}/{len(todo)} downloaded", flush=True)


if __name__ == "__main__":
    main()
