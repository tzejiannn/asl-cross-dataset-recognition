"""
Runs pose_extraction.dump_pose_for_dataset over a video folder with N worker processes. Identical
extraction settings to the single-process script (same MediaPipe Holistic legacy API, same
model_complexity, same output format), so poses from different datasets stay comparable. Already
extracted clips are skipped, so it can be re-run to resume.

Usage (from this folder):
    python pose_extraction_parallel.py --videos_dir ../04_asl_citizen_download/videos \
        --out_dir poses_asl_citizen --workers 12
"""
import argparse
import os
from multiprocessing import Pool

from pose_extraction import dump_pose_for_dataset


def _run(job):
    videos_dir, out_dir, ids, complexity = job
    dump_pose_for_dataset(videos_dir, out_dir, video_ids=ids, model_complexity=complexity)
    return len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model_complexity", type=int, default=2, choices=[0, 1, 2])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(args.videos_dir) if f.lower().endswith(".mp4"))
    todo = [v for v in ids if not os.path.exists(os.path.join(args.out_dir, v + ".pkl"))]
    print(f"{len(ids)} videos, {len(todo)} to extract, {args.workers} workers", flush=True)
    chunks = [todo[i::args.workers * 8] for i in range(args.workers * 8)]
    done = 0
    with Pool(args.workers) as pool:
        for n in pool.imap_unordered(_run, [(args.videos_dir, args.out_dir, c, args.model_complexity) for c in chunks if c]):
            done += n
            print(f"{done}/{len(todo)} extracted", flush=True)


if __name__ == "__main__":
    main()
