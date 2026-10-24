"""
Extracts MediaPipe Holistic pose keypoints for WLASL videos, saving one .pkl file per video in
the exact on-disk format OpenHands uses (see scripts/mediapipe_extract.py):

    {"keypoints": np.ndarray (T, 75, 3), "confidences": np.ndarray (T, 75)}

75 = 33 body + 21 left-hand + 21 right-hand points (face mesh excluded). `pose_transforms.
PoseSelect` later reduces this to the 27-point `mediapipe_holistic_minimal_27` subset.

Expects trimmed per-instance videos named `{video_id}.mp4`, i.e. the WLASL clips in
`../03_wlasl_filter/videos/`.

Usage (from this folder):
    python pose_extraction.py --videos_dir ../03_wlasl_filter/videos --out_dir poses_wlasl
"""
import argparse
import gc
import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

N_FACE_LANDMARKS = 468
N_BODY_LANDMARKS = 33
N_HAND_LANDMARKS = 21


def _process_landmarks(component, n_points, is_body):
    kps = np.zeros((n_points, 3))
    conf = np.zeros(n_points)
    if component is not None:
        landmarks = component.landmark
        kps = np.array([[p.x, p.y, p.z] for p in landmarks])
        conf = np.array([p.visibility for p in landmarks]) if is_body else np.ones(n_points)
    return kps, conf


def load_frames_from_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def extract_keypoints_for_video(video_path, holistic):
    frames = load_frames_from_video(video_path)
    if not frames:
        return None

    keypoints, confs = [], []
    for frame in frames:
        results = holistic.process(frame)

        body_kps, body_conf = _process_landmarks(results.pose_landmarks, N_BODY_LANDMARKS, is_body=True)
        lh_kps, lh_conf = _process_landmarks(results.left_hand_landmarks, N_HAND_LANDMARKS, is_body=False)
        rh_kps, rh_conf = _process_landmarks(results.right_hand_landmarks, N_HAND_LANDMARKS, is_body=False)

        # Face landmarks are intentionally never requested/concatenated: mediapipe_holistic_minimal_27
        # only draws from body + both hands (see pose_config.py).
        keypoints.append(np.concatenate([body_kps, lh_kps, rh_kps]))
        confs.append(np.concatenate([body_conf, lh_conf, rh_conf]))

    return {"keypoints": np.stack(keypoints), "confidences": np.stack(confs)}


def dump_pose_for_dataset(videos_dir, out_dir, video_ids=None, model_complexity=2):
    import mediapipe as mp

    os.makedirs(out_dir, exist_ok=True)

    if video_ids is None:
        video_ids = sorted(
            os.path.splitext(f)[0] for f in os.listdir(videos_dir) if f.lower().endswith(".mp4")
        )

    skipped = []
    for video_id in tqdm(video_ids, desc="Extracting poses"):
        save_path = os.path.join(out_dir, video_id + ".pkl")
        if os.path.exists(save_path):
            continue

        video_path = os.path.join(videos_dir, video_id + ".mp4")
        if not os.path.isfile(video_path):
            skipped.append(video_id)
            continue

        # A fresh Holistic instance per video avoids a known mediapipe memory-leak issue
        # (https://github.com/google/mediapipe/issues/2152), same as OpenHands' extractor.
        holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=model_complexity)
        try:
            data = extract_keypoints_for_video(video_path, holistic)
        finally:
            holistic.close()
            del holistic
            gc.collect()

        if data is None:
            skipped.append(video_id)
            continue

        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=4)

    if skipped:
        print(f"Skipped {len(skipped)} video(s) with no readable frames / missing file: {skipped[:10]}"
              f"{' ...' if len(skipped) > 10 else ''}")
    print(f"Pose data saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_dir", required=True, help="Directory of trimmed {video_id}.mp4 files")
    parser.add_argument("--out_dir", required=True, help="Output directory for {video_id}.pkl pose files")
    parser.add_argument("--model_complexity", type=int, default=2, choices=[0, 1, 2])
    args = parser.parse_args()

    dump_pose_for_dataset(args.videos_dir, args.out_dir, model_complexity=args.model_complexity)
