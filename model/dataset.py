"""
Loads the extracted pose files (pose_extraction.py) for one subset of a split file.

A split file (split.json) maps each video_id to {"subset": ..., "action": [class_id, 1, -1]}, the
format of WLASL's nslt_100.json. The clips are already trimmed to a single sign, so the frame range
in "action" isn't used.
"""
import json
import os
import pickle

import torch
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    """
    Args:
        split_file: path to the split file.
        subset: which subset to load, e.g. "train", "val", "test" or "wlasl".
        pose_dirs: folder, or list of folders searched in order, holding {video_id}.pkl files.
        transforms: applied to each clip (see ablation_configs.build_transforms).
        use_z: keep the z coordinate (3 channels) instead of only x and y.
    """

    def __init__(self, split_file, subset, pose_dirs, transforms, use_z=False):
        pose_dirs = [pose_dirs] if isinstance(pose_dirs, str) else pose_dirs
        self.transforms = transforms
        self.use_z = use_z

        self.data = []  # (video_id, class_id, pose_path), in split-file order
        missing = 0
        for video_id, entry in json.load(open(split_file)).items():
            if entry["subset"] != subset:
                continue
            paths = [os.path.join(d, video_id + ".pkl") for d in pose_dirs]
            pose_path = next((p for p in paths if os.path.isfile(p)), None)
            if pose_path is None:
                missing += 1
            else:
                self.data.append((video_id, entry["action"][0], pose_path))
        if missing:
            print(f"PoseDataset: {missing} clip(s) in subset '{subset}' have no pose file and were skipped.")
        if not self.data:
            raise RuntimeError(f"No pose files found for subset '{subset}' in {pose_dirs}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, label, pose_path = self.data[index]
        pose = pickle.load(open(pose_path, "rb"))
        kps = pose["keypoints"] if self.use_z else pose["keypoints"][:, :, :2]  # (T, 75, C)
        data = {
            "frames": torch.tensor(kps, dtype=torch.float32).permute(2, 0, 1),  # (C, T, 75)
            "confidences": torch.tensor(pose["confidences"], dtype=torch.float32),  # (T, 75)
            "label": torch.tensor(label, dtype=torch.long),
            "file": video_id,
        }
        return self.transforms(data)

    @staticmethod
    def collate_fn(batch):
        """Stacks a batch, zero-padding every clip at the end to the longest clip's length."""
        max_frames = max(x["frames"].shape[1] for x in batch)
        frames = torch.stack([torch.nn.functional.pad(x["frames"], (0, 0, 0, max_frames - x["frames"].shape[1]))
                              for x in batch])
        labels = torch.stack([x["label"] for x in batch])
        return {"frames": frames, "labels": labels, "files": [x["file"] for x in batch]}
