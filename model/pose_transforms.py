"""
Pose transforms. Each takes a dict whose "frames" entry is a (C, T, V) tensor (coordinates x
frames x points) and returns it with "frames" changed.

From OpenHands (openhands/datasets/pose_transforms.py): Compose, PoseSelect,
CenterAndScaleNormalize, ShearTransform, RotationTransform.
Added for this project: TrimToHandSpan, ConfidenceMask, AddMidpointJoint, TemporalResample.
"""
import numpy as np
import torch

from pose_config import POINTS_27, SHOULDERS_27


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: dict) -> dict:
        for transform in self.transforms:
            data = transform(data)
        return data


class TrimToHandSpan:
    """
    Crops a clip to the frames where at least one hand is detected, plus `margin` frames either
    side, so the idle start and end of ASL Citizen's webcam recordings don't dominate. Clips with
    fewer than 2 such frames are left whole; clips still longer than `max_len` are evenly
    subsampled to `max_len` frames. Used for every run (not an ablation factor).
    Must run first: it reads the 75-point confidences (wrists are points 33 and 54).
    """

    def __init__(self, margin=5, max_len=150):
        self.margin = margin
        self.max_len = max_len

    def __call__(self, data: dict) -> dict:
        x, conf = data["frames"], data["confidences"]  # (C, T, V), (T, V)
        T = conf.shape[0]
        hand = torch.nonzero((conf[:, 33] > 0) | (conf[:, 54] > 0)).flatten()
        if len(hand) >= 2:
            a, b = max(0, int(hand[0]) - self.margin), min(T, int(hand[-1]) + self.margin + 1)
            x, conf = x[:, a:b], conf[a:b]
        if self.max_len and x.shape[1] > self.max_len:
            keep = torch.linspace(0, x.shape[1] - 1, self.max_len).round().long()
            x, conf = x[:, keep], conf[keep]
        data["frames"], data["confidences"] = x, conf
        return data


class ConfidenceMask:
    """
    Sets points whose MediaPipe confidence is below `threshold` to (0, 0). Must run before
    PoseSelect, since the confidences cover all 75 points.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, data: dict) -> dict:
        x = data["frames"]  # (C, T, V=75)
        conf = data["confidences"]  # (T, V=75)
        mask = (conf >= self.threshold).unsqueeze(0).to(x.dtype)  # (1, T, V) broadcasts over C
        data["frames"] = x * mask
        return data


class PoseSelect:
    """Keeps only the points in `pose_indexes` (default: the 27-point set)."""

    def __init__(self, pose_indexes=None):
        self.pose_indexes = pose_indexes or POINTS_27

    def __call__(self, data: dict) -> dict:
        x = data["frames"]
        data["frames"] = x[:, :, self.pose_indexes]
        return data


class AddMidpointJoint:
    """Adds one extra point halfway between points `a` and `b` (the two wrists), as the last point."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, data: dict) -> dict:
        x = data["frames"]  # (C, T, V)
        mid = (x[:, :, self.a] + x[:, :, self.b]) / 2
        data["frames"] = torch.cat([x, mid.unsqueeze(-1)], dim=-1)
        return data


class CenterAndScaleNormalize:
    """
    Moves the midpoint of two reference points (default: the shoulders) to the origin and divides
    by their distance, averaged over the clip (or per frame with frame_level=True).
    """

    def __init__(self, reference_point_indexes=None, scale_factor=1, frame_level=False):
        self.reference_point_indexes = reference_point_indexes or SHOULDERS_27
        self.scale_factor = scale_factor
        self.frame_level = frame_level

    def __call__(self, data: dict) -> dict:
        x = data["frames"]
        C, T, V = x.shape
        x = x.permute(1, 2, 0)  # CTV -> TVC

        if self.frame_level:
            for t in range(x.shape[0]):
                center, scale = self._center_and_scale_for_one_skeleton(x[t])
                x[t] -= center
                x[t] *= scale
        else:
            center, scale = self._center_and_scale(x)
            x = x - center
            x = x * scale

        data["frames"] = x.permute(2, 0, 1)  # TVC -> CTV
        return data

    def _center_and_scale_for_one_skeleton(self, x):
        ind1, ind2 = self.reference_point_indexes
        point1, point2 = x[ind1], x[ind2]
        center = (point1 + point2) / 2
        dist = torch.sqrt(((point1 - point2) ** 2).sum(-1))
        scale = self.scale_factor / dist
        if torch.isinf(scale).any():
            return 0, 1
        return center, scale

    def _center_and_scale(self, x):
        transposed_x = x.permute(1, 0, 2)  # TVC -> VTC
        ind1, ind2 = self.reference_point_indexes
        points1 = transposed_x[ind1].reshape(-1, x.shape[-1])
        points2 = transposed_x[ind2].reshape(-1, x.shape[-1])

        center = torch.mean((points1 + points2) / 2, dim=0)
        mean_dist = torch.mean(torch.sqrt(((points1 - points2) ** 2).sum(-1)))
        scale = self.scale_factor / mean_dist
        if torch.isinf(scale).any():
            return 0, 1
        return center, scale


class ShearTransform:
    """
    2D shear augmentation (train-time only). Shears x and y; a z channel, if present, is passed
    through unchanged, so a 3D run gets the same augmentation in the image plane as a 2D run.
    """

    def __init__(self, shear_std=0.1):
        self.shear_std = shear_std

    def __call__(self, data: dict) -> dict:
        x, rest = data["frames"][:2], data["frames"][2:]  # (x, y) and z (empty for 2D input)
        x = x.permute(1, 2, 0)  # CTV -> TVC
        shear_matrix = torch.eye(2)
        shear_matrix[0][1] = torch.tensor(np.random.normal(loc=0, scale=self.shear_std, size=1)[0])
        res = torch.matmul(x, shear_matrix)
        data["frames"] = torch.cat([res.permute(2, 0, 1), rest])  # TVC -> CTV
        return data


class RotationTransform:
    """
    2D rotation augmentation (train-time only). Rotates in the x-y plane; a z channel, if present,
    is passed through unchanged (a rotation about the camera axis leaves depth as it is).
    """

    def __init__(self, rotation_std=0.1):
        self.rotation_std = rotation_std

    def __call__(self, data: dict) -> dict:
        x, rest = data["frames"][:2], data["frames"][2:]  # (x, y) and z (empty for 2D input)
        x = x.permute(1, 2, 0)  # CTV -> TVC
        angle = torch.tensor(np.random.normal(loc=0, scale=self.rotation_std, size=1)[0])
        cos, sin = torch.cos(angle), torch.sin(angle)
        rotation_matrix = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float32)
        res = torch.matmul(x, rotation_matrix)
        data["frames"] = torch.cat([res.permute(2, 0, 1), rest])  # TVC -> CTV
        return data


class TemporalResample:
    """Stretches or squeezes every clip to exactly `target_length` frames (linear interpolation)."""

    def __init__(self, target_length=64):
        self.target_length = target_length

    def __call__(self, data: dict) -> dict:
        x = data["frames"]  # (C, T, V)
        C, T, V = x.shape
        x = x.permute(0, 2, 1).reshape(1, C * V, T)  # (1, C*V, T) for F.interpolate
        x = torch.nn.functional.interpolate(x, size=self.target_length, mode="linear", align_corners=False)
        x = x.reshape(C, V, self.target_length).permute(0, 2, 1)  # -> (C, target_length, V)
        data["frames"] = x
        return data
