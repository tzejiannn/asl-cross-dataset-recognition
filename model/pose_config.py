"""
Keypoint sets and skeleton graphs.

Each pose file holds 75 points per frame (see pose_extraction.py):
    0-32   body       (MediaPipe Pose; 0 nose, 2/5 eyes, 11/12 shoulders, 13/14 elbows)
    33-53  left hand  (MediaPipe Hands, 21 points, first one = wrist)
    54-74  right hand (same layout)

POINTS_27 and GRAPH_27 are copied from OpenHands (`mediapipe_holistic_minimal_27`). The hand-only
sets are this project's additions for the ablation study.

A point set lists which of the 75 points to keep. A graph's node numbers are positions within its
point set (not the 75-point numbers); `center` is the node the ST-GCN partitions neighbours around,
and `inward_edges` are the bones, each pointing towards the center.
"""

# --- 27 points (OpenHands): 7 body points + 10 per hand -------------------------------------
# 10 hand points = wrist, thumb tip, and the base (MCP) and tip of each of the other four fingers
POINTS_27 = [
    0, 2, 5, 11, 12, 13, 14,                  # nose, eyes, shoulders, elbows
    33, 37, 38, 41, 42, 45, 46, 49, 50, 53,   # left hand
    54, 58, 59, 62, 63, 66, 67, 70, 71, 74,   # right hand
]
# nodes: 0 nose, 1-2 eyes, 3-4 shoulders, 5-6 elbows, 7-16 left hand (7 = wrist), 17-26 right hand (17 = wrist)
GRAPH_27 = dict(num_nodes=27, center=0, inward_edges=[
    [2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17],
    [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], [7, 15], [15, 16],
    [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], [17, 23], [23, 24], [17, 25], [25, 26],
])
SHOULDERS_27 = [3, 4]  # normalisation reference (OpenHands' default)
WRISTS_27 = [7, 17]

# --- 20 points: the same 10 points per hand, without the body ---------------------------------
POINTS_HANDS_20 = POINTS_27[7:]
# nodes: 0-9 left hand (0 = wrist), 10-19 right hand (10 = wrist). Without arms the two hands
# would be separate graphs, so an extra wrist-to-wrist edge [0, 10] joins them.
GRAPH_HANDS_20 = dict(num_nodes=20, center=0, inward_edges=[
    [0, 1], [0, 2], [2, 3], [0, 4], [4, 5], [0, 6], [6, 7], [0, 8], [8, 9],
    [10, 11], [10, 12], [12, 13], [10, 14], [14, 15], [10, 16], [16, 17], [10, 18], [18, 19],
    [0, 10],
])
WRISTS_HANDS_20 = [0, 10]

# --- 42 points: all 21 points per hand ------------------------------------------------------------
POINTS_HANDS_42 = list(range(33, 75))


def _hand_edges(wrist):
    """Wrist to the first joint of each finger, then along the finger to its tip."""
    edges = []
    for first in (1, 5, 9, 13, 17):  # thumb, index, middle, ring, pinky
        edges.append([wrist, wrist + first])
        edges += [[wrist + j, wrist + j + 1] for j in range(first, first + 3)]
    return edges


# nodes: 0-20 left hand (0 = wrist), 21-41 right hand (21 = wrist), plus the wrist-to-wrist edge
GRAPH_HANDS_42 = dict(num_nodes=42, center=0, inward_edges=_hand_edges(0) + _hand_edges(21) + [[0, 21]])
WRISTS_HANDS_42 = [0, 21]

# --- Shoulders in the full 75-point array -----------------------------------------------------
# for conditions that normalise on the shoulders BEFORE selecting the hand points
SHOULDERS_75 = [11, 12]
