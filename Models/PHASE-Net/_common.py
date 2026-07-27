"""
Shared pieces used by the PHASE-Net scripts: model loading, UBFC preprocessing,
and the train/test split of the released checkpoint.

Kept in one place so the evaluator, the signal cache, and any benchmark all see
identical preprocessing - otherwise a comparison between them means nothing.
"""

import glob
import os
import re
import sys

import cv2
import numpy as np
import torch

REPO = r"D:\00-TU-CLAUSTHAL\keiko-rppg\PhaseNet"
sys.path.insert(0, REPO)

WEIGHTS = os.path.join(REPO, "weights", "phasenet_ubfc_epoch9.pth")
HAAR = os.path.join(REPO, "dataset", "haarcascade_frontalface_default.xml")

# Must match how the checkpoint was trained (from the repo's config YAML)
CLIP_LEN = 128
IMG_SIZE = 128
FS = 30
LARGE_BOX_COEF = 1.5

from neural_methods.model.PhaseNet.PhaseNet import PhaseNet  # noqa: E402


def load_model(device):
    model = PhaseNet(
        feature_dim=128, latent_dim=32, hidden_dim=128, tcn_layers=4,
        encoder_channels=[16, 32, 64, 128], encoder_expand_ratio=4,
        temporal_module="gated_tcn",
    )
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


def detect_face_box(frame_rgb):
    """Haar cascade, biggest face, box enlarged 1.5x about its centre."""
    detector = cv2.CascadeClassifier(HAAR)
    faces = detector.detectMultiScale(frame_rgb)
    if len(faces) < 1:
        return None
    box = list(faces[int(np.argmax(faces[:, 2]))]) if len(faces) >= 2 else list(faces[0])
    box[0] = max(0, box[0] - (LARGE_BOX_COEF - 1.0) / 2 * box[2])
    box[1] = max(0, box[1] - (LARGE_BOX_COEF - 1.0) / 2 * box[3])
    box[2] = LARGE_BOX_COEF * box[2]
    box[3] = LARGE_BOX_COEF * box[3]
    return np.asarray(box, dtype=int)


def read_and_crop_video(video_file):
    """Stream a video, detect the face on the FIRST frame only (as in training),
    and crop every frame to 128x128. Returns (frames_uint8, face_found)."""
    cap = cv2.VideoCapture(video_file)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read {video_file}")
    first_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    box = detect_face_box(first_rgb)

    def crop(rgb):
        f = rgb
        if box is not None:
            f = f[max(box[1], 0):min(box[1] + box[3], f.shape[0]),
                  max(box[0], 0):min(box[0] + box[2], f.shape[1])]
        return cv2.resize(f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    faces = [crop(first_rgb)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces.append(crop(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return np.asarray(faces, dtype=np.uint8), (box is not None)


def read_wave(bvp_file):
    """UBFC ground_truth.txt: line 1 is the reference BVP signal."""
    with open(bvp_file, "r") as f:
        first = f.read().split("\n")[0]
    return np.asarray([float(x) for x in first.split()])


def read_device_hr(bvp_file):
    """UBFC ground_truth.txt line 2: the heart rate reported by the CMS50E itself.

    This matters because it is an INDEPENDENT reference. The usual rPPG convention
    estimates the reference HR by running the same FFT pipeline over the ground-truth
    BVP waveform - but that estimate can itself be wrong, which makes a correct
    prediction look like an error. The oximeter's own reading needs no signal
    processing from us, so it can arbitrate.

    Returns the full per-sample series; take the median for a robust video-level
    value, since the device emits occasional dropout samples (values near 0-1).
    """
    with open(bvp_file, "r") as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    if len(lines) < 2:
        return None
    hr = np.asarray([float(x) for x in lines[1].split()])
    return hr


def device_hr_median(bvp_file, low=40.0, high=180.0):
    """Robust video-level device HR: median over physiologically plausible samples."""
    hr = read_device_hr(bvp_file)
    if hr is None or hr.size == 0:
        return float("nan")
    valid = hr[(hr >= low) & (hr <= high)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


# The full official DATASET_2 subject list (42 subjects).
ALL_SUBJECTS = sorted([
    "subject1", "subject3", "subject4", "subject5", "subject8", "subject9",
    "subject10", "subject11", "subject12", "subject13", "subject14", "subject15",
    "subject16", "subject17", "subject18", "subject20", "subject22", "subject23",
    "subject24", "subject25", "subject26", "subject27", "subject30", "subject31",
    "subject32", "subject33", "subject34", "subject35", "subject36", "subject37",
    "subject38", "subject39", "subject40", "subject41", "subject42", "subject43",
    "subject44", "subject45", "subject46", "subject47", "subject48", "subject49",
])


def train_test_split(begin=0.72):
    """Reproduce the split of the released 'UBFC_UBFC_072_100' checkpoint.

    BaseLoader.split_raw_data takes range(int(begin*N), int(end*N)) over the
    glob-ordered subject list, so the model was TRAINED on the first 72% and
    tested on the rest. Scoring a training subject measures memorisation.
    """
    n = len(ALL_SUBJECTS)
    cut = int(begin * n)
    return ALL_SUBJECTS[:cut], ALL_SUBJECTS[cut:]
