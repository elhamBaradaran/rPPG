"""
Compute the POS baseline on the SAME videos and the SAME face crops as PHASE-Net.

WHY THIS SCRIPT EXISTS
  Comparing a classical method against a deep model is only meaningful if the only
  thing that differs is the algorithm. So this script deliberately reuses
  `_common.read_and_crop_video` - the identical Haar face box, the identical 1.5x
  enlargement, the identical 128x128 crop that PHASE-Net was fed. From those exact
  frames it computes the spatially averaged RGB trace that POS works from.

TWO VARIANTS OF POS ARE COMPUTED
  1. "reference" - the repository's own `POS_WANG`, i.e. the implementation used by
     the rPPG-Toolbox and the PHASE-Net paper's comparison table. It follows Wang et
     al. (2017) faithfully: a 1.6-second sliding window with overlap-add.
  2. "ours" - the simplified version written from scratch in Models/POS. The
     projection is mathematically identical; what differs is that it applies the
     formula to one long block instead of a sliding window, and uses a wider
     band-pass (0.7-4.0 Hz, order 3 instead of 0.75-3.0 Hz, order 1).

  Comparing the two answers a second, separate question: how much did the
  simplification in our own implementation cost?

OUTPUT
  pos_cache.npz - per subject: the RGB trace and both POS signals.
  The raw RGB trace is stored too, so other POS variants can be tried later without
  re-reading 24 GB of video.

USAGE
    python 09_pos_baseline.py --data "D:\\00-TU-CLAUSTHAL\\keiko-rppg\\UBFC"
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
from scipy.signal import butter, filtfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _common import (CLIP_LEN, FS, REPO, read_and_crop_video,   # noqa: E402
                     read_wave, train_test_split)

# --- NumPy 2.x compatibility shim -------------------------------------------
# The repository's POS was written against NumPy 1.x and calls np.mat(), which was
# removed in NumPy 2.0. Rather than editing the authors' file - which would mean we
# were no longer running the reference implementation - we restore the old alias to
# its documented replacement. The algorithm itself is untouched.
if not hasattr(np, "mat"):
    np.mat = np.asmatrix
# -----------------------------------------------------------------------------

# The repository's own POS implementation - the reference the paper compares against.
sys.path.insert(0, REPO)
from unsupervised_methods.methods.POS_WANG import POS_WANG      # noqa: E402

CACHE = os.path.join(HERE, "pos_cache.npz")


def rgb_trace(frames):
    """Spatial mean of R, G, B for every frame -> (T, 3).

    This is all POS needs: one colour triplet per frame. The pulse is a tiny
    fluctuation in that triplet, far too small to see by eye.
    """
    return frames.reshape(len(frames), -1, 3).mean(axis=1)


def pos_ours(rgb, fs):
    """The simplified POS written from scratch in Models/POS/heartrate.py.

    Same projection as Wang et al.:
        s1 = G - B
        s2 = G + B - 2R          (after dividing each channel by its own mean)
        pulse = s1 + (std(s1)/std(s2)) * s2
    but applied to the whole recording at once rather than a sliding window.
    """
    x = np.asarray(rgb, dtype=np.float64).T          # (3, T) -> R, G, B rows
    mean = np.mean(x, axis=1, keepdims=True)
    n = x / np.where(mean == 0, 1e-9, mean)          # per-channel normalisation
    s1 = n[1] - n[2]
    s2 = n[1] + n[2] - 2 * n[0]
    alpha = np.std(s1) / (np.std(s2) + 1e-9)
    pulse = s1 + alpha * s2
    pulse = pulse - np.mean(pulse)
    # the band-pass used in Models/POS: 0.7-4.0 Hz, 3rd order
    b, a = butter(3, [0.7 / (fs / 2), 4.0 / (fs / 2)], btype="band")
    return filtfilt(b, a, pulse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=CACHE)
    args = ap.parse_args()

    subjects = sorted(glob.glob(os.path.join(args.data, "subject*")),
                      key=lambda p: int(re.search(r"subject(\d+)", p).group(1)))
    subjects = [s for s in subjects
                if os.path.isfile(os.path.join(s, "vid.avi"))
                and os.path.isfile(os.path.join(s, "ground_truth.txt"))]
    if not subjects:
        sys.exit("No complete subjects found.")

    train_split, test_split = train_test_split()
    print(f"Computing POS for {len(subjects)} subject(s)")
    print("Using the same face crops PHASE-Net was given, so only the algorithm differs.\n")
    print(f"{'subject':<12}{'frames':>8}{'split':>7}   time")
    print("-" * 40)

    store = {}
    for sd in subjects:
        sid = os.path.basename(sd)
        t0 = time.time()

        frames, _ = read_and_crop_video(os.path.join(sd, "vid.avi"))
        bvps = read_wave(os.path.join(sd, "ground_truth.txt"))

        # trim to whole clips, exactly as the PHASE-Net cache did, so the two
        # caches line up sample for sample
        n = min(len(frames), len(bvps)) // CLIP_LEN
        if n == 0:
            print(f"{sid:<12}  too short - skipped")
            continue
        frames = frames[:n * CLIP_LEN]

        rgb = rgb_trace(frames)
        sig_ref = np.asarray(POS_WANG(frames, FS), dtype=np.float64)
        sig_ours = pos_ours(rgb, FS)

        split = "test" if sid in test_split else ("train" if sid in train_split else "?")
        store[f"{sid}__rgb"] = rgb.astype(np.float32)
        store[f"{sid}__pos_ref"] = sig_ref.astype(np.float32)
        store[f"{sid}__pos_ours"] = sig_ours.astype(np.float32)
        store[f"{sid}__split"] = np.array(split)
        print(f"{sid:<12}{len(frames):>8}{split:>7}   [{time.time()-t0:5.1f}s]")

    np.savez_compressed(args.out, **store)
    ids = sorted({k.split("__")[0] for k in store},
                 key=lambda s: int(s.replace("subject", "")))
    print(f"\nCached {len(ids)} subject(s) -> {args.out} "
          f"({os.path.getsize(args.out)/1024**2:.1f} MB)")
    print("Next:  python 10_compare_models.py --data \"<UBFC folder>\"")


if __name__ == "__main__":
    main()
