"""
Cache PHASE-Net's raw output for every UBFC subject, once.

WHY
  Running the model costs ~8 s per subject. Comparing heart-rate extraction
  methods should not pay that cost every time. This script runs the model once
  and saves the raw predicted signal plus the reference BVP for each subject, so
  any number of HR-extraction strategies can then be benchmarked in seconds - and
  compared fairly, because they all see exactly the same signals.

OUTPUT
  signal_cache.npz  - per subject: model output, reference BVP, fs, train/test split

USAGE
    python 05_cache_signals.py --data "D:\\00-TU-CLAUSTHAL\\keiko-rppg\\UBFC"
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _common import (CLIP_LEN, FS, load_model, read_and_crop_video,  # noqa: E402
                     read_wave, train_test_split)

CACHE = os.path.join(HERE, "signal_cache.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=CACHE)
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    model = load_model(device)

    subjects = sorted(glob.glob(os.path.join(args.data, "subject*")),
                      key=lambda p: int(re.search(r"subject(\d+)", p).group(1)))
    subjects = [s for s in subjects
                if os.path.isfile(os.path.join(s, "vid.avi"))
                and os.path.isfile(os.path.join(s, "ground_truth.txt"))]
    if not subjects:
        sys.exit("No complete subjects found.")

    train_split, test_split = train_test_split()
    print(f"Caching {len(subjects)} subject(s) on {device}\n")

    store = {}
    for sd in subjects:
        sid = os.path.basename(sd)
        t0 = time.time()
        faces, found = read_and_crop_video(os.path.join(sd, "vid.avi"))
        bvps = read_wave(os.path.join(sd, "ground_truth.txt"))

        n = min(len(faces), len(bvps)) // CLIP_LEN
        if n == 0:
            print(f"  {sid}: too short - skipped")
            continue
        faces, bvps = faces[:n * CLIP_LEN], bvps[:n * CLIP_LEN]

        preds = []
        for c in range(n):
            clip = faces[c * CLIP_LEN:(c + 1) * CLIP_LEN]
            x = np.transpose(clip, (3, 0, 1, 2)).astype(np.float32)   # (3,T,H,W), 0-255
            with torch.no_grad():
                out, _ = model(torch.from_numpy(x).unsqueeze(0).to(device))
            preds.append(out[0].float().cpu().numpy())
        pred = np.concatenate(preds)

        split = "test" if sid in test_split else ("train" if sid in train_split else "?")
        store[f"{sid}__pred"] = pred.astype(np.float32)
        store[f"{sid}__ref"] = np.asarray(bvps, dtype=np.float32)
        store[f"{sid}__meta"] = np.array([n, FS, 1 if found else 0], dtype=np.int32)
        store[f"{sid}__split"] = np.array(split)
        print(f"  {sid:<12} {n:2d} clips, {len(pred):5d} samples, "
              f"{split:<5} [{time.time()-t0:4.1f}s]")

    np.savez_compressed(args.out, **store)
    ids = sorted({k.split("__")[0] for k in store}, key=lambda s: int(s.replace("subject", "")))
    print(f"\nCached {len(ids)} subject(s) -> {args.out} "
          f"({os.path.getsize(args.out)/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
