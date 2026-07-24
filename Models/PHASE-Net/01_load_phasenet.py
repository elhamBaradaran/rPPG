"""
Milestone 1 - Load PHASE-Net and run it once.

GOAL: prove that (a) the model code imports, (b) the pretrained UBFC weights load
correctly, and (c) the model can process a video clip on this machine.

We do NOT need the dataset or the full rPPG-Toolbox pipeline for this - we import
ONLY the model class, which avoids the heavy TensorFlow/retinaface dependency.

Run it with:
    python 01_load_phasenet.py
"""

import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# 1) Tell Python where the PhaseNet repo lives, so we can import its model code
# ---------------------------------------------------------------------------
REPO = r"D:\00-TU-CLAUSTHAL\keiko-rppg\PhaseNet"
sys.path.insert(0, REPO)

WEIGHTS = os.path.join(REPO, "weights", "phasenet_ubfc_epoch9.pth")

# This import reaches ONLY the model file - no data loaders, so no TensorFlow.
from neural_methods.model.PhaseNet.PhaseNet import PhaseNet  # noqa: E402


def main():
    print("=" * 70)
    print("PHASE-Net - Milestone 1: load model + pretrained weights")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 2) Build the model with EXACTLY the settings the checkpoint was saved with
    #    (these came from configs/*.yaml and the trainer code)
    # -----------------------------------------------------------------------
    model = PhaseNet(
        feature_dim=128,
        latent_dim=32,
        hidden_dim=128,
        tcn_layers=4,
        encoder_channels=[16, 32, 64, 128],
        encoder_expand_ratio=4,
        temporal_module="gated_tcn",
    )
    print("\n[1] Model object created OK")

    # -----------------------------------------------------------------------
    # 3) Load the pretrained weights (trained on UBFC-rPPG by the authors)
    # -----------------------------------------------------------------------
    state_dict = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[2] Weights loaded from:", os.path.basename(WEIGHTS))
    print("    missing keys   (in model, not in file):", len(missing))
    print("    unexpected keys(in file, not in model):", len(unexpected))
    if missing:
        print("    -> missing sample:", missing[:5])
    if unexpected:
        print("    -> unexpected sample:", unexpected[:5])
    if not missing and not unexpected:
        print("    *** PERFECT MATCH - every weight found its place ***")

    # -----------------------------------------------------------------------
    # 4) Count parameters (compare with the paper's claim of 0.29 M)
    # -----------------------------------------------------------------------
    total = sum(p.numel() for p in model.parameters())
    # These two submodules are only used during TRAINING (reconstruction loss),
    # so they do not count toward inference cost.
    train_only = sum(p.numel() for n, p in model.named_parameters()
                     if n.startswith("decoder.") or n.startswith("projection_encoder."))
    print("\n[3] Parameter count")
    print("    total (whole checkpoint) : {:,} ({:.3f} M)".format(total, total / 1e6))
    print("    training-only modules    : {:,} ({:.3f} M)".format(train_only, train_only / 1e6))
    print("    inference path           : {:,} ({:.3f} M)".format(
        total - train_only, (total - train_only) / 1e6))
    print("    paper claims             : 0.290 M")

    # -----------------------------------------------------------------------
    # 5) Run the model once on a fake video clip
    #    Shape = (batch, RGB, frames, height, width) = (1, 3, 128, 128, 128)
    #    128 frames at 30 fps  ~=  4.3 seconds of video
    # -----------------------------------------------------------------------
    model.eval()  # eval mode: turns off dropout, skips the reconstruction loss
    dummy = torch.randn(1, 3, 128, 128, 128)
    print("\n[4] Test input shape:", tuple(dummy.shape), "(1 clip, RGB, 128 frames, 128x128)")

    device_used = None
    pred = None

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model_gpu = model.to("cuda")
            x = dummy.to("cuda")
            t0 = time.time()
            with torch.no_grad():           # no_grad = inference only, saves lots of memory
                pred, _ = model_gpu(x)
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3
            device_used = "GPU (cuda)"
            print("\n[5] Forward pass on GPU: SUCCESS")
            print("    time            : {:.2f} s".format(dt))
            print("    peak GPU memory : {:.2f} GB (your card has 4.00 GB)".format(peak))
        except torch.cuda.OutOfMemoryError:
            print("\n[5] GPU ran OUT OF MEMORY - falling back to CPU (this is OK)")
            torch.cuda.empty_cache()
            model = model.cpu()
            pred = None
        except Exception as e:
            print("\n[5] GPU attempt failed:", type(e).__name__, e)
            torch.cuda.empty_cache()
            model = model.cpu()
            pred = None
    else:
        print("\n[5] No CUDA available - using CPU")

    if pred is None:
        t0 = time.time()
        with torch.no_grad():
            pred, _ = model(dummy)
        dt = time.time() - t0
        device_used = "CPU"
        print("    CPU forward pass: SUCCESS  ({:.2f} s)".format(dt))

    # -----------------------------------------------------------------------
    # 6) Look at what came out: the predicted rPPG pulse waveform
    # -----------------------------------------------------------------------
    pred = pred.detach().float().cpu()
    print("\n[6] Output (the predicted pulse waveform)")
    print("    device used  :", device_used)
    print("    output shape :", tuple(pred.shape), "-> one value per frame")
    print("    min / max    : {:.4f} / {:.4f}".format(pred.min().item(), pred.max().item()))
    print("    first 8 values:", [round(v, 4) for v in pred[0, :8].tolist()])

    print("\n" + "=" * 70)
    print("MILESTONE 1 COMPLETE - PHASE-Net runs on your machine.")
    print("(The numbers are meaningless here because the input was random noise;")
    print(" next step is feeding it a REAL face video.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
