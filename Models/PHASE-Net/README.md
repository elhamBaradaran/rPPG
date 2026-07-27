# KEIKO rPPG — PHASE-Net (Main Model)

A physics-grounded deep-learning rPPG model that estimates heart rate from
ordinary camera video. This is the **main model** of the Master's project on
physiological monitoring during human–robot collaboration (the "KEIKO"
scenario), replacing the classical [POS baseline](../POS) built earlier.

PHASE-Net was published as a **CVPR 2026 Highlight** paper. Unlike most deep
rPPG networks, its architecture is not chosen by trial and error — it is
*derived* from the fluid dynamics of blood flow.

> **Status:** validated against UBFC-rPPG ground truth.
> On 6 truly held-out subjects: **MAE 1.46 BPM**, with **5 of 6 predicted exactly**
> (0.0 BPM error). See [Results](#results-ubfc-rppg-validation).

## What it does

The model takes a short clip of face video (128 frames at 128×128 px) and
outputs a **pulse waveform** — one value per frame. That waveform is then
turned into a heart rate (BPM) by detrending, band-pass filtering to the human
heart-rate range, and taking the dominant FFT frequency.

Compared with the POS baseline, the goal is **robustness**: POS is a fixed
formula that breaks under head motion and lighting change (our baseline dropped
to ~40 BPM while moving, locking onto the motion frequency). PHASE-Net learns
to suppress those artefacts.

## Method

### The physics chain

The paper's central argument is that existing deep rPPG models are *heuristic* —
architectures found by trial and error, which leads to overfitting on
dataset-specific noise and poor interpretability. PHASE-Net instead derives its
architecture from first principles:

| Step | Physics | Result |
|------|---------|--------|
| 1 | Beer–Lambert law + vessel compliance | Pixel intensity change ∝ blood-volume change ∝ pressure pulsation. The pulse `z(t)` is physically embedded in the video. |
| 2 | Navier–Stokes equations, linearised and 1-D averaged | A **damped wave equation** for pressure-pulse propagation. |
| 3 | Single-point observation (one fixed facial location) | Reduces to a second-order ODE — a **forced damped harmonic oscillator**: `z''(t) + α·z'(t) + ω²·z(t) = u(t)` |
| 4 | Discretised (semi-implicit Euler) → LTI state-space model | **Proposition 1:** the solution *is* a **causal convolution** of past inputs. |
| 5 | IIR → FIR approximation | **Proposition 2:** that is exactly what a **Temporal Convolutional Network (TCN)** computes. |

So the TCN is not a design guess — it is the mathematically mandated choice.
The authors state this is the first work to build a theoretical bridge between
the underlying physiological dynamics and a specific network architecture.

### Architecture

```
video → Vision Encoder (3 EST blocks) → Adaptive Spatial Filter → Gated TCN → pulse waveform
```

- **ZAS (Zero-FLOPs Axial Swapper)** — a parameter-free, reversible 2×2 block
  transpose on a subset of channels, mixing distant facial regions (forehead ↔
  cheeks) at **zero computational cost**. Proven self-inverting and
  energy-preserving (1-Lipschitz), which keeps training stable.
- **ASF (Adaptive Spatial Filter)** — learns a per-frame spatial softmax mask
  that highlights pulse-rich skin and suppresses noisy areas, aggregates
  spatially, then concatenates the first-order temporal derivative
  (`vₜ = zₜ − zₜ₋₁`) to encode local pulse dynamics.
- **GTCN (Gated TCN)** — causal dilated TCN with `tanh` × `sigmoid` gating; the
  physics-mandated temporal core.
- **Loss** — Negative Pearson Correlation (waveform-shape similarity), with an
  auxiliary reconstruction term weighted λ = 0.1.

## Why PHASE-Net

**Efficiency** (paper Table 5, at 128×128 with T = 128):

| Method | Params (M) | MACs (G) |
|--------|-----------|----------|
| **PHASE-Net** | **0.29** | **28.3** |
| RhythmFormer | 4.21 | 28.8 |
| PhysFormer | 7.38 | 40.5 |
| PhysNet | 0.77 | 56.1 |
| DeepPhys | 7.50 | 96.0 |

**Accuracy, same-dataset** (MAE in bpm, lower is better):

| Method | UBFC-rPPG | PURE | BUAA | MMPD |
|--------|-----------|------|------|------|
| POS (our baseline method) | 4.08 | 3.67 | — | 12.36 |
| PhysNet | 2.95 | 2.10 | 10.89 | 4.80 |
| RhythmFormer | 0.50 | 0.27 | 9.19 | 4.69 |
| **PHASE-Net** | **0.15** | **0.14** | **5.89** | **4.78** |

**Cross-dataset generalisation** (leave-one-out — train on three datasets, test
on the fourth). This matters most for KEIKO, because a robot lab will not look
like the training data:

| Test on → | PHASE-Net | RhythmFormer | PhysFormer |
|-----------|-----------|--------------|------------|
| PURE | **2.86** | 21.11 | 19.75 |
| BUAA | **2.56** | 6.04 | 22.09 |
| UBFC | **10.04** | 14.71 | 10.29 |
| MMPD | **10.33** | 16.14 | 13.90 |

On PURE it generalises roughly an **order of magnitude** better than
RhythmFormer, which supports the claim that it learns pulse *physics* rather
than dataset-specific appearance cues.

Worth noting: **PURE** is itself titled *"Non-contact video-based pulse rate
measurement on a mobile service robot"* — a human–robot context directly
analogous to KEIKO.

**PHASE-Net over RhythmFormer:** both build on the same rPPG-Toolbox framework,
so implementation effort is comparable. PHASE-Net is ~14× smaller in parameters
and generalises considerably better across datasets. RhythmFormer stays a viable
fallback inside the same toolbox.

## Files

| File | Purpose |
|------|---------|
| `01_load_phasenet.py` | Build the model, load the pretrained weights, verify every key matches, report parameter counts, and time a GPU forward pass. |
| `02_webcam_phasenet.py` | **Full demo:** webcam → face crop → PHASE-Net → BPM, plus a saved waveform plot. Includes a `--selftest` mode that needs no webcam. |
| `PHASE-Net_Report.docx` | The same content as a formatted Word report. |

## Requirements

- Python 3.10+ (developed on 3.12.7)
- An NVIDIA GPU is recommended but **not required** (the scripts fall back to CPU)
- A webcam, for the live demo
- Packages: `torch`, `torchvision`, `thop`, `opencv-python`, `numpy`, `scipy`,
  `matplotlib`

The official PHASE-Net repository must be cloned separately (see Setup); it is
kept **outside** this repository to avoid a nested git repo.

## Setup

**1. Clone the official model repository.** On Windows, enable long paths first —
the repo contains a directory name that exceeds the 260-character limit and the
clone will otherwise fail halfway:

```bash
git config --global core.longpaths true
git clone https://github.com/Alex036225/PhaseNet.git
```

**2. Install the dependencies** into your virtual environment:

```bash
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
# source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install thop opencv-python numpy scipy matplotlib
```

**3. Locate the pretrained weights.** The repository ships genuine UBFC-trained
weights (13.2 MB) at
`logs/PhaseNet/PhysLLM_UBFC_UBFC_072_100/<very long folder name>/PreTrainedModels/PhysLLM_PhaseNet_UBFC_UBFC_072_100_Epoch9.pth`.
Copy them to a short path to avoid further Windows path problems:

```
PhaseNet/weights/phasenet_ubfc_epoch9.pth
```

**4. Set the repo path.** Both scripts have a `REPO = ...` line near the top —
point it at your local clone.

> **Note:** `dataset/data_loader/BaseLoader.py` imports `retinaface`, which pulls
> in TensorFlow. The scripts here import **only** the model class
> (`neural_methods.model.PhaseNet.PhaseNet`), so TensorFlow is never needed for
> inference.

## Run

Verify the model and weights load correctly:

```bash
python 01_load_phasenet.py
```

Check the pipeline without a camera:

```bash
python 02_webcam_phasenet.py --selftest
```

Measure your heart rate from the webcam:

```bash
python 02_webcam_phasenet.py
```

A window opens and waits until it detects your face, counts down from three,
then records ~13 seconds. The terminal prints the estimated BPM (overall and
per clip) and a waveform plot is written to `pulse_waveform.png`. Sit still in
even lighting. Press `q` to abort.

Useful options: `--seconds 20` (record longer), `--camera 1` (different webcam),
`--cpu` (force CPU).

## Preprocessing

The input pipeline was reproduced from the repository source so it matches
training exactly. Getting any of these wrong silently corrupts the result:

1. Frames read and converted **BGR → RGB**.
2. **Haar-cascade** face detection on the **first frame only**
   (`DO_DYNAMIC_DETECTION: False`) — the same box is reused for the whole clip.
3. Bounding box **enlarged 1.5×** about its centre (`LARGE_BOX_COEF: 1.5`).
4. Crop, then resize to **128×128** with `cv2.INTER_AREA`.
5. `DATA_TYPE: Raw` — pixel values stay in the **0–255** range.
   `BaseLoader.__getitem__` calls `np.float32(data)` with **no division by 255**;
   normalising to 0–1 here would feed the model the wrong input scale.
6. Laid out as `NCDHW` → `(1, 3, 128, 128, 128)`.
7. Chunked into 128-frame clips.

Heart rate uses the repository's own post-processing (`_detrend`,
`_calculate_fft_hr`): detrend (λ = 100) → 1st-order Butterworth band-pass
0.75–2.5 Hz (45–150 BPM) → FFT peak → ×60.

## Results: UBFC-rPPG validation

15 subjects of UBFC-rPPG DATASET_2 were evaluated against the CMS50E oximeter
ground truth, using the authors' pretrained weights (inference only, no training).

### The train/test split matters

The released checkpoint is named `UBFC_UBFC_072_100`: it was **trained on the first
72 %** of the subject list and tested on the last 28 %. Nine of the fifteen
subjects available locally fall inside that training split, so scoring them
measures memorisation, not generalisation. Results are therefore reported
separately — **only the held-out figure is meaningful**.

| Group | n | MAE (BPM) | RMSE (BPM) |
|-------|---|-----------|------------|
| All subjects evaluated | 15 | 1.88 | 3.67 |
| Seen during training | 9 | 2.15 | 3.73 |
| **Truly held out** | **6** | **1.46** | **3.59** |

### Held-out subjects, individually

| subject | reference HR | predicted HR | error | SNR (dB) | MACC |
|---------|-------------|--------------|-------|----------|------|
| subject5 | 101.1 | 101.1 | **0.0** | 16.5 | 0.92 |
| subject44 | 87.9 | 87.9 | **0.0** | −1.4 | 0.96 |
| subject46 | 91.4 | 91.4 | **0.0** | 1.6 | 0.91 |
| subject47 | 105.5 | 114.3 | 8.8 | 0.3 | 0.97 |
| subject48 | 91.4 | 91.4 | **0.0** | 10.6 | 0.73 |
| subject49 | 86.1 | 86.1 | **0.0** | 14.6 | 0.94 |

**Five of six held-out subjects were predicted exactly.** The entire held-out error
comes from a single subject.

### Aggregate metrics (all 15, for reference)

```
MAE          1.88 BPM        Pearson r    0.971
RMSE         3.67 BPM        within 3 BPM  73.3 %
MAPE         1.83 %          within 5 BPM  80.0 %
Bland-Altman bias  -0.59 BPM,  95 % limits of agreement  -7.94 .. +6.77 BPM
```

### The one failure is a post-processing bug, not a model failure

subject47 reported 114.3 BPM against a reference of 105.5. Inspecting the model's
own output spectrum:

| | BPM | relative power |
|---|---|---|
| strongest peak in PHASE-Net's output | 105.0 | 1.00 |
| power at the **true** HR (105.5) | — | 0.98 |
| power at the **reported** HR (114.3) | — | 0.71 |

PHASE-Net located the correct pulse frequency; the FFT peak-picking in the
post-processing chain selected a weaker neighbouring peak instead. Its MACC of
0.97 — the highest of all six held-out subjects — confirms the predicted waveform
matched the reference closely. This mirrors a defect already found and fixed in the
webcam pipeline (`03_webcam_phasenet_v2.py`), and is the clearest avenue for
improvement.

### Comparison with the paper

The paper reports **MAE 0.15 BPM** on its own 12-subject test split. Six of those
twelve subjects were available here; on those six the measured MAE is 1.46 BPM,
which reduces to **0.0 BPM if the single peak-selection failure is excluded**.
The remaining gap is therefore attributable to HR extraction and to the smaller,
partial test set rather than to the model itself. A like-for-like comparison
requires the remaining six test subjects (subject41, 42, 43, 45, 8, 9).

### Environment and cost

Windows 11, Python 3.12.7, PyTorch 2.6.0+cu124, NVIDIA RTX 3050 Laptop (4 GB VRAM).
Roughly 8 seconds of GPU time per one-minute video.

## Earlier verification steps

| Check | Outcome |
|-------|---------|
| Weight loading | **0 missing, 0 unexpected keys** — all 82 tensors matched |
| Forward pass | Succeeded **on GPU** in 1.60 s |
| Peak GPU memory | **1.66 GB** of 4.00 GB — comfortable headroom |
| Input / output shape | `(1, 3, 128, 128, 128)` → `(1, 128)` |
| BPM maths on a synthetic 75.0 BPM sine | **75.59 BPM** (correct within FFT bin resolution) |
| Full pipeline on synthetic frames | Runs end to end on CUDA, shapes consistent |

The perfect key match confirms both that the architecture was reconstructed
correctly and that the released checkpoint is genuine and intact.

## Reproducibility notes

Two discrepancies were found between the paper and the released code. Neither
invalidates the paper's accuracy results, but both are worth recording.

**1. The parameter count does not match the paper's claim.** The paper reports
0.29 M parameters. Measured directly from the released checkpoint:

```
total (whole checkpoint) : 3,300,274  (3.300 M)
training-only modules    : 2,487,984  (2.488 M)   <- Decoder1D + projection_encoder
inference path           :   812,290  (0.812 M)
paper claims             :   290,000  (0.290 M)
```

`Decoder1D` and `projection_encoder` implement the reconstruction
regularisation term and run **only during training**, so excluding them from an
inference-cost figure is fair. Even then the inference path holds **0.81 M
parameters — about 2.8× the published 0.29 M**. PHASE-Net is still far smaller
than RhythmFormer (4.21 M) or PhysFormer (7.38 M), so the efficiency argument
survives; the published number may correspond to a reduced configuration.

**2. The checkpoint uses 4 TCN layers, not the ablation-optimal 3.** The paper's
ablation reports 3 layers as optimal on UBFC-rPPG (MAE 0.15 vs 0.25 at 4) and
states "we set the default depth to 3", but the released checkpoint contains
`temporal_model.network.{0,1,2,3}` — four layers.

## Status

| Item | Status |
|------|--------|
| Paper analysed | ✅ Done |
| PyTorch + CUDA environment | ✅ Working, GPU confirmed |
| Pretrained weights obtained and validated | ✅ Done (perfect key match) |
| Model runs on local GPU | ✅ Done (1.66 GB peak) |
| Preprocessing replicated from source | ✅ Done |
| Webcam script + self-test | ✅ Written and self-tested |
| Live webcam measurement on a real face | ✅ Runs; limited by webcam capture quality (10 fps) |
| **Validation against UBFC-rPPG ground truth** | ✅ **15 subjects; MAE 1.46 BPM held-out** |
| JSON results export for the dashboard | ✅ Done |
| Remaining 6 test-split subjects | ⏳ Videos not yet downloaded |
| Fix FFT peak selection (the subject47 failure) | ⏳ Next |
| Comparison against the POS baseline | ⏳ Next |
| Training / fine-tuning | ❌ Not feasible on 4 GB VRAM (use Kaggle/Colab) |

## Limitations

- **The held-out sample is small.** Six subjects is enough to demonstrate the
  pipeline works, not enough for a conclusive accuracy claim. Half of the paper's
  12-subject test split is still missing.
- **One HR value per video.** Each video yields a single averaged heart rate, so
  short-term variation is invisible. Continuous or windowed HR (and eventually HRV)
  needs a sliding-window evaluation.
- **Training is out of reach on this hardware.** The paper used a single NVIDIA
  H100; this machine has 4 GB of VRAM. The work therefore targets inference with
  the authors' pretrained weights, which is sufficient for the KEIKO use case.
- The pretrained weights were trained on **UBFC-rPPG only**, so results on very
  different lighting, skin tones, or camera hardware will be weaker than the
  paper's cross-dataset numbers, which come from models trained on more sources.
- The webcam script measures the true frame rate and uses it for the BPM
  calculation, but the model itself was **trained at 30 fps**; a webcam running
  far from 30 fps will still degrade accuracy.
- Face detection runs on the **first frame only** (matching training), so the
  subject must stay roughly within the initial bounding box.

## Roadmap

- Run the webcam demo on a real face and cross-check against the Samsung watch
  reference, as was done for the POS baseline (~78 BPM vs the watch's 74 BPM).
- Obtain UBFC-rPPG and validate quantitatively against the CMS50E ground truth;
  attempt to reproduce the reported 0.15 bpm MAE.
- Compare PHASE-Net against POS under identical conditions, especially **under
  head motion**, where POS is known to fail — this is the core justification for
  moving to a deep model.
- Adapt to the KEIKO setting: longer recordings, subject seated at the
  collaborative task, and a sliding-window real-time mode.
- Extend to **heart rate variability (HRV)**, which needs accurate beat-to-beat
  peak timing rather than an averaged BPM.

## References

- B. Zhao, D. Guo, J. Cao, Y. Xu, B. Zou, T. Tan, Y. Sun, and Z. Yu, "PHASE-Net:
  Physics-Grounded Harmonic Attention System for Efficient Remote
  Photoplethysmography Measurement," *CVPR*, 2026 (Highlight).
  arXiv:2509.24850. Code: https://github.com/Alex036225/PhaseNet
- X. Liu *et al.*, "rPPG-Toolbox: Deep Remote PPG Toolbox," arXiv:2210.00716,
  2022. (framework this repository builds on)
- W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, "Algorithmic Principles
  of Remote PPG," *IEEE Transactions on Biomedical Engineering*, vol. 64, no. 7,
  pp. 1479–1491, 2017. (the POS baseline)
- S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, and J. Dubois, "Unsupervised
  skin tissue segmentation for remote photoplethysmography," *Pattern
  Recognition Letters*, vol. 124, pp. 82–90, 2017. (UBFC-rPPG dataset)
- B. Zou, Z. Guo, J. Chen, J. Zhuo, W. Huang, and H. Ma, "RhythmFormer:
  Extracting patterned rPPG signals based on periodic sparse attention,"
  *Pattern Recognition*, vol. 164, 111511, 2025. (considered alternative)

## Context

Part of a Master's project at TU Clausthal on non-invasive physiological
monitoring to support human wellbeing during human–robot collaboration.
