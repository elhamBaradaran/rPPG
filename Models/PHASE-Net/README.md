# KEIKO rPPG — PHASE-Net (Main Model)

A physics-grounded deep-learning rPPG model that estimates heart rate from
ordinary camera video. This is the **main model** of the Master's project on
physiological monitoring during human–robot collaboration (the "KEIKO"
scenario), replacing the classical [POS baseline](../POS) built earlier.

PHASE-Net was published as a **CVPR 2026 Highlight** paper. Unlike most deep
rPPG networks, its architecture is not chosen by trial and error — it is
*derived* from the fluid dynamics of blood flow.

> **Status:** validated against UBFC-rPPG ground truth.
> On 6 truly held-out subjects, using a 10-second windowed protocol:
> **MAE 0.39 BPM** against the reference waveform, **2.96 BPM** against the
> oximeter's own readout. See [Results](#results).

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

Numbered scripts are the main path, in order; the rest are utilities.

| File | Purpose |
|------|---------|
| `01_load_phasenet.py` | Build the model, load the pretrained weights, verify every key matches, report parameter counts, and time a GPU forward pass. |
| `02_webcam_phasenet.py` | First webcam demo: webcam → face crop → PHASE-Net → BPM. Kept for reference; **prefer v2**. |
| `03_webcam_phasenet_v2.py` | **Live demo.** Overlapping sliding windows with Hann overlap-add, zero-padded FFT, and an honest confidence verdict from three independent signals. `--selftest` needs no webcam. |
| `04_ubfc_eval.py` | **The validation.** Scores the model against UBFC-rPPG ground truth and writes both CSV and dashboard JSON. |
| `results_export.py` | Model-agnostic JSON writer: metrics, Bland-Altman, waveforms, and full run traceability. |
| `import_ubfc.py` | Unpack UBFC subject folders downloaded from Google Drive into the expected layout. |
| `fetch_ground_truth.py` | Retrieve `ground_truth.txt` for any subject whose video arrived without it. |
| `11_live_motion_test.py` | Live still/motion/still recording with both methods shown side by side. |
| `12_motion_analysis.py` | Offline analysis of one motion recording. |
| `13_record_full.py` | Records a condition, saving a region large enough for the face to move within. |
| `14_static_vs_dynamic.py` | Processes one recording twice - fixed crop versus face tracking. |
| `15_motion_protocol.py` | **The controlled protocol:** error against measured motion, across conditions. |
| `cam_check.py` | Diagnose why a webcam captures below 30 fps (exposure, backend, or display loop). |
| `PHASE-Net_Report.docx` | An earlier snapshot of this content as a formatted Word report. |

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
`--cpu` (force CPU). If the reported frame rate is far below 30, run `cam_check.py`
— the model is trained at 30 fps and a slow camera degrades accuracy.

### Validate against UBFC-rPPG

Obtain UBFC-rPPG DATASET_2 (42 subjects, `subjectN/vid.avi` + `subjectN/ground_truth.txt`).
Downloading the folders from Google Drive produces zip archives; unpack them with:

```bash
python import_ubfc.py          # unpack and arrange
python import_ubfc.py --status # show which subjects are complete
python fetch_ground_truth.py   # fill in any missing ground_truth.txt
```

Then score the model:

```bash
python 04_ubfc_eval.py --data "path/to/UBFC"
```

Add `--limit 3` to try a few subjects first, or `--delete-after` to remove each
video once it has been scored (results are kept). Output goes to `ubfc_results.csv`
and to `results/` as JSON.

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

### One heart rate per video is the wrong protocol

The first evaluation gave one HR per 60-second video, and produced an apparent
8.8 BPM failure on subject47. That failure turned out to be an artefact of the
protocol, not a fault of the model.

Heart rate is not constant over a minute. On subject47 the oximeter's own readout
ranges from **103 to 120 BPM** within the single recording, and a sliding-window
estimate of the model's output drifts 95 → 119 BPM. The whole-video spectrum is
therefore genuinely **bimodal**: 105.5 and 114.3 are both real. Argmax compares
whichever mode happened to dominate the prediction against whichever dominated the
reference and calls the difference an error. Two further symptoms confirmed the
diagnosis:

- The baseline quantises HR to `30/2048 × 60 = 0.88 BPM` bins, so the original
  "0.0 BPM error" on five subjects meant *same bin*, not exact agreement.
- Under the video-level protocol the result swung between **1.46 and 4.49 BPM**
  depending only on which peak-picking method was used — a sign the protocol, not
  the method, was unstable.

`08_windowed_eval.py` therefore estimates HR in overlapping 10-second windows
(1-second step), where the signal is close to stationary, and aggregates. This is
also what the rPPG-Toolbox supports via `INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW`,
which the shipped PhaseNet config leaves **off**.

### Results

Reported two ways. *vs reference* applies identical processing to the predicted and
ground-truth waveforms — the convention rPPG papers use. *vs device* compares against
the CMS50E's own HR readout (line 2 of `ground_truth.txt`), which involves none of our
signal processing and is therefore the stricter, more honest figure.

| Protocol | Group | n | vs reference | vs device |
|----------|-------|---|--------------|-----------|
| Video-level (original) | held out | 6 | 1.46 | 3.23 |
| Video-level (original) | all | 15 | 1.88 | 7.40 |
| **Windowed 10 s / 1 s** | **held out** | **6** | **0.39** | **2.96** |
| **Windowed 10 s / 1 s** | all | 15 | 0.53 | 6.08 |

Windowing raises the sample size from 15 videos to **780 windows**, and the
subject47 "failure" disappears: the model reports 111.2 BPM where the device reports
112.0 — an error of 0.8 BPM.

The choice of estimator barely matters once windowed (held-out MAE 0.39–0.61 across
periodogram, Welch, 6 s and 10 s windows), whereas under the video-level protocol the
same choices swung the result three-fold. Stability across reasonable choices is
itself evidence that the windowed protocol is the correct one.

### The reference itself is not always trustworthy

Cross-checking three independent readings of each recording — the device's HR
readout, an FFT of the ground-truth BVP, and counting beats in that BVP — only
**6 of 15 subjects** have a ground truth where all three agree.

| subject | device | FFT of BVP | beat count | verdict |
|---------|--------|-----------|------------|---------|
| subject5 | 99 | 101 | 100 | consistent |
| subject49 | 87 | 86 | 87 | consistent |
| subject25 | **92** | 114 | 115 | device readout is wrong |
| subject27 | **89** | 112 | 110 | device readout is wrong |

On subject25 and subject27 two independent analyses of the BVP waveform agree with
each other and disagree with the device by more than 20 BPM. This is why the field
uses the BVP waveform rather than the device readout as ground truth — and why the
"vs device" column above is pessimistic: it includes subjects whose device readout
is itself faulty.

### Comparison with the paper

The paper reports **MAE 0.15 BPM** on its own 12-subject test split, using the
convention of processing both signals identically. The comparable figure here is
**0.39 BPM on 6 of those 12 subjects** under a 10 s windowed protocol. The remaining
gap is plausibly explained by the partial test set and by unstated differences in the
extraction chain — inconsistent reporting of these parameters is a documented
reproducibility problem in rPPG. A like-for-like comparison needs the remaining six
subjects (subject41, 42, 43, 45, 8, 9).

### Environment and cost

Windows 11, Python 3.12.7, PyTorch 2.6.0+cu124, NVIDIA RTX 3050 Laptop (4 GB VRAM).
Roughly 8 seconds of GPU time per one-minute video. POS, for comparison, needs about
4.5 seconds per video and no GPU at all.

## Against the POS baseline

`09_pos_baseline.py` computes POS from **exactly the face crops PHASE-Net receives** —
same Haar box, same 1.5× enlargement, same 128×128 resize — and `10_compare_models.py`
scores everything with the same 10-second windowed protocol. The only variable is the
algorithm. Two POS variants are included: the repository's own `POS_WANG` (Wang et al.
2017 with a 1.6 s sliding window, the implementation the paper compares against) and the
simplified one written from scratch in [`Models/POS`](../POS).

**Held-out subjects (n = 6):**

| Method | vs reference | worst | vs device | worst |
|--------|-------------|-------|-----------|-------|
| **PHASE-Net** | **0.39** | **0.6** | 2.96 | 5.4 |
| POS — reference | 1.01 | 3.3 | **2.87** | 5.0 |
| POS — ours | 2.01 | 8.4 | 3.74 | 10.1 |

**All 15 subjects — where the difference in robustness shows:**

| Method | vs reference | worst | vs device | worst |
|--------|-------------|-------|-----------|-------|
| **PHASE-Net** | **0.53** | **2.2** | 6.08 | 23.9 |
| POS — reference | 2.87 | 22.4 | 6.35 | 22.8 |
| POS — ours | 5.02 | 27.2 | 6.60 | 20.1 |

Three things follow.

**1. The convention flatters the deep model.** Against the reference waveform PHASE-Net is
2.6× better; against the oximeter's own readout the two are **tied** (2.96 vs 2.87). This
is not a contradiction. PHASE-Net's training objective is a negative-Pearson loss on the
ground-truth BVP waveform, so scoring it against that waveform tests it on the very
target it was optimised for. The device readout is outside that loop.

**2. The real advantage is consistency, not average accuracy.** Across all 15 subjects
PHASE-Net's worst case is **2.2 BPM** against POS's **22.4**. POS fails outright on some
subjects (subject27: 22.4 BPM error) where PHASE-Net stays within 1.5. For patient
monitoring, bounded error matters more than a better mean.

**3. Our own POS costs about 30 % accuracy versus the reference** (3.74 vs 2.87 BPM
against the device). The projection maths is identical; what is missing is the 1.6-second
sliding window of the original formulation.

**Caveat:** UBFC subjects are largely still, so this comparison does not test the motion
robustness that motivates a deep model in the first place.

## Under head motion — an unexpected result

`11_live_motion_test.py` records 90 seconds in three phases — still, head motion, still —
and displays both methods live. Heart rate barely changes over 90 seconds, so the still
phases serve as each method's own reference for the middle one, and no oximeter is needed.
Capture runs in a dedicated thread and was measured at a true **30.0 fps**.

| Method | still HR | during motion | drift | range during motion |
|--------|---------|--------------|-------|--------------------|
| PHASE-Net | 77.9 | 94.8 | **13.4** | 75 – 113 |
| POS | 78.1 | 77.2 | **1.2** | 75 – 80 |

*(smartwatch reference 83 BPM; both baselines sit about 5 BPM below it)*

This looked like the opposite of the expected result. A second recording then gave the
reverse ranking (PHASE-Net 10.5, POS 13.7), which showed the real problem: **the motion
intensity was never measured, only described.** A single recording cannot answer "is it
robust?" when the thing being varied is uncontrolled.

## A controlled motion protocol

`13_record_full.py` and `15_motion_protocol.py` replace that with a dose–response
experiment. Five conditions, each recorded as its own still → condition → still sandwich
so every condition carries its own heart-rate baseline. Crucially the motion is
**measured, not labelled**: the *motion dose* is the extra mean frame-to-frame pixel
change during the condition, which registers rotation, illumination shifts, blur and
speech — everything that face displacement alone misses.

| Condition | motion dose | max displacement | PHASE-Net drift | POS drift |
|-----------|------------|-----------------|----------------|-----------|
| `still` (control) | −0.05 | 8 px | **8.0** | **6.8** |
| `talk` | 0.29 | 9 px | 7.8 | 5.9 |
| `full` | 0.55 | 28 px | 10.5 | 13.7 |
| `slow` head turns | 1.04 | 48 px | 15.5 | 12.9 |
| `fast` head turns | 2.88 | 37 px | 15.4 | 14.1 |

Correlation between measured motion and drift: **+0.81** for PHASE-Net, **+0.68** for POS.
Motion genuinely drives the error — but not by the mechanism first assumed.

### Three hypotheses, all rejected

**1. "The face leaves the static crop."** Tested directly by `14_static_vs_dynamic.py`,
which processes one recording twice — fixed box versus re-detecting the face every second
— so the crop strategy is the only variable. The face never moved more than **21 % of a
face width**, never left the box, and the correlation between displacement and error was
**+0.07**, i.e. none. Dynamic tracking made things *worse* (drift 24.6 versus 10.5) because
the re-detected box jitters frame to frame and injects artificial motion; it was unstable
even during the still phases. **A slightly misaligned but stable crop beats a
well-centred jittery one.**

**2. "Facial appearance change breaks it."** The `talk` condition moves the face 9 px
while changing its appearance continuously. Its drift (7.8) is indistinguishable from the
still control (8.0). Speech costs nothing.

**3. "One method is inherently more motion-robust."** Both degrade by roughly the same
factor — PHASE-Net 8.0 → 15.5, POS 6.8 → 14.1 — and the curve saturates: tripling the
motion from `slow` to `fast` changes nothing (15.5 → 15.4).

### What the protocol did establish

**The noise floor is 7–8 BPM.** Sitting perfectly still, both methods wander by that much;
in the `still` recording PHASE-Net ranged over 68–108 BPM with no movement at all. Motion
roughly doubles this, to 13–15 BPM, and then plateaus.

**PHASE-Net's absolute reading is far more consistent than POS's.** Across the five
recordings, baseline offsets from the smartwatch reference (83 BPM):

| | offsets | spread |
|---|---------|--------|
| PHASE-Net | −3.0, −0.6, −4.3, −3.4, −1.9 | **3.7 BPM** |
| POS | −3.4, −16.8, −4.6, −5.8, −12.4 | 13.4 BPM |

PHASE-Net stays within about 4 BPM of the reference every time; POS is occasionally wrong
by 17. This is the same conclusion the UBFC comparison reached — the deep model's value is
bounded error, not a better average — now reproduced on independently recorded data.

### The finding that reframes the project

```
UBFC-rPPG, clean data           PHASE-Net  0.39 BPM
this webcam, sitting perfectly still       8.0  BPM
```

**A twenty-fold gap, before any movement is involved.** Several experiments went into
asking why performance collapses under motion, when the honest answer is that it is
already twenty times worse than the model's demonstrated capability while sitting still.
Motion merely doubles an already poor number.

The bottleneck is therefore **capture quality — lighting, camera, distance, face size in
frame — not motion robustness and not the model.** UBFC videos are recorded under
controlled illumination with the face filling much of the frame; this webcam setup is not.
Improving the recording conditions, and measuring the effect on the noise floor, should
come before any further work on motion.

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
| Live webcam capture at the trained frame rate | ✅ Fixed — 30.0 fps via a dedicated capture thread |
| **Validation against UBFC-rPPG ground truth** | ✅ **15 subjects; MAE 0.39 BPM held-out** |
| Windowed evaluation protocol | ✅ Done — the subject47 failure was a protocol artefact |
| JSON results export for the dashboard | ✅ Done |
| **Comparison against the POS baseline** | ✅ **Done — same crops, same protocol** |
| **Motion test (still / motion / still)** | ✅ Recorded and analysed |
| **Static vs dynamic face box, controlled test** | ✅ Done — the static box was **not** the cause |
| **Controlled motion protocol (5 conditions)** | ✅ Done — noise floor 7-8 BPM even at rest |
| Improve capture quality and re-measure the noise floor | ⏳ Next, and now the priority |
| Remaining 6 test-split subjects | ⏳ Videos not yet downloaded |
| Training / fine-tuning | ❌ Not feasible on 4 GB VRAM (use Kaggle/Colab) |

## Limitations

- **The held-out sample is small.** Six subjects is enough to demonstrate the
  pipeline works, not enough for a conclusive accuracy claim. Half of the paper's
  12-subject test split is still missing.
- **Face detection runs on the first frame only**, matching the training
  configuration. Head motion therefore pushes the face out of the crop, and the
  motion test suggests this is the dominant failure mode — see
  [the motion result](#under-head-motion--an-unexpected-result). Until dynamic
  tracking is added, the subject must stay within the initial bounding box.
- **The motion result rests on a single recording** of a single subject, with
  gentle movement. It is a lead, not a conclusion.
- **Only average heart rate, not HRV.** Heart rate variability needs beat-to-beat
  timing accurate to a few milliseconds; at 30 fps one frame is already 33 ms, so
  it requires sub-frame peak interpolation and a much cleaner signal than the
  motion test currently delivers.
- **Training is out of reach on this hardware.** The paper used a single NVIDIA
  H100; this machine has 4 GB of VRAM. The work therefore targets inference with
  the authors' pretrained weights, which is sufficient for the KEIKO use case.
- The pretrained weights were trained on **UBFC-rPPG only**, so results on very
  different lighting, skin tones, or camera hardware will be weaker than the
  paper's cross-dataset numbers, which come from models trained on more sources.
- The model was **trained at 30 fps**, so a webcam running far from that will
  degrade accuracy. `cam_check.py` diagnoses this; note that a single-threaded
  capture loop was measured at 10 fps on hardware that comfortably delivers 30, so
  check the software before blaming the camera.

## Roadmap

- **Improve capture quality and re-measure the noise floor.** This is now the
  priority: 8 BPM of drift while sitting perfectly still dwarfs anything motion
  adds. Controlled lighting, a shorter subject-to-camera distance so the face
  fills more of the frame, and a re-run of the `still` control to quantify the
  gain.
- **Explain the between-recording variability.** The `full` recording held a
  still-phase standard deviation of 1.81 BPM while `still` and `fast` reached
  13.60 under the same instruction. Whatever differs between those sessions is
  currently as large as the effect being measured.
- **Complete the paper's test split** (subject41, 42, 43, 45, 8, 9) for a
  like-for-like comparison against the reported 0.15 BPM.
- **A results dashboard** reading `results/*.json`: Bland-Altman, error vs SNR,
  and a waveform viewer, so both models can be compared visually.
- **Adapt to KEIKO**: longer recordings, subject seated at the collaborative
  task, and a continuous sliding-window mode rather than one HR per video.
- **Heart rate variability (HRV)**, which needs accurate beat-to-beat peak timing
  rather than an averaged BPM.

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

