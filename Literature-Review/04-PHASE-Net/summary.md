



# Paper Summary: PHASE-Net — Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement

## 1. Metadata
- **Title:** PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement
- **Venue:** CVPR 2026 (**Highlight**)
- **Authors:** Bo Zhao, Dan Guo, Junzhe Cao, Yong Xu, Bochao Zou, Tao Tan, Yue Sun, Zitong Yu
- **Source Link:** [arXiv:2509.24850](https://arxiv.org/abs/2509.24850) | [Official Code](https://github.com/Alex036225/PhaseNet)
- **Status:** [x] Read | [x] Implemented — see [`Models/PHASE-Net`](../../Models/PHASE-Net)

## 2. Visual Summary (AI-Generated)
> [!TIP]
> This video provides an AI-generated summary of the paper using NotebookLM to facilitate faster understanding of the core concepts.

<!-- Upload the NotebookLM video to GitHub (drag it into any issue comment) and paste
     the resulting https://github.com/user-attachments/assets/... link on the line below. -->
https://github.com/user-attachments/assets/089195ce-c8bf-418a-b584-ab0c5fd0b39f

## 3. Problem Statement
rPPG accuracy degrades badly under head motion and changing illumination. The authors
argue that the deeper issue is **how deep rPPG models are designed**: architectures are
chosen by empirical trial and error, so they are effectively black boxes. That leads to
two failures — models overfit to dataset-specific noise and generalise poorly, and their
decisions cannot be justified theoretically.

The paper's central question: *can a model's architecture be a direct embodiment of the
signal's physical laws, rather than a product of data fitting?*

## 4. Methodology Overview

**The physics chain.** The architecture is derived, not guessed:

| Step | Physics | Result |
|------|---------|--------|
| 1 | Beer–Lambert law + vessel compliance | Pixel intensity change ∝ blood-volume change ∝ pressure pulsation, so the pulse is physically embedded in the video |
| 2 | Navier–Stokes equations, linearised and 1-D averaged | A damped wave equation for pressure-pulse propagation |
| 3 | Observation at one fixed facial location | Reduces to a second-order ODE — a **forced damped harmonic oscillator** |
| 4 | Discretisation (semi-implicit Euler) → state-space model | **Proposition 1:** the solution *is* a causal convolution of past inputs |
| 5 | IIR → FIR approximation | **Proposition 2:** which is exactly what a **Temporal Convolutional Network** computes |

So the TCN is presented not as a heuristic choice but as the mathematically mandated one.

**Architecture:** `video → Vision Encoder (3 EST blocks) → Adaptive Spatial Filter → Gated TCN → pulse waveform`

- **ZAS (Zero-FLOPs Axial Swapper)** — parameter-free, reversible 2×2 block transpose on
  a subset of channels, mixing distant facial regions at zero computational cost.
- **ASF (Adaptive Spatial Filter)** — learns a per-frame spatial mask that highlights
  pulse-rich skin, then concatenates the first-order temporal derivative.
- **GTCN** — causal dilated TCN with `tanh` × `sigmoid` gating.
- **Loss:** Negative Pearson correlation on the waveform shape.

## 5. Key Findings

**Efficiency** — the smallest model in its comparison (128×128, T = 128):

| Method | Params (M) | MACs (G) |
|--------|-----------|----------|
| **PHASE-Net** | **0.29** | **28.3** |
| RhythmFormer | 4.21 | 28.8 |
| PhysFormer | 7.38 | 40.5 |
| DeepPhys | 7.50 | 96.0 |

**Accuracy** (MAE, bpm — lower is better):

| Method | UBFC-rPPG | PURE | BUAA | MMPD |
|--------|-----------|------|------|------|
| POS | 4.08 | 3.67 | — | 12.36 |
| RhythmFormer | 0.50 | 0.27 | 9.19 | 4.69 |
| **PHASE-Net** | **0.15** | **0.14** | **5.89** | **4.78** |

**Cross-dataset generalisation** (leave-one-out) is the strongest result — on PURE it
transfers roughly an order of magnitude better than RhythmFormer (2.86 vs 21.11 MAE),
supporting the claim that it learns pulse physics rather than dataset appearance.

## 6. Relevance to My Project

This is the **main model** for the KEIKO scenario, selected over RhythmFormer because:

- **Robustness under motion and illumination change** is exactly where my
  [POS baseline](../../Models/POS) fails — it locks onto the motion frequency and drops
  to ~40 BPM during head movement.
- **Cross-dataset generalisation matters most here**, because a robot laboratory will
  not resemble the training data.
- **It is lightweight** (0.29 M claimed parameters), which matters for eventual
  real-time use alongside the cobot.
- Notably, **PURE** — one of its benchmark datasets — is itself titled *"Non-contact
  video-based pulse rate measurement on a mobile service robot"*, i.e. a human–robot
  interaction context directly analogous to KEIKO.

## 7. Takeaways for My Repository

- [x] Reproduce the model and load the released weights (perfect key match, 82/82 tensors)
- [x] Reproduce the preprocessing from source rather than guessing it
- [x] Validate against UBFC-rPPG ground truth with a windowed protocol
- [x] Report held-out subjects separately from those inside the checkpoint's training split
- [x] Compare directly against the POS baseline on the same videos, same crops, same protocol
- [ ] Obtain the remaining test-split subjects for a like-for-like comparison with 0.15 BPM
- [ ] Adapt to continuous monitoring for the KEIKO collaborative task

**Discrepancies found while reproducing** (details in
[`Models/PHASE-Net/README.md`](../../Models/PHASE-Net/README.md)):

1. The paper reports **0.29 M parameters**, but the released checkpoint totals 3.30 M, of
   which 2.49 M is training-only. The inference path holds **0.81 M — about 2.8×** the
   published figure.
2. The ablation concludes 3 TCN layers are optimal and states that depth is used, but the
   **released checkpoint has 4**.
3. The shipped config evaluates **one heart rate per video**. Since heart rate varies
   within a minute, that protocol is ill-posed; switching to a 10-second windowed protocol
   changed the held-out MAE from 1.46 to **0.39 BPM**.
4. Measured against the paper's own convention PHASE-Net beats POS by 2.6×, but against
   the oximeter's independent readout the two are **tied** (2.96 vs 2.87 BPM). Its real
   advantage is consistency — a worst case of 2.2 BPM against 22.4 for POS.
5. Under head motion the model **destabilised badly** (drift 13.4 BPM, spikes to 141)
   while POS stayed within 1.2 BPM. The likely cause is the static face box the training
   config specifies, not the architecture — meaning the paper's motion-robustness claim
   may be **conditional on dynamic face tracking**. Under investigation.
