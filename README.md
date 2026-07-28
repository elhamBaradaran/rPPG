# Camera-Only Heart Rate Monitoring for Human–Robot Collaboration

Estimating a person's heart rate from an **ordinary camera** — no sensors, no contact —
so that a collaborative robot can sense how its human partner is doing.

Master's research at **TU Clausthal**, carried out within the
**[KEIKO](https://www.simzentrum.de/forschungsprojekte/keiko/)** project.

---

## Context: the KEIKO project

**KEIKO** — *Kognitiv und Empathisch Intelligente Kollaborierende Roboter*
(Cognitively and Empathically Intelligent Collaborating Robots) — develops robots that
can read a human partner's mental and emotional state and adapt to it, rather than
blindly repeating a programmed motion.

- **Collaboration:** TU Clausthal with the **University of Göttingen**, coordinated by
  the Simulation Science Center (SWZ) Göttingen–Clausthal
- **Funding:** Lower Saxony Ministry for Science and Culture, SPRUNG programme
- **Supervision:** [Prof. Dr. Jörg Philipp Müller](https://studip.tu-clausthal.de/dispatch.php/profile?username=jmue)
  (Department of Informatics, TU Clausthal), a principal investigator in KEIKO —
  with **Sakif Hossain** supervising this work directly

One of KEIKO's six research strands is the **contactless acquisition of physiological
parameters using optical methods**. That is what this repository addresses: recovering
the pulse from video alone, as a signal a cobot could eventually use to judge a partner's
state during a shared assembly task.

The technique is **rPPG** (remote photoplethysmography). Each heartbeat pushes blood into
the face and changes how much light the skin absorbs. The change is far too small to see,
but a camera can measure it.

---

## Current status

PHASE-Net is implemented and validated against the UBFC-rPPG dataset, whose ground truth
comes from a CMS50E pulse oximeter.

| Metric | Value |
|--------|-------|
| **MAE, held-out subjects** (10 s windowed protocol) | **0.39 BPM** |
| MAE against the oximeter's own readout | 2.96 BPM |
| Evaluation windows | 780 |
| Inference cost | ~8 s of GPU time per one-minute video, 1.66 GB VRAM |

Only subjects genuinely outside the released checkpoint's training split are counted —
see [the validation write-up](Models/PHASE-Net/README.md#results) for why that
distinction matters and how the numbers change without it.

---

## Demo

The classical baseline running live on a webcam. The left window tracks the face and
reports beats per minute; the right window shows the recovered pulse waveform.

**Forehead region only:**

https://github.com/user-attachments/assets/93d7ef82-0775-49cd-9530-12dc0122e42c

**Forehead and both cheeks** — sampling more skin gives a steadier reading when the
subject is still:

https://github.com/user-attachments/assets/e3506e95-3f59-489f-a5db-f855b2583cf4

Both recordings were cross-checked against a Samsung smartwatch worn at the same time.
The rPPG output settled at **~78 BPM** against the watch's **74 BPM** — about 4 BPM
apart, which is within the expected range for classical POS under good conditions
(subject still, even lighting). This is an informal check rather than a controlled
evaluation; the quantitative results above come from the UBFC-rPPG dataset instead.

---

## Repository structure

| Folder | Contents |
|--------|----------|
| **[`Models/POS`](Models/POS)** | Classical **POS** baseline (Wang et al., 2017), built from scratch with OpenCV/NumPy/SciPy. Real-time webcam demo, no neural network, no training. |
| **[`Models/PHASE-Net`](Models/PHASE-Net)** | The **main model** — PHASE-Net (CVPR 2026 Highlight). Live demo, UBFC validation, HR-extraction benchmarks. |
| **[`Literature-Review`](Literature-Review)** | Paper summaries, each with an AI-generated video overview. |
| **`results/`** | Machine-readable evaluation output (JSON) with full traceability, ready for a results dashboard. |

---

## How the project developed

**1. A classical baseline, built by hand.** POS projects the RGB signal onto a plane
chosen to be insensitive to skin tone and lighting. Implementing it from scratch made the
physics concrete — and exposed its limits: validated against a smartwatch it reads
~78 BPM vs 74, but under head motion it locks onto the motion frequency and collapses to
~40 BPM. That failure is the reason to move to a learned model.

**2. A physics-grounded deep model.** PHASE-Net derives its architecture from the
Navier–Stokes equations of blood flow: the pulse obeys a damped-oscillator equation whose
discrete solution is a causal convolution — that is, a Temporal Convolutional Network. The
architecture is therefore mandated by the physics rather than found by trial and error.
It is also ~14× smaller than RhythmFormer and generalises considerably better across
datasets, which matters because a robot laboratory looks nothing like the training data.

---

## Findings from reproducing the paper

Reproducing published work carefully surfaced three issues worth recording:

**1. The released checkpoint's training split overlaps the evaluation set.** It is named
`UBFC_UBFC_072_100` — trained on the first 72 % of subjects. Nine of the fifteen subjects
available locally fall inside that split, so scoring them measures memorisation. Held-out
results are reported separately throughout.

**2. One heart rate per video is an ill-posed protocol.** Heart rate varies within a
minute — on one subject the oximeter itself reports 103–120 BPM — so a whole-video
spectrum is genuinely multi-modal and taking its argmax compares whichever mode happened
to dominate each signal. What looked like an 8.8 BPM model failure was this artefact.
A 10-second windowed protocol reduced the held-out MAE from 1.46 to **0.39 BPM**, and the
result became stable across estimator choices where before it swung three-fold.

**3. The parameter count does not match the paper.** The paper reports 0.29 M; the
released checkpoint totals 3.30 M, of which 2.49 M is used only during training. The
inference path holds **0.81 M — roughly 2.8×** the published figure. PHASE-Net remains
far smaller than its competitors, so the efficiency argument survives.

A fourth observation concerns the data itself: cross-checking three independent readings
of each recording — the device readout, an FFT of the reference waveform, and a beat
count — only **6 of 15 subjects** agree. On two subjects the oximeter's own HR readout is
wrong by more than 20 BPM, which both waveform analyses independently confirm.

---

## Quick start

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows;  source venv/bin/activate on macOS/Linux

# classical baseline - only needs a webcam
pip install opencv-python numpy scipy
python Models/POS/heartrate.py
```

The deep model needs PyTorch and the official PHASE-Net weights — see
[`Models/PHASE-Net/README.md`](Models/PHASE-Net/README.md#setup) for the full setup,
including the Windows long-path workaround required to clone the upstream repository.

---

## Roadmap

- [x] Classical POS baseline with a live webcam demo
- [x] PHASE-Net running locally, preprocessing reproduced from source
- [x] Validation against UBFC-rPPG with clinical agreement statistics (Bland–Altman)
- [x] Windowed evaluation protocol
- [ ] Direct comparison of PHASE-Net against POS on identical recordings, under motion
- [ ] Results dashboard reading `results/*.json`
- [ ] Continuous monitoring adapted to the KEIKO collaborative task
- [ ] Heart rate variability, which needs beat-to-beat timing rather than an averaged rate

---

## References

- B. Zhao *et al.*, "PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient
  Remote Photoplethysmography Measurement," *CVPR*, 2026 (Highlight).
  [arXiv:2509.24850](https://arxiv.org/abs/2509.24850)
- W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan, "Algorithmic Principles of Remote
  PPG," *IEEE Transactions on Biomedical Engineering*, 64(7), 2017. — the POS baseline
- S. Bobbia *et al.*, "Unsupervised skin tissue segmentation for remote
  photoplethysmography," *Pattern Recognition Letters*, 124, 2017. — UBFC-rPPG dataset
- X. Liu *et al.*, "rPPG-Toolbox: Deep Remote PPG Toolbox,"
  [arXiv:2210.00716](https://arxiv.org/abs/2210.00716), 2022.
