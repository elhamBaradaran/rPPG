# KEIKO rPPG — Real-Time Webcam Heart Rate (POS Baseline)

A camera-only, non-invasive system that estimates a person's heart rate in real
time from an ordinary webcam, using remote photoplethysmography (rPPG). This is
the **classical POS baseline** for a larger Master's project on physiological
monitoring during human–robot collaboration (the "KEIKO" scenario).

## What it does

The system reads the live webcam feed, locates the face, measures the tiny color
changes in the facial skin caused by the blood pulse, and turns those changes
into a live heart-rate estimate (in beats per minute) plus a pulse waveform.

The pipeline:

1. Capture the webcam feed (OpenCV).
2. Detect the face (Haar cascade) and extract one or more skin regions of
   interest (ROIs).
3. Average the R, G, B color of the ROI(s) on every frame.
4. Collect the recent color values in a 10-second sliding buffer.
5. Apply the **POS** algorithm (Plane-Orthogonal-to-Skin, Wang et al., 2017) to
   combine the color channels into a single pulse signal, suppressing lighting
   and motion.
6. Band-pass filter the signal to the human heart-rate range (0.7–4 Hz).
7. Use an FFT to find the dominant frequency and convert it to BPM.

## Method

POS is a **classical, training-free** rPPG method. It requires no neural network
and no pretrained weights, which makes it a lightweight first baseline. It works
by projecting the normalized RGB signal onto a plane chosen to be insensitive to
the skin-tone / lighting direction, isolating the pulse component.

Two versions of the ROI stage are provided:

- **Forehead only** (`heartrate.py`) — measures a single forehead patch.
- **Forehead + cheeks** (`heartrate_multi_roi.py`) — measures the forehead and
  both cheeks and averages them with equal weight. Sampling more skin from
  several regions typically improves the signal-to-noise ratio and gives a
  steadier reading when the subject is still. It does **not** remove motion
  sensitivity: large head movement still disturbs all regions at once.

## Files

The scripts are intentionally kept separate as a step-by-step build-up of the
pipeline:

| File | Purpose |
|------|---------|
| `webcam_test.py`         | Open the webcam and display the live feed. |
| `face_test.py`           | Detect the face and draw a bounding box. |
| `color_test.py`          | Extract the forehead ROI and print its average color. |
| `buffer_test.py`         | Add a 10-second sliding buffer of color values. |
| `pos_test.py`            | Apply POS and draw the raw pulse waveform. |
| `heartrate.py`           | **Full pipeline (forehead only):** POS + band-pass + FFT + live BPM. |
| `heartrate_multi_roi.py` | **Full pipeline (forehead + both cheeks):** usually steadier when still. |

`heartrate.py` and `heartrate_multi_roi.py` are the complete demos; the others
show each earlier stage in isolation.

## Requirements

- Python 3.10+ (developed on 3.12)
- A webcam
- Packages: `opencv-python`, `numpy`, `scipy`

## Setup

```bash
# create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
# source venv/bin/activate

# install dependencies
pip install opencv-python numpy scipy
```

## Run

Forehead-only demo:

```bash
python heartrate.py
```

Forehead + cheeks demo (usually a steadier reading when still):

```bash
python heartrate_multi_roi.py
```

Two windows open: the camera feed with the ROI box(es) and live BPM, and the
filtered pulse waveform. Sit still, facing the camera in even lighting, and wait
~10 seconds for the buffer to fill before the first reading appears. Press `q`
(with a window focused) to quit.
## Demo




https://github.com/user-attachments/assets/93d7ef82-0775-49cd-9530-12dc0122e42c

https://github.com/user-attachments/assets/e3506e95-3f59-489f-a5db-f855b2583cf4




Both pipelines were recorded live and cross-checked against a Samsung watch used
as a reference heart-rate monitor. In both recordings the rPPG output settled at
**~78 BPM** while the watch reported **74 BPM** — a difference of about 4 BPM,
which is within the expected error range for a classical POS method under good
conditions (subject still, even lighting).

This is a single informal comparison rather than a controlled evaluation, so it
confirms that the pipeline produces plausible, close-to-reference readings, but
it is not sufficient to claim that one ROI variant is more *accurate* than the
other. The multi-ROI version's main benefit is a **steadier** reading, not
necessarily a closer one in a single measurement. Quantitative evaluation on a
labeled dataset (see Roadmap) is the proper next step.


## Limitations

This is an early baseline and its results should be read as such:

- The estimate is **sensitive to head motion and lighting**; it is most reliable
  when the subject is still and evenly lit. Under motion the reported BPM can be
  badly wrong (it may lock onto the motion frequency instead of the pulse).
- Using forehead + cheeks improves stability when still, but does **not** fix the
  motion problem.
- The reading jitters by several BPM between updates.
- The pipeline currently **assumes a 30 fps camera** rather than measuring the
  true frame rate; the absolute BPM depends on this being correct.
- Validation so far is **informal only** (visual comparison against a manual
  finger-pulse count); it has not been evaluated quantitatively against dataset
  ground truth.

These limitations are expected for classical POS and motivate the move to
deep-learning models.

## Roadmap

- Quantitative evaluation of POS on the UBFC-rPPG dataset (MAE / RMSE against
  ground truth), e.g. via the rPPG-Toolbox.
- Add a deep-learning baseline (TS-CAN).
- Move to the main model (RhythmFormer or PHASE-Net).
- Robustness testing on MMPD.

## References

- W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, "Algorithmic Principles
  of Remote PPG," *IEEE Transactions on Biomedical Engineering*, vol. 64, no. 7,
  pp. 1479–1491, 2017. doi:10.1109/TBME.2016.2609282 (POS method)

## Context

Part of a Master's project at TU Clausthal on non-invasive physiological
monitoring to support human wellbeing during human–robot collaboration.
