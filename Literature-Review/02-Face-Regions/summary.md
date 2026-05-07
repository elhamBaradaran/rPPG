# Synthesized Study: ROI Selection & Hybrid Algorithms in rPPG

This document synthesizes findings from two pivotal papers: one focusing on the optimization of facial regions (ROI) for signal extraction, and the other on hybrid algorithm development for contactless pulse monitoring in real-world environments.

---

## 1. Metadata & Source Links

### Paper A: Facial ROI Optimization
- **Title:** The role of face regions in remote photoplethysmography for contactless heart rate monitoring
- **Journal:** npj Digital Medicine (Nature Portfolio, 2025)
- **Official Publisher Link:** [Nature Link](https://www.nature.com/articles/s41746-025-01814-9)
- **PubMed Index:** [PubMed Entry](https://pubmed.ncbi.nlm.nih.gov/40715566/)
- **Status:** [x] Read | [ ] In Progress

### Paper B: Algorithm Development & Denoising
- **Title:** Remote photoplethysmography for contactless pulse rate monitoring: algorithm development and accuracy assessment
- **Journal:** Physiological Measurement (IOP Science, 2025)
- **Official Publisher Link:** [IOP Science Link](https://iopscience.iop.org/article/10.1088/1361-6579/ae1804)
- **Status:** [x] Read | [ ] In Progress

---

## 2. Visual Summary (AI-Generated)
> [!TIP]
> This section contains the AI-generated video summaries (using NotebookLM) for both papers to provide a quick visual understanding of their methodologies.

https://github.com/user-attachments/assets/38b2a063-352a-4203-bbdd-73037d78e49e

---

## 3. Key Scientific Findings & Synthesis

### A. Dynamic ROI Selection (From Paper A)
- **Insight:** Traditional rPPG often uses the whole face (36.8% of studies), leading to noise from hair, eyes, and background.
- **Best Regions:** Analyzing 70 papers proved that the **forehead (24.5%)** and **cheek (21.7%)** areas show significantly higher signal-to-noise ratio (SNR) and accuracy.
- **Limitation:** Subject movement and looking down degrades these ROI signals.

### B. Robust Hybrid Pulse Extraction (From Paper B)
- **Insight:** Real-world applications suffer from severe motion artifacts and ambient light changes.
- **Proposed Solution:** A hybrid algorithm combining **frequency-domain analysis** (for initial pulse rate estimation) and **time-domain processing** (for refinement) significantly enhances robust tracking under noisy conditions.

---

## 4. Relevance & Integration in the KEIKO Scenario
During the Human-Robot Collaboration (KEIKO) task (arranging blocks with a robot), the subject's face moves and they frequently look down. To address this:
1. **Vision Pipeline (Paper A):** We will implement a landmark tracker to focus dynamically on the **forehead and cheeks** instead of the whole face.
2. **Signal Pipeline (Paper B):** We will process these localized skin signals using a **hybrid frequency-time domain filter** to prevent motion from ruining the heart-rate calculation.

---

## 5. Next Steps for Implementation
- [ ] Implement Face Landmark detector (MediaPipe) to segment the forehead and cheek patches.
- [ ] Build a robust hybrid processing script (combining FFT and peak detection) based on Paper B.
- [ ] Validate the contactless system against the Shimmer3R GSR+ Unit during physical activities.
