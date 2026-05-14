


# Paper Summary: rPPG Estimation: Vision Transformer With 3-D Temporal Central Difference

## 1. Metadata
- **Title:** rPPG Estimation: Vision Transformer With 3-D Temporal Central Difference
- **Authors:** Mohamed Khalil Ben Salah, Philippe Jouvet, and Rita Noumeir
- **Journal:** IEEE Transactions on Instrumentation and Measurement (2025)
- **Source Link:** [IEEE Xplore - Official](https://ieeexplore.ieee.org/document/10912685)
- **Status:** [x] Read | [ ] In Progress

## 2. Visual Summary (AI-Generated)
> [!TIP]
> This video summary covers the feed-forward integration of 3D-CNNs and ViViTs as explained in the paper.



https://github.com/user-attachments/assets/c9de7444-7e5c-49cf-9cf3-118dd567108c



---

## 3. Problem Statement
The paper addresses the inability of traditional CNN-based rPPG models to capture long-range temporal dependencies and their sensitivity to motion noise. It aims to create a robust system for contact-free heart rate monitoring, essential for clinical settings and scenarios like the COVID-19 pandemic.

## 4. Methodology: Sequential Hybrid Architecture
This paper proposes a **feed-forward (sequential) integration** of CNNs and Transformers, which is more efficient than parallel dual-branch architectures:
1. **3D-CNN Feature Extraction:** Captures local spatiotemporal features using **3-D Temporal Central Difference Convolution (3DCDC-T)** and **Convolutional Block Attention Module (CBAM)**. These layers specifically detect subtle temporal gradients (pulse variations) while suppressing background noise.
2. **Video Vision Transformer (ViViT):** The refined feature maps are passed to a ViViT where **Multihead Self-Attention (MHSA)** models global contextual relationships and long-range temporal dependencies across frames.
3. **Progressive Refinement:** This approach builds a comprehensive representation of data from local variations to global patterns.

## 5. Key Findings & Performance
- **Accuracy Improvement:** Achieved a **22.55% improvement in MAE** and a **55.80% improvement in RMSE** on the UBFC-rPPG dataset compared to state-of-the-art models.
- **ROI Optimization:** Scientific testing proved that combining **Forehead and Cheeks** as ROIs significantly outperforms using the full face or individual regions alone.
- **Real-Time Efficiency:** The model is optimized for real-time sensing with an inference time of only **0.22 seconds** per video.
- **Generalization:** Demonstrated superior performance in cross-dataset testing on the ECG-Fitness dataset, showing robustness against motion and lighting changes.

## 6. Strategic Relevance to KEIKO Project
- **Motion Handling:** The 3DCDC-T technique is the ideal solution for the motion artifacts expected in our **KEIKO block-arranging scenario**.
- **Implementation Goal:** I will focus on implementing the "Forehead + Cheeks" combined ROI strategy and explore the 3D-CNN + Transformer hybrid backend for our lab setup.

## 7. Implementation Tasks
- [ ] Set up the environment for ViViT and 3D-CNN hybrid inference in Python.
- [ ] Implement the 3DCDC-T layer logic to handle temporal gradients.
- [ ] Validate the model’s 0.22s inference speed on local hardware to ensure live-feed compatibility.
