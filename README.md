# Low-Light Robustness of YOLOv11n-CDL on RDD2022

**Group No.: 19**  
**Project Area:** Road Damage Detection

## Team Members

| Roll Number | Name |
|---|---|
| 2023BCS0086 | Hima Prasobh |
| 2023BCS0179 | K. Saranya |
| 2023BCS0097 | Irine Anna Johnson |

## 1. Project Overview

This project investigates the low-light robustness of a lightweight road-damage detector based on YOLOv11n-CDL using a controlled subset of the Road Damage Dataset 2022 (RDD2022).

Road damage such as cracks and potholes can be difficult to detect when images have reduced brightness, low contrast, noise, shadows, or complex backgrounds. This project evaluates architectural improvements, low-light training augmentation, and adaptive gamma correction under clean and synthetic low-light conditions.

The project combines selected ideas from two research papers:

- **YOLOv11n-CDL:** ConvSmart, Double-Stage Attention (DSA), and a P2 small-object detection path.
- **TinyDark-YOLO:** the Adaptive Gamma Enhancement (AGE) concept.

It extends these ideas through low-light training augmentation, synthetic dark test sets, and a brightness-gated AGE module.

> **Scope:** This is an extension study, not a direct reproduction of either source paper. The AIFI and LEHead modules from TinyDark-YOLO are not implemented.

## 2. Research Questions

1. **RQ1:** Does CDL improve YOLOv11n detection performance on the selected RDD2022 subset?
2. **RQ2:** Does CDL alone improve robustness under low-light conditions?
3. **RQ3:** Do low-light training augmentation and learned gamma correction help, with or without a brightness gate?

## 3. Model Variants

The experiment evaluates five variants, R1–R5.

| Variant | Configuration |
|---|---|
| R1 — Baseline | Stock YOLOv11n |
| R2 — CDL | YOLOv11n + ConvSmart + DSA + P2 detection path |
| R3 — CDL + augmentation | R2 + low-light training augmentation |
| R4 — CDL + AGE + augmentation | R3 + learned, non-gated AGE |
| R5 — CDL + gated AGE + augmentation | R3 + brightness-gated AGE |

### Component Attribution

| Component | Source or contribution | Used in |
|---|---|---|
| ConvSmart | Adapted from YOLOv11n-CDL | R2–R5 |
| Double-Stage Attention (DSA) | Adapted from YOLOv11n-CDL | R2–R5 |
| P2 detection path | Adapted from YOLOv11n-CDL | R2–R5 |
| Adaptive Gamma Enhancement (AGE) | Inspired by TinyDark-YOLO; modified for this project | R4–R5 |
| Low-light training augmentation | Project-specific implementation | R3–R5 |
| Synthetic dark test sets | Project-specific evaluation protocol | Evaluation |
| Brightness-gated AGE | Project-specific extension | R5 |
| AIFI and LEHead | TinyDark-YOLO components | Not implemented |

## 4. Dataset

The experiment uses a reduced RDD2022 subset containing images from seven source groups:

- Japan
- India
- Czech
- Norway
- United States
- China MotorBike
- China Drone

Four road-damage classes are evaluated.

| Class ID | Description |
|---|---|
| `D00` | Longitudinal crack |
| `D10` | Transverse crack |
| `D20` | Alligator crack |
| `D40` | Pothole |

### Frozen Dataset Split

| Split | Number of images |
|---|---:|
| Training | 3,198 |
| Validation | 398 |
| Test | 403 |

The frozen split is used for the reported comparisons. The full experiment uses 60 training epochs, AdamW, cosine learning-rate scheduling, batch size 16, and 640-pixel input images. The reported experiment contains one run per model.

## 5. Low-Light Evaluation

Two synthetic dark test sets are derived from the frozen test split:

- **Mild dark:** linear-light exposure scaled by `0.25`, followed by Poisson–Gaussian noise.
- **Severe dark:** linear-light exposure scaled by `0.08`, followed by Poisson–Gaussian noise.

The test corruption pipeline is separate from the low-light training augmentation pipeline. These controlled synthetic conditions do not replace evaluation on real nighttime road imagery.

## 6. Results

The following table reports the saved experiment's mean Average Precision at IoU 0.5 (**mAP50**). Values are percentages.

| Variant | Clean mAP50 | Mild-dark mAP50 | Severe-dark mAP50 |
|---|---:|---:|---:|
| R1 — Baseline | 27.434 | 11.248 | 0.848 |
| R2 — CDL | 30.090 | 10.856 | 1.254 |
| R3 — CDL + augmentation | 29.430 | 14.417 | 2.831 |
| R4 — CDL + AGE + augmentation | 27.460 | 14.460 | 2.799 |
| R5 — CDL + gated AGE + augmentation | 30.180 | 15.565 | 3.307 |

### Key Observations

- R2 improves clean mAP50 over the baseline, but CDL alone does not improve the mild-dark result in this experiment.
- R3 improves the observed mild-dark and severe-dark results over R2, suggesting that low-light training augmentation contributes to robustness under the synthetic dark conditions.
- R4 achieves a similar mild-dark result to R3 but has a lower clean score.
- R5 records the highest observed mAP50 on both dark test sets while maintaining a clean score close to R2.
- The results come from one run per model and a 403-image test set. Small differences should be interpreted cautiously and do not establish statistically confirmed superiority.

Fresh training may produce different results because model training is stochastic. The table represents the saved experiment outputs.

## 7. Repository Structure

The repository should include the code, notebooks, checkpoints, frozen data split, dark test sets, results, figures, and supporting documentation needed to inspect the experiment.

```text
repository-root/
├── README.md
├── notebooks/
│   ├── RDD2022_YOLOv11n_CDL+Adaptive_AGE _full_experiment_from_scratch.ipynb
│   └── rdd2022-project.ipynb
├── common.py
├── frozen_split/
│   └── rdd2022_yolo_4k/
├── test_dark_mild/
├── test_dark_severe/
├── runs/
│   ├── R1_baseline_scratch/
│   ├── R2_cdl_scratch/
│   ├── R3_cdl_aug/
│   ├── R4_cdl_age_aug_v2/
│   └── R5_cdl_gated_age_aug/
├── main_results_table.csv
└── results, figures, and supporting documentation
```

This is an indicative structure. Keep the paths consistent with the actual repository and notebook configuration.

Large datasets and model checkpoints may need to be stored outside GitHub if they exceed repository file-size limits. If they are stored separately, document how to obtain them and where the notebooks expect them. Never commit credentials, access tokens, or other secrets.

## 8. Notebooks

### 8.1 Full Experiment Notebook

**File:** `notebooks/RDD2022_YOLOv11n_CDL+Adaptive_AGE _full_experiment_from_scratch.ipynb`

This notebook documents the end-to-end experimental workflow:

1. Set up the runtime and dependencies.
2. Prepare the reduced RDD2022 dataset, YOLO labels, and frozen split.
3. Define custom modules and model configurations.
4. Train and evaluate R1–R5.
5. Generate synthetic mild-dark and severe-dark test sets.
6. Produce result tables, plots, and further analyses.

**Important:** The notebook may reuse existing checkpoints or run outputs when they are available. Inspect the run and checkpoint paths before describing a particular execution as fresh training from scratch.

### 8.2 Saved-Checkpoint Reproduction Notebook

**File:** `notebooks/rdd2022-project.ipynb`

This notebook reproduces the evaluation of the saved checkpoints without retraining:

1. Locate the dataset and saved artifacts.
2. Register the custom modules required by the checkpoints.
3. Load the saved `best.pt` checkpoints for R1–R5.
4. Evaluate the models on clean, mild-dark, and severe-dark test sets.
5. Compare reproduced metrics with `main_results_table.csv`.
6. Inspect per-class AP50 and performance degradation under darkness.

This notebook requires the matching checkpoints, custom module definitions, test datasets, dataset YAML files, and saved CSV results at the configured paths.

## 9. How to Run

Before running the notebooks:

1. Use a Python environment compatible with the package versions specified in the notebook.
2. Make the frozen dataset split and dark test sets available.
3. Ensure `common.py`, the five model checkpoints, dataset YAML files, and result CSVs are present.
4. Review and update the configured paths for your environment.
5. Run the **full experiment notebook** for the training and end-to-end workflow.
6. Run the **reproduction notebook** when you want to evaluate existing checkpoints and compare metrics without retraining.

A GPU is recommended for full training. The reproduction workflow avoids training but still requires sufficient resources to load the models and evaluate the datasets. Refer to the notebooks for the exact setup and execution commands.

## 10. Limitations and Future Work

1. The experiment uses a reduced RDD2022 subset rather than the full benchmark.
2. The held-out test split contains 403 images.
3. Only one run per model is reported; variability across random seeds is not measured.
4. The dark test sets are synthetic and may not represent all real nighttime conditions.
5. AGE is modified for this project and is not identical to the complete TinyDark-YOLO architecture.
6. The AIFI and LEHead modules from TinyDark-YOLO are not implemented.
7. The brightness-gated AGE extension requires further validation.
8. Future work should include real nighttime road imagery, multiple training seeds, larger-scale evaluation, and edge-device testing.

## 11. References

1. J. Dai and Y. Gao, “YOLOv11n-CDL: Accurate and Lightweight Pavement Defect Detection via Enhanced Multi-Scale Attention and Feature Fusion,” *Journal of Civil Engineering and Management*, vol. 32, no. 1, pp. 119–132, 2026. https://doi.org/10.3846/jcem.2026.26166
2. X. Zheng, L. Chen, and J. Zhang, “TinyDark-YOLO for adaptive and lightweight object detection in low-light conditions,” *Scientific Reports*, vol. 16, article 25712, 2026. https://doi.org/10.1038/s41598-026-58443-9
3. D. Arya et al., “RDD2022: A multi-national image dataset for automatic road damage detection,” 2022.

## Citation and Attribution

This project adapts and combines selected ideas from the cited studies. ConvSmart, DSA, and the P2 detection path are attributed to YOLOv11n-CDL. AGE is inspired by TinyDark-YOLO and modified for this experiment. Low-light training augmentation, synthetic dark evaluation, and brightness-gated AGE are project-specific extensions.

Please cite the original papers when discussing or reusing their methods.

---

**Group 19 — Road Damage Detection**  
**Project:** Low-Light Robustness of YOLOv11n-CDL on RDD2022
