# YOLOv11n-CDL Reproduction + Low-Light Robustness Extension — Report

## Research question
Does CDL's design (ConvSmart + DSA + P2 head) make YOLOv11n more robust to low light than
stock YOLOv11n, does low-light training augmentation improve it, and can an AGE-style learned
gamma correction add further robustness, with or without a brightness gate?

## Method
See `common.py`, `cdl.yaml`, `cdl_age.yaml`, `cdl_age_gated.yaml` on Drive and
`YOLOv11n-CDL_reconstruction_spec.md`. Ladder: R1 baseline, R2 CDL, R3 CDL+aug,
R4 CDL+aug+naive AGE, R5 CDL+aug+gated AGE. Single seed per model.

## Results (mAP@0.5, test)
| Model | clean | dark_mild | dark_severe |
|---|---|---|---|
| R1 baseline | 27.4 | 11.2 | 0.8 |
| R2 CDL | 30.1 | 10.9 | 1.3 |
| R3 CDL + aug | 29.4 | 14.4 | 2.8 |
| R4 + naive AGE | 27.5 | 14.5 | 2.8 |
| R5 + gated AGE | 30.2 | 15.6 | 3.3 |

Full tables: `main_results_table.csv`, `per_class_ap_table.csv`, `degradation_table.csv`,
`per_country_map_R2_vs_R5.csv`, `figures/training_curves.png`.

## Analysis
- **CDL vs baseline:** R2 improves clean mAP50 by 2.7 points (27.4 -> 30.1), consistent in direction
  with the paper, but gives no low-light benefit on its own (mild: 10.9 vs 11.2).
- **Augmentation:** low-light training augmentation is the largest single improvement in dark
  conditions (R3 vs R2: mild +3.5, severe +1.5) at a small clean cost (-0.7).
- **Naive AGE (R4):** no dark-condition gain over R3 and lower clean mAP (27.5 vs 29.4).
  A likely cause is that gamma is applied unconditionally, including to well-lit images.
- **Gated AGE (R5):** restores clean mAP to the level of R2/R3 (30.2) and scores slightly higher than R3 on
  both dark sets (+1.1 mild, +0.5 severe). With one seed and 403 test images, these gaps are within
  plausible run-to-run variance, so R5 should be read as comparable to R3 on dark data rather than a
  confirmed improvement. The clearer result is R4 -> R5: the gate removes the clean-image regression.

## Limitations
See the Limitations cell and `decisions_log.md`.
