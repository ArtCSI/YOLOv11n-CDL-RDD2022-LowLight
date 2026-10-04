# Decisions log

| item | choice | reason |
|---|---|---|
| pinned versions | {"ultralytics": "8.3.152", "torch": "2.11.0+cu128", "albumentations": "1.4.18"} | ground rule 8 — record exact resolved versions once, at the top of the run |
| filename prefix bug | use loop variable `src` directly instead of img_path.parent.parent.parent.name | old logic depended on zip nesting depth, which RDD2022 releases don't keep consistent |
| pool size / split method | ~4,000 images (3,200/400/400), contiguous blocks of sorted filenames per country | spec Section 2 fixes #2 and #3 — smaller pool for 3-day Colab budget; contiguous blocks reduce (not eliminate) near-duplicate-frame leakage vs random shuffling |
| keep_c2psa | kept (reading (a): SPPF -> C2PSA -> DSA) | spec 4.2 default; Table 1's 3.3M param figure for +CD is closer to backbone+C2PSA+DSA than to C2PSA removed — verify against this build's own param count at Checkpoint 1 |
| DSA fusion conv | Conv1x1 + BN + SiLU for both fusion points | spec 4.2 — paper leaves this unspecified |
| IBN placement | backbone layers 0, 1, 3 only (stem + P2/4 + P3/8 ConvSmart); layers 5, 7 and neck stay plain BN | spec 10.2 — deeper features carry semantic content IN can damage |
| epochs / patience | 60 epochs, patience 15, identical for R1-R4 | user-chosen budget; same rule for every run so the comparison is fair |
| optimizer lr | AdamW, lr0=0.00125, cos_lr=True | explicit 'AdamW' defaults to lr0=0.01; 0.00125 matches what optimizer='auto' chose for nc=4 |
| image cache | cache='disk' | RAM cache exceeded free Colab RAM on the earlier 8k run and is non-deterministic |
| pipeline B photon peak | poisson_peak=255 | peak=30 gave SNR ~1-2 (noise swamps signal); 255 keeps darkness the dominant effect |
| dark test-set params | pipeline B, exposure_scale mild=0.25 severe=0.08, seed=0, generated once | spec 10.3 — physically motivated linear-light scaling + Poisson/Gaussian noise, kept separate from pipeline A's code and parameter ranges (spec caveat 10.5) |
| AGE placement | single AGE layer at index 0, before ConvSmart stem, operating in image space | TinyDark-YOLO design — gamma predicted from a 256x256 downsample, applied to full-res input |
