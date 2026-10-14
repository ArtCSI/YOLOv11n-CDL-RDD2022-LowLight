# Low-Light Robustness of YOLOv11n-CDL on RDD2022

A controlled study of road-damage detection under normal illumination
and **synthetic low-light conditions**. We compare five YOLOv11n-based
variants on a fixed subset of the Road Damage Dataset 2022 (RDD2022).

-   **Repository:**
    https://github.com/ArtCSI/YOLOv11n-CDL-RDD2022-LowLight
-   **Project artifacts:** [Shared Google Drive
    folder](https://drive.google.com/drive/folders/1JnWVzfztHVAJXTQTd-hIwviLU57PEOmJ?usp=drive_link)
-   **Group:** 19
-   **Affiliation:** Department of Computer Science and Engineering,
    Indian Institute of Information Technology Kottayam, Kerala, India

> **Important:** The shared Drive folder may require access permission.
> If you cannot open it, request access from the project owner. The
> notebooks, model checkpoints, dataset, and generated test sets may not
> all be stored in Git because they are large.

## Team members

| Roll number | Name |
|---|---|
| 2023BCS0086 | Hima Prasobh |
| 2023BCS0179 | K. Saranya |
| 2023BCS0097 | Irine Anna Johnson |

**Group:** 19  
**Department:** Computer Science and Engineering  
**Institution:** Indian Institute of Information Technology Kottayam, Kerala, India

## Contents

-   [Project overview](#project-overview)
-   [Model variants](#model-variants)
-   [Dataset and evaluation protocol](#dataset-and-evaluation-protocol)
-   [Results](#results)
-   [Repository contents](#repository-contents)
-   [Choose how to run the project](#choose-how-to-run-the-project)
-   [Option A: Evaluate saved
    checkpoints](#option-a-evaluate-saved-checkpoints-no-retraining)
-   [Option B: Run the full
    experiment](#option-b-run-the-full-experiment-and-train)
-   [Run in Google Colab](#run-in-google-colab)
-   [Run locally](#run-locally)
-   [Required files and folder
    layout](#required-files-and-folder-layout)
-   [Troubleshooting](#troubleshooting)
-   [Limitations and attribution](#limitations-and-attribution)

## Project overview

Road-damage detectors can lose accuracy when cracks and potholes are
observed under reduced illumination, low contrast, or noise. We
investigate whether selected architecture changes, low-light training
augmentation, and adaptive gamma correction improve YOLOv11n performance
under controlled synthetic dark conditions.

Our work adapts selected ideas from two studies:

1.  **YOLOv11n-CDL:** ConvSmart, Double-Stage Attention (DSA), and a P2
    higher-resolution detection path.
2.  **TinyDark-YOLO:** the idea of Adaptive Gamma Enhancement (AGE),
    adapted for this road-damage experiment.

This is an extension study, not a complete reproduction of either source
paper. We do **not** implement the AIFI or LEHead components from
TinyDark-YOLO. The low-light training augmentation, synthetic dark test
protocol, and brightness-gated AGE are project-specific extensions.

### Research questions

-   Does the CDL architecture improve detection over stock YOLOv11n on
    our selected RDD2022 subset?
-   Does CDL alone improve performance under low-light conditions?
-   Do low-light training augmentation and AGE, with or without a
    brightness gate, improve performance across illumination conditions?

## Model variants

  -----------------------------------------------------------------------
  Variant                 Configuration           Purpose
  ----------------------- ----------------------- -----------------------
  **R1 --- Baseline**     Stock YOLOv11n          Reference model

  **R2 --- CDL**          R1 + ConvSmart + DSA +  Evaluate selected CDL
                          P2 detection path       components

  **R3 --- CDL +          R2 + low-light training Evaluate training-time
  augmentation**          augmentation            exposure/noise
                                                  variation

  **R4 --- Non-gated      R3 + learned, non-gated Evaluate gamma
  AGE**                   AGE                     correction without a
                                                  brightness gate

  **R5 ---                R3 + AGE with           Reduce the correction
  Brightness-gated AGE**  brightness-dependent    for brighter inputs
                          gating                  
  -----------------------------------------------------------------------

R4 and R5 are alternative extensions of R3. AGE was selected because
gamma correction directly addresses illumination changes. We recreated
the gamma-correction formulation described in the source paper, but the
learned gamma predictor was unstable in our experiments. We therefore
used a constrained, formula-based adaptation. Since the original source
code was unavailable, we do not claim an identical reproduction of the
original implementation.

For R5, the brightness gate is:

``` text
g(b) = clip(1 - b / 0.4, 0, 1)
gamma_effective = 1 + g(b) * (gamma - 1)
```

Here, `b` is the mean brightness of the input. When `b >= 0.4`, the gate
is zero and the transformation becomes the identity
(`gamma_effective = 1`).

## Dataset and evaluation protocol

We use a reduced RDD2022 subset with the following frozen split:

  Split             Images   Label files
  ------------ ----------- -------------
  Training           3,198         3,198
  Validation           398           398
  Test                 403           403
  **Total**      **3,999**     **3,999**

The four road-damage classes are:

  Class ID   Description
  ---------- -------------------------
  `D00`      Longitudinal crack
  `D10`      Transverse crack
  `D20`      Alligator/network crack
  `D40`      Pothole

### Test conditions

We evaluate the same frozen 403-image test split under three conditions:

-   **Clean:** original held-out test images.
-   **Mild dark:** exposure scale `0.25`, followed by Poisson--Gaussian
    noise.
-   **Severe dark:** exposure scale `0.08`, followed by
    Poisson--Gaussian noise.

These dark sets are synthetic stress tests, not real nighttime captures.
The evaluation corruption is kept separate from the training
augmentation. The original boxes are retained because the test
corruption changes image appearance rather than object geometry.

### Training configuration

  -----------------------------------------------------------------------
  Setting                             Value
  ----------------------------------- -----------------------------------
  Input size                          640 × 640

  Epochs                              60

  Optimizer                           AdamW

  Learning-rate schedule              Cosine

  Batch size                          16

  Main metrics                        Precision, recall, mAP@0.5,
                                      mAP@0.5:0.95, per-class AP

  Reported runs                       One saved run per variant
  -----------------------------------------------------------------------

A GPU runtime is strongly recommended for full training. Exact results
may vary if the models are trained again.

## Results

The table below summarizes the saved experiment. mAP@0.5 values are
percentages.

  Variant                          Clean   Mild dark   Severe dark
  ----------------------------- -------- ----------- -------------
  R1 --- Baseline                 27.434      11.248         0.848
  R2 --- CDL                      30.090      10.856         1.254
  R3 --- CDL + augmentation       29.430      14.417         2.830
  R4 --- Non-gated AGE            27.460      14.460         2.799
  R5 --- Brightness-gated AGE     30.180      15.565         3.307

### Interpretation

-   R2 improves clean mAP@0.5 over R1, but CDL alone does not improve
    the mild-dark score in this experiment.
-   R3 improves over R2 on both synthetic dark sets, suggesting that
    low-light training augmentation contributes to the observed gains.
-   R5 has the highest observed mAP@0.5 on the clean, mild-dark, and
    severe-dark sets among the saved runs.
-   The absolute severe-dark scores remain low for every variant.
-   Only one run per model is reported, so these results are descriptive
    and do not establish statistical superiority across random seeds.

## Repository contents

The expected project layout is similar to the following. Some large data
files or generated artifacts may be supplied separately rather than
tracked in Git.

``` text
YOLOv11n-CDL-RDD2022-LowLight/
├── README.md
├── common.py
├── notebooks/
│   ├── 01_reproduce_results.ipynb
│   └── RDD2022_YOLOv11n_CDL_Adaptive_AGE_full_experiment_from_scratch.ipynb
├── frozen_split/
│   └── rdd2022_yolo_4k.zip
├── test_dark_mild/
├── test_dark_severe/
├── runs/
│   ├── R1_baseline_scratch/weights/best.pt
│   ├── R2_cdl_scratch/weights/best.pt
│   ├── R3_cdl_aug/weights/best.pt
│   ├── R4_cdl_age_aug_v2/weights/best.pt
│   └── R5_cdl_gated_age_aug/weights/best.pt
├── main_results_table.csv
├── results_tables.csv
├── per_class_ap_table.csv
├── degradation_table.csv
└── figures/
```

Notebook filenames can vary slightly between repository revisions. Use
the notebook that matches the workflow described below, and update any
path settings if your local filenames differ.

## Choose how to run the project

  -----------------------------------------------------------------------
  Goal                    Workflow                Retrains models?
  ----------------------- ----------------------- -----------------------
  Check saved R1--R5      **Reproduction          No
  checkpoints against the notebook**              
  saved metrics                                   

  Rebuild the pipeline    **Full-experiment       Yes, unless a cell
  and train R1--R5        notebook**              explicitly reuses an
                                                  existing checkpoint
  -----------------------------------------------------------------------

**Start with Option A if your goal is to verify the reported results.**
Full training takes substantially more time and storage.

## Option A: Evaluate saved checkpoints (no retraining)

This workflow loads existing checkpoints and evaluates them on the
clean, mild-dark, and severe-dark test sets. It does not train the
models or recreate the frozen split.

### 1. Get the repository

``` bash
git clone https://github.com/ArtCSI/YOLOv11n-CDL-RDD2022-LowLight.git
cd YOLOv11n-CDL-RDD2022-LowLight
```

### 2. Obtain the project artifacts

Open the [shared project Drive
folder](https://drive.google.com/drive/folders/1JnWVzfztHVAJXTQTd-hIwviLU57PEOmJ?usp=drive_link).
If you have access, add a shortcut to the project folder in your Google
Drive named:

``` text
My Drive/rdd2022_project
```

The reproduction notebook expects the saved artifacts under the project
root. Before running, confirm that the project directory contains:

-   `common.py`
-   `frozen_split/rdd2022_yolo_4k.zip`
-   `test_dark_mild/`
-   `test_dark_severe/`
-   `runs/` with the five saved `best.pt` checkpoints
-   The saved results CSV files used by the notebook

If you cannot access the shared folder, request permission from the
project owner. Do not replace missing checkpoints with newly trained
weights if your goal is to verify the original saved results.

### 3. Open the reproduction notebook

Open `notebooks/01_reproduce_results.ipynb` in Google Colab. If the
notebook has a different filename in the repository, select the notebook
whose title says that it reproduces or validates saved results without
retraining.

### 4. Enable the runtime and run setup cells

For a GPU-backed runtime, choose **Runtime → Change runtime type → T4
GPU** when that option is available. Run the notebook's setup cells. The
experiment pins these key versions:

``` python
%pip install -q "ultralytics==8.3.152" "albumentations==1.4.18" --no-deps
%pip install -q "albucore==0.0.17" "eval-type-backport"
```

The notebook also relies on a compatible PyTorch/torchvision
installation provided by the runtime. Do not blindly upgrade PyTorch
independently of the notebook environment. Restart the runtime if the
setup cell asks you to.

### 5. Check paths and artifacts

Run the validation cells and confirm that the frozen split, both dark
test sets, `common.py`, and all five checkpoints are detected. If a path
is missing, correct the Drive shortcut or notebook path before
continuing.

### 6. Run all cells in order

Run the reproduction notebook from top to bottom. It loads the saved
model weights, registers the custom model modules, evaluates the three
test conditions, and compares reproduced mAP@0.5 values with the saved
result table.

Review the printed metric tables and any generated CSV files. This
workflow evaluates existing checkpoints; it is not a fresh training run.

## Option B: Run the full experiment and train

Use this workflow if you want to rebuild the data preparation and
training pipeline. It is compute-intensive; a GPU runtime and sufficient
storage are recommended.

### 1. Clone the repository and open the full-experiment notebook

``` bash
git clone https://github.com/ArtCSI/YOLOv11n-CDL-RDD2022-LowLight.git
cd YOLOv11n-CDL-RDD2022-LowLight
```

Open the full-experiment notebook from `notebooks/` in Google Colab and
enable a GPU runtime if available.

### 2. Configure persistent storage

The full-experiment notebook was designed for Google Colab and uses:

``` text
/content/drive/MyDrive/rdd2022_project
```

Mount Google Drive when prompted and ensure that the project folder is
available at that location. The notebook stores checkpoints, frozen
data, generated dark test sets, figures, and run logs under the project
directory so they can survive session disconnections.

### 3. Install the pinned dependencies

Run the notebook's dependency cell. The key pinned packages are:

``` python
PINS = {
    "ultralytics": "8.3.152",
    "albumentations": "1.4.18",
}
%pip install -q "ultralytics==8.3.152" "albumentations==1.4.18" --no-deps
%pip install -q "albucore==0.0.17" "eval-type-backport"
```

Keep the notebook's PyTorch and torchvision versions compatible with the
selected runtime.

### 4. Prepare the dataset

The full-experiment notebook downloads the RDD2022 archive from the
official Figshare record when needed:

-   Dataset record: https://doi.org/10.6084/m9.figshare.21431547

The full archive is large and download/extraction may take time. Follow
the notebook's preparation cells to create the YOLO-format data and the
frozen split. Do not regenerate the split when comparing against the
saved experiment; use the existing frozen split if it is available.

### 5. Run the experiment in order

Run all cells from top to bottom. The pipeline prepares the dataset and
labels, defines/registers custom model components, trains R1--R5,
creates the synthetic dark evaluation sets, evaluates saved checkpoints,
and writes tables and plots.

Some cells may reuse outputs or checkpoints that already exist. Check
the logs and paths before interpreting a run as freshly trained. If you
want a new independent run, use a separate output directory and record
the configuration and random seed; do not overwrite the saved artifacts
used for the reported results.

### 6. Inspect the outputs

The project uses artifacts such as:

-   `runs/<run_name>/weights/best.pt`
-   `main_results_table.csv`
-   `results_tables.csv`
-   `per_class_ap_table.csv`
-   `degradation_table.csv`
-   plots and evaluation summaries under the configured output directory

Use the actual output paths printed by the notebook. Fresh training may
not reproduce the exact reported metrics because model training is
stochastic.

## Run locally

Local execution is possible, but the notebooks were designed around
Google Colab and Google Drive. You may need to adjust paths and install
compatible GPU/PyTorch dependencies.

### 1. Create a Python environment

**Windows PowerShell**

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyterlab
```

**Linux/macOS**

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab
```

These commands set up the environment and JupyterLab only. Install the
model dependencies using the notebook's pinned setup instructions.

### 2. Get the code and data

Clone the repository, obtain the dataset/artifacts, and ensure that
`common.py`, the frozen split, both dark test sets, the checkpoint
files, and the required result CSVs are present. The reproduction
notebook will fail if required artifacts are missing.

### 3. Adjust paths

Update Google Drive paths in the notebook to local paths. Keep the
expected folder structure intact or update every corresponding path
setting.

### 4. Launch JupyterLab

``` bash
jupyter lab
```

Open the appropriate notebook and run it from top to bottom. Full
training on CPU is not recommended for practical use; a compatible GPU
is strongly preferred.

## Required files and folder layout

  ------------------------------------------------------------------------
  File or folder                       Purpose
  ------------------------------------ -----------------------------------
  `common.py`                          Custom modules such as ConvSmart
                                       and DSA, plus registration
                                       utilities

  `frozen_split/rdd2022_yolo_4k.zip`   Frozen train/validation/test
                                       dataset

  `test_dark_mild/`                    Mild-dark test images and labels

  `test_dark_severe/`                  Severe-dark test images and labels

  `runs/.../weights/best.pt`           Saved weights for R1--R5

  `main_results_table.csv` or          Saved metrics for comparison
  `results_tables.csv`                 

  `per_class_ap_table.csv`             Per-class AP results, when supplied

  Dataset YAML files                   Paths and class names used for
                                       evaluation
  ------------------------------------------------------------------------

A notebook file by itself is not enough to reproduce the reported
results. Large datasets and checkpoints may be stored outside Git;
obtain them from the linked project artifacts.

## Troubleshooting

### `FileNotFoundError` for data, YAML, checkpoint, or CSV files

-   Check the exact path printed in the error.
-   Confirm the dataset or artifact has been downloaded or mounted.
-   In Colab, verify that the Drive shortcut is named `rdd2022_project`
    and is under `My Drive`.
-   Update the notebook's path configuration if your directory layout
    differs.

### Custom module or checkpoint loading errors

-   Ensure `common.py` is present.
-   Run the custom-module import/registration cells before loading
    checkpoints.
-   Use the matching checkpoint and code version. Replacing custom
    modules with stock YOLO modules changes the experiment.

### CUDA/GPU is unavailable

-   Enable a GPU runtime in Colab, if available.
-   Check that PyTorch detects the GPU with `torch.cuda.is_available()`.
-   Evaluation may run on CPU but will be slower. Full training is best
    run with a compatible GPU.

### Package/version errors

-   Follow the pinned versions in the notebook.
-   Avoid upgrading PyTorch, torchvision, Ultralytics, or Albumentations
    independently.
-   Restart the runtime after changing core packages and rerun the setup
    cells.

### Reproduced metrics differ from the saved CSV

-   Confirm the correct checkpoint, frozen split, class mapping, and
    test-set paths.
-   Check that the notebook evaluates the clean, mild-dark, and
    severe-dark sets in the intended order.
-   The reproduction workflow evaluates saved checkpoints; training new
    checkpoints can yield different values.

## Limitations and attribution

-   We use a reduced RDD2022 subset, not the full benchmark.
-   The held-out test split contains 403 images.
-   One saved run per variant is reported; run-to-run variance is not
    established.
-   The dark test sets are synthetic and do not represent all real
    nighttime conditions.
-   AGE is a formula-based adaptation inspired by TinyDark-YOLO, not a
    claim of identical reproduction of the full implementation.
-   AIFI and LEHead from TinyDark-YOLO are not implemented.
-   The brightness gate is a project-specific extension and requires
    further validation.
-   Multiple training seeds, independent real nighttime images, and
    deployment-oriented benchmarks remain useful follow-up work.

### References

1.  J. Dai and Y. Gao, "YOLOv11n-CDL: Accurate and Lightweight Pavement
    Defect Detection via Enhanced Multi-Scale Attention and Feature
    Fusion," *Journal of Civil Engineering and Management*, 2026.
    https://doi.org/10.3846/jcem.2026.26166
2.  X. Zheng, L. Chen, and J. Zhang, "TinyDark-YOLO for adaptive and
    lightweight object detection in low-light conditions," *Scientific
    Reports*, 2026. https://doi.org/10.1038/s41598-026-58443-9
3.  D. Arya et al., "RDD2022: A multi-national image dataset for
    automatic road damage detection," 2022.
    https://doi.org/10.6084/m9.figshare.21431547

We acknowledge the source methods and distinguish them from our
project-specific low-light augmentation, synthetic test protocol,
constrained AGE adaptation, and brightness gate.

------------------------------------------------------------------------

**Quick start:** If you only need to verify the reported metrics, use
the reproduction notebook with the saved artifacts. If you want to
retrain the models, use the full-experiment notebook and a GPU runtime.
