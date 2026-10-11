# Does a brain tumour model know when it is failing?

3D brain tumour segmentation on **BraTS 2021** with **SwinUNETR**, and a study of what happens when one of the four MRI scans is missing.

**Live demo:** [huggingface.co/spaces/gzone00/brain-tumor-detection](https://huggingface.co/spaces/gzone00/brain-tumor-detection): 20 test patients in a 3D viewer, with the doctors' outline, the model's prediction, the prediction without T1ce, and Grad-CAM maps.

---

## The short version

A SwinUNETR model trained on all four MRI sequences (FLAIR, T1, T1ce, T2) reaches a mean Dice of **0.885** on 100 patients it never saw.

When a scan goes missing, things change:

- **Without FLAIR**, the whole-tumour outline breaks (0.915 → 0.434), but the model's uncertainty goes up about **3×**, so it is at least warning us.
- **Without T1ce**, the active (enhancing) tumour almost disappears (0.860 → **0.015**), and the uncertainty **barely changes**. The model fails quietly.

A second model trained with **modality dropout** (random scans switched off during training) recovers much of this: without FLAIR the whole tumour goes back to 0.831, and without T1ce the active tumour comes back to 0.391, at a cost of about 0.02 Dice when all scans are present.

## Results (100 test patients)

**Dice score** (1.0 = perfect overlap with the doctors' label)

| Model | Missing scan | Whole tumour | Tumour core | Active tumour |
|---|---|---|---|---|
| Normal | none | 0.915 | 0.880 | 0.860 |
| Normal | FLAIR | **0.434** | 0.571 | 0.581 |
| Normal | T1ce | 0.870 | 0.199 | **0.015** |
| Normal | T1 | 0.898 | 0.856 | 0.836 |
| Normal | T2 | 0.885 | 0.847 | 0.853 |
| Dropout | none | 0.911 | 0.837 | 0.855 |
| Dropout | FLAIR | **0.831** | 0.805 | 0.828 |
| Dropout | T1ce | 0.892 | 0.563 | **0.391** |

**Uncertainty** (spread of 8 flipped predictions, inside the tumour area, normal model)

| Condition | Mean Dice | Uncertainty |
|---|---|---|
| All four scans | 0.885 | 0.0172 |
| No FLAIR | 0.545 | 0.0515 (≈3× higher) |
| No T1ce | 0.363 | 0.0196 (barely changes) |

Spearman correlation between uncertainty and Dice: r = −0.115, p = 0.047. Real, but far too weak to catch the missing-T1ce failure.

**Grad-CAM** (share of attention inside the real tumour region)

| Model | Scans | Whole | Core | Active |
|---|---|---|---|---|
| Normal | all four | 36% | 2% | 5% |
| Normal | no T1ce | 19% | 3% | 3% |
| Dropout | all four | 13% | 7% | 4% |
| Dropout | no T1ce | 18% | 9% | 4% |

On one patient early on, Grad-CAM seemed to show the active-tumour attention vanishing without T1ce. On 100 patients that does not hold up: for the small regions only a few percent of the attention lands inside the tumour at all. So Grad-CAM is reported here as a limitation, and the conclusions rest on Dice and uncertainty.

## How it was built

- **Data:** BraTS 2021 adult glioma. 199 training, 49 validation, 100 test patients (one validation case skipped because of a missing file). The test patients were never used for training or model selection.
- **Model:** SwinUNETR (MONAI), 4 input channels, 3 overlapping output regions (whole tumour, tumour core, enhancing tumour), feature size 48, 96³ patches, sliding-window inference.
- **Training:** Dice loss, AdamW (lr 1e-4), cosine schedule, mixed precision, random flips and intensity shifts. Free Kaggle T4 GPU.
  - Normal model: stopped at epoch 70 because of the 12-hour Kaggle limit (best validation Dice 0.887).
  - Dropout model: 100 epochs, each scan dropped with probability 0.5 (up to 3 at once).
- **Missing scan:** simulated by replacing that channel with zeros.
- **Uncertainty:** test-time augmentation with 8 flip combinations; standard deviation inside the predicted tumour area.
- **Grad-CAM:** last decoder stage, computed over 96³ windows and stitched together.

## Files

| File | What it is |
|---|---|
| `project-dropping-out.ipynb` | Training notebook (dropout model) |
| `new-test-00.ipynb` | Normal model training and the 100-patient Dice and uncertainty test |
| `website.ipynb` | Corrected dropout results, Grad-CAM on 100 patients, files for the 20 website patients |
| `dice_all_patients.csv` | Dice per patient, model and missing scan |
| `uncertainty_all_patients.csv` | Uncertainty per patient and condition |
| `gradcam_all_patients.csv` | Grad-CAM attention per patient, model, condition and region |
| `history.json` | Training history of the normal model |
| `index.html`, `results.html`, `missing.html`, `patients.html` | The project website |

Model weights (about 250 MB each) and MRI data are not stored here because of their size. The data is available from the BraTS 2021 challenge.

## Limitations

- One dataset only; results may not carry over to other scanners or hospitals.
- 100 test patients is enough for large differences (0.86 vs 0.02), but small ones (0.880 vs 0.837) should be read with care.
- The two models trained for different numbers of epochs (70 vs 100).
- Zeroing a channel is a simple stand-in for a really missing scan.
- Grad-CAM was not informative for small tumour regions.

This is a research project, not a medical tool.

## Data credit

BraTS 2021 (RSNA-ASNR-MICCAI Brain Tumor Segmentation Challenge). Please cite the BraTS 2021 papers if you use the data.
