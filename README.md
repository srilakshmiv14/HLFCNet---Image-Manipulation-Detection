# HLFC-Net

**A hierarchical latent–frequency cross-attention network for diffusion-based image manipulation detection.**

HLFC-Net is an end-to-end deep network for **pixel-level localisation** of diffusion-based image healing. It jointly exploits three complementary feature streams — a hybrid Swin / ConvNeXt **semantic stem**, a **Latent Trajectory Inversion (LTI)** module that extracts multi-step DDIM error maps from a frozen Stable Diffusion 1.5 prior, and an **Adaptive Phase-Amplitude Frequency (APAF)** module operating in the spectral domain — and reconciles them through bidirectional **Dual-Stream Cross-Fusion (DSCF)** attention. This release accompanies the journal submission and provides the model code, the trained checkpoint, and all reported evaluation outputs needed to reproduce the figures and tables in the paper.

---

## Highlights

- **Pixel-level localisation** of diffusion-based image healing on TGIF and TGIF2-FLUX, with cross-dataset evaluation on DF2023 and a COCO val2017 authentic control set.
- **Latent Trajectory Inversion** — a single, low-noise DDIM trajectory comparison that is precomputed and cached, making training on a single 24 GB GPU feasible.
- **Cross-dataset score-inversion phenomenon** — we identify and quantify a previously undocumented failure mode of diffusion-trained image-level scorers when an authentic real-photograph control is added at evaluation time.

---

## Headline results

| Evaluation                                  | Samples | Pixel-AUC | Pixel-IoU @0.5 | Image-AUC |
|---------------------------------------------|--------:|----------:|---------------:|----------:|
| TGIF / TGIF2-FLUX validation (in-domain)    |  22,165 |    0.730  |         0.111  |    0.558  |
| DF2023 cross-dataset (manipulated only)     |   5,000 |    0.628  |         0.055  |       —   |
| DF2023 + COCO val2017 (with authentic ctrl) |  10,000 |  see `results/df2023_with_authentic_eval/` |  —  | inverted (paper §4.4.3) |

Full per-subset and per-manipulation-type breakdowns are in `results/`.

---

## Repository structure

```
HLFC-Net/
├── README.md                ← this file
├── LICENSE                  ← MIT
├── CITATION.cff             ← citation metadata
├── requirements.txt         ← Python dependencies
├── code/                    ← Python sources (flat layout — preserves imports)
│   ├── hlfcnet_model.py             ← model definition (SemanticStem, LTIModule, APAFModule, DSCFModule, PredictionHead, ImageLevelHead)
│   ├── tgif_flux_unified_dataset.py ← TGIF + TGIF2-FLUX unified loader
│   ├── unified_forensics_dataset.py ← base unified-forensics loader
│   ├── tgif_phase1_loader.py        ← legacy TGIF Phase-1 loader
│   ├── df2023_dataset.py            ← DF2023 cross-dataset loader
│   ├── coco_authentic_dataset.py    ← COCO val2017 authentic loader (eval control)
│   ├── phase7_full_scale_training.py ← main training entry point (LTI precompute + train)
│   ├── train_image_level_head.py    ← image-level head training (frozen backbone)
│   ├── build_coco_train_lti_cache.py ← LTI cache builder for COCO slice
│   ├── thesis_evaluation.py         ← full evaluation pipeline (TGIF val)
│   ├── eval_best_model.py           ← single-checkpoint quick eval
│   ├── eval_df2023_cross.py         ← cross-dataset eval (DF2023, manipulated)
│   ├── eval_df2023_with_authentic.py ← cross-dataset eval with COCO authentic control
│   ├── eval_with_image_head.py      ← evaluation using the learned image-level head
│   ├── generate_viz_grids.py        ← qualitative visualisations (TGIF/FLUX)
│   └── generate_viz_grid_df2023.py  ← qualitative visualisations (DF2023)
├── checkpoints/
│   ├── best_model.pth           ← pixel-level backbone (145 MB)
│   └── image_head_best.pth      ← image-level head (1.5 MB)
├── results/                  ← all evaluation outputs referenced in the paper
│   ├── tgif_val_eval/                ← TGIF/FLUX validation (22,165 samples) — metrics, curves, viz grids, mentor plots
│   ├── df2023_cross_eval/            ← DF2023 cross-dataset (manipulated only)
│   ├── df2023_with_authentic_eval/   ← DF2023 + COCO authentic control (score-inversion analysis)
│   ├── image_head_tgif_val_eval/     ← image-level head on TGIF val
│   ├── image_head_full_eval/         ← image-level head full evaluation
│   └── df2023_viz_grids/             ← Fig. 7 qualitative grids
└── figures/                  ← all figures and supplements as published (.tif)
```

---

## Installation

Tested on Ubuntu 22.04, Python 3.12, CUDA 12.1, and an NVIDIA RTX 3090 Ti (24 GB).

```bash
# 1. Create and activate an environment
python3.12 -m venv hlfc_env
source hlfc_env/bin/activate

# 2. Install PyTorch matching your CUDA version (example: CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install the remaining requirements
pip install -r requirements.txt
```

> **Note on Stable Diffusion 1.5.** The LTI module loads the frozen VAE, U-Net and DDIM scheduler from `runwayml/stable-diffusion-v1-5` via 🤗 `diffusers`. If that repository is unavailable, a community mirror such as `benjamin-paine/stable-diffusion-v1-5` can be used by passing `model_id=...` to `LTIModule`.

---

## Pretrained checkpoints

| File                              | Size  | Description                                        |
|-----------------------------------|------:|----------------------------------------------------|
| `checkpoints/best_model.pth`      | 145 MB | Backbone trained on TGIF + TGIF2-FLUX (Phase 7).   |
| `checkpoints/image_head_best.pth` | 1.5 MB | Image-level head trained on top of the frozen backbone. |

These reproduce the numbers reported in the paper exactly.

---

## Quick start — running the evaluation

The fastest way to reproduce a headline number is the in-domain evaluation on TGIF / TGIF2-FLUX validation:

```bash
cd code

python thesis_evaluation.py \
    --checkpoint  ../checkpoints/best_model.pth \
    --tgif-root   /path/to/TGIF \
    --flux-root   /path/to/TGIF2_FLUX \
    --output-dir  ../results/repro_tgif_val
```

Cross-dataset evaluation on DF2023 with the COCO authentic control:

```bash
python eval_df2023_with_authentic.py \
    --checkpoint   ../checkpoints/best_model.pth \
    --df2023-root  /path/to/DF2023_V15_val \
    --coco-root    /path/to/coco_val2017 \
    --output-dir   ../results/repro_df2023_authentic
```

Evaluation with the image-level head:

```bash
python eval_with_image_head.py \
    --checkpoint   ../checkpoints/best_model.pth \
    --head-checkpoint ../checkpoints/image_head_best.pth \
    --df2023-root  /path/to/DF2023_V15_val \
    --coco-root    /path/to/coco_val2017 \
    --output-dir   ../results/repro_image_head
```

All evaluation scripts accept `--help` for the full argument list.

---

## Training from scratch

End-to-end training is split into two phases for tractability on a single 24 GB GPU.

**Phase A — precompute the LTI cache (one-off, ~5–6 hours on a 3090 Ti):**
The training script automatically caches the frozen-SD output to disk on first use (`--lti-cache /path/to/cache`). Subsequent epochs reuse the cache and complete in ~2 hours each.

**Phase B — main training (~2 hours per epoch with the cache):**

```bash
cd code

python phase7_full_scale_training.py \
    --tgif-root   /path/to/TGIF \
    --flux-root   /path/to/TGIF2_FLUX \
    --lti-cache   /path/to/lti_cache \
    --batch-size  4 \
    --grad-accum  2 \
    --epochs      50 \
    --early-stop-patience 5 \
    --amp-dtype   bf16 \
    --output-dir  ../checkpoints_new
```

**Phase C — image-level head:**

```bash
python train_image_level_head.py \
    --backbone-checkpoint ../checkpoints_new/best_model.pth \
    --tgif-root           /path/to/TGIF \
    --flux-root           /path/to/TGIF2_FLUX \
    --output-dir          ../checkpoints_new
```

---

## Datasets

The package does **not** redistribute the training images. They are obtained directly from the respective authors:

| Dataset      | Used for                  | Source / link                                                          |
|--------------|---------------------------|------------------------------------------------------------------------|
| TGIF         | Training + validation      | Diffusion-tampered image set; see TGIF paper.                          |
| TGIF2-FLUX   | Training + validation      | FLUX-tampered image set, paired with TGIF.                              |
| DF2023       | Cross-dataset evaluation   | DF2023 V1.5 validation split (5,000 manipulated images).                |
| COCO val2017 | Authentic control          | MS-COCO val2017, 5,000 authentic photographs.                          |

Place each under any path of your choice and pass them via the `--*-root` flags above.

---

## Architecture

A reduced overview is shown below; the canonical diagram is `figures/Fig 2.tif`.

```
Input image X (3 × 1024 × 1024)
        │
        ├─► SemanticStem (Swin + ConvNeXt)  ──► F_stem   (256 × 64 × 64)
        │
        ├─► LTIModule (frozen SD1.5 VAE+UNet, DDIM)
        │       trajectory error at t ∈ {1, 2, 3}
        │       ──► F_latent (256 × 64 × 64)
        │
        └─► APAFModule (2-D Hanning ▸ FFT ▸ amplitude gate ▸ iFFT)
                ──► F_freq  (256 × 64 × 64)
                                │
        DSCFModule (bidirectional cross-attention)
                ──► F_synergy (256 × 64 × 64)
                                │
        PredictionHead (4× upsample)
                ──► Y_hat (1 × 1024 × 1024 sigmoid mask)
                ──► A     (1 × 1024 × 1024 spatial-softmax attention)

         (optional) ImageLevelHead ──► image-level logit
```

All component modules are defined in `code/hlfcnet_model.py`. The frozen-SD branch is offloaded to CPU after the LTI cache is built; only the trainable LTI projection runs on GPU during training.

---

## Reproducing the published results

The contents of each subdirectory under `results/` correspond directly to the numbers, curves and qualitative figures in the paper:

| Paper element                                | Folder under `results/`                |
|----------------------------------------------|----------------------------------------|
| Headline TGIF / TGIF2-FLUX metrics, Table 1  | `tgif_val_eval/metrics_summary.json`   |
| Per-subset IoU and MCC, Table 2              | `tgif_val_eval/metrics_summary.json` (`per_bucket`) |
| Training curves, Fig. 8                      | `tgif_val_eval/mentor_plots/01_training_curves.{pdf,png}` |
| ROC and PR curves                            | `tgif_val_eval/{roc_curve,pr_curve}.{pdf,png}` |
| DF2023 cross-dataset, Table 3                | `df2023_cross_eval/metrics_summary.json` |
| DF2023 per-manipulation-type bars, Fig. 9    | `df2023_cross_eval/per_type_bars.{pdf,png}` |
| Cross-dataset score inversion, Fig. 10       | `df2023_with_authentic_eval/score_distributions.{pdf,png}` |
| Qualitative grids (TGIF/FLUX), Fig. 7        | `tgif_val_eval/viz_grids/grid*.png`     |
| Qualitative grids (DF2023), Fig. 7a          | `df2023_viz_grids/fig7_*.png`           |
| Image-level head evaluation, Table 4         | `image_head_tgif_val_eval/`, `image_head_full_eval/` |

The raw per-sample CSVs (`per_sample_metrics.csv`) and prediction arrays (`predictions.npz`) are included so that all aggregate numbers can be independently recomputed.

---

## Limitations and known caveats

- **Cross-dataset image-level scoring** — the paper documents (Section 4.4.3) that a max-pooled image-level scorer trained on TGIF-style content produces inverted score distributions on DF2023 when an authentic real-photograph control is added. The `ImageLevelHead` partially mitigates this, but not entirely.
- **Frozen Stable Diffusion 1.5** — the LTI module depends on `runwayml/stable-diffusion-v1-5`. Downstream users are bound by that model's license.
- **GPU memory** — training requires a 24 GB GPU. Lower-memory devices can run inference and evaluation but not training at the published batch size.

---

## License

Released under the MIT License — see `LICENSE`. The Stable Diffusion 1.5 weights loaded by the LTI module are governed by their own license, which applies independently.

---

## Citation

If you use this code or the released checkpoint in your research, please cite the accompanying paper. A machine-readable citation is provided in `CITATION.cff`.

```bibtex
@article{hlfcnet2026,
  title   = {HLFC-Net: A hierarchical latent--frequency cross-attention network for diffusion-based image manipulation detection},
  author  = {Srilakshmi V, Dr. S. Ravichandra},
  journal = {<Journal>},
  year    = {2026}
}
```

---

## Contact

For questions or bug reports, please open an issue on the repository or contact the corresponding author listed in the paper.
