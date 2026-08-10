# ResMatching

Official implementation of **ResMatching**, accepted at [ISBI 2026](https://biomedicalimaging.org/2026/).

> **ResMatching: Noise-Resilient Computational Super-Resolution via Guided Conditional Flow Matching**  
> Anirban Ray, Vera Galinova, Florian Jug
> [[arXiv]](https://arxiv.org/abs/2510.26601) | [[Interactive Results]](https://rayanirban.github.io/resmatching/)
> 
![Figure 1](assets/figure1.png)

## Abstract

> Computational Super-Resolution (CSR) in fluorescence microscopy has, despite being an ill-posed problem, a long history. At its very core, CSR is about finding a prior that can be used to extrapolate frequencies in a micrograph that have never been imaged by the image-generating microscope. It stands to reason that, with the advent of better data-driven machine learning techniques, stronger prior can be learned and hence CSR can lead to better results. Here, we present **ResMatching**, a novel CSR method that uses guided conditional flow matching to learn such improved data-priors. We evaluate ResMatching on 4 diverse biological structures from the BioSR dataset and compare its results against 7 baselines. ResMatching consistently achieves competitive results, demonstrating in all cases the best trade-off between data fidelity and perceptual realism. We observe that CSR using ResMatching is particularly effective in cases where a strong prior is hard to learn, e.g. when the given low-resolution images contain a lot of noise. Additionally, we show that ResMatching can be used to sample from an implicitly learned posterior distribution and that this distribution is calibrated for all tested use-cases, enabling our method to deliver a pixel-wise data-uncertainty term that can guide future users to reject uncertain predictions.

## Installation

```bash
pip install uv
uv sync
```

## Example Notebook

**Start here for the quickest hands-on introduction:** [notebooks/resmatching_walkthrough.ipynb](notebooks/resmatching_walkthrough.ipynb)

This beginner-friendly notebook is aimed at biologists and microscopists and is the easiest way to get a feel for the full ResMatching workflow.

The notebook explains each cell in detail, includes a tiny 2 to 3 epoch training demo, runs small-sample inference with a pre-trained model, computes lightweight metrics, plots calibration, and visualizes the input / ground truth / MMSE / posterior samples.

## Datasets

Experiments use the [BioSR](https://figshare.com/articles/dataset/BioSR/13264793/9) dataset. The following subsets are supported:

| Subset | Structure |
|--------|-----------|
| `ccp` | Clathrin-Coated Pits |
| `er` | Endoplasmic Reticulum |
| `factin` | F-actin |
| `mt` | Microtubules |
| `mt_noisy` | Microtubules data with additional noise added |

## Reproducing results

There are two workflows depending on whether you want to train from scratch or use pre-trained checkpoints.

> By default, `scripts/metrics.py` evaluates the current local inference outputs in `data/<subset>/<split>_results/`. Use `--paper-results` to download and evaluate the exact saved prediction stacks used for the paper metrics from Zenodo. Full retraining may produce slight variance due to non-deterministic operations.

---

### Option A — Train from scratch

**Step 1. Download data**

```bash
# Download all subsets
uv run python scripts/download_data.py

# Or download a specific subset
uv run python scripts/download_data.py --subset ccp
```

Data is saved to `data/<subset>/` by default.

**Step 2. Train**

```bash
uv run python scripts/train.py ccp
```

The best checkpoint is saved to `checkpoints/ccp/best_model.pth`. Training runs for 200 epochs by default.

**Step 3. Run inference**

```bash
uv run python scripts/infer.py ccp --checkpoint checkpoints/ccp/best_model.pth
```

Writes multi-sample TIFFs to `data/ccp/test_results/` and `data/ccp/val_results/`.

**Step 4. Compute metrics**

```bash
# Evaluate the current local inference outputs
uv run python scripts/metrics.py ccp

# Or evaluate the exact paper result stacks from Zenodo
uv run python scripts/metrics.py ccp --paper-results
```

The default command reads from `data/ccp/test_results/`. With `--paper-results`, the script downloads the archive to `data/ccp/paper_result_samples/`, reads `test_result_samples/`, and prints PSNR, MicroMS3IM, LPIPS, and FID. Add `--split val` to evaluate the validation stacks. The Zenodo archive for `mt_noisy` is named `mtNoisy`, which the script handles automatically.

**Step 5. (Optional) Calibration**

```bash
uv run python scripts/calibrate.py ccp --results-dir data/ccp
```

Reads `val_results/` and `test_results/` under `data/ccp/` and saves a calibration curve to `data/ccp/calibration.pdf`.

---

### Option B — Use pre-trained checkpoints

**Step 1. Download data**

```bash
uv run python scripts/download_data.py --subset ccp
```

**Step 2. Download pre-trained checkpoints**

```bash
# Download all checkpoints
uv run python scripts/download_models.py

# Or download a specific checkpoint
uv run python scripts/download_models.py --subset ccp
```

Checkpoints are saved to `checkpoints/<subset>/best_model.pth` by default.

**Step 3. Run inference**

```bash
uv run python scripts/infer.py ccp --checkpoint checkpoints/ccp/best_model.pth
```

**Step 4. Compute metrics**

```bash
# Evaluate the current local inference outputs
uv run python scripts/metrics.py ccp

# Or evaluate the exact paper result stacks from Zenodo
uv run python scripts/metrics.py ccp --paper-results
```

**Step 5. (Optional) Calibration**

```bash
uv run python scripts/calibrate.py ccp --results-dir data/ccp
```

## ⚠️ Manual Download of Datasets and Pre-trained Models

If downloading the datasets or pretrained models using the provided script results in an error, you can download them manually from the following link:

**Datasets and pretrained models:** [Zenodo download link](https://zenodo.org/records/21876266)

After downloading, extract the files and place them in the appropriate dataset and model directories used by ResMatching.

### Model file names

The downloaded best-model checkpoints may include the dataset name as a prefix. For example:

```text
ccp_best_model.pth
```

Before placing a checkpoint in its corresponding model directory, remove the dataset-name prefix so that the file is named:

```text
best_model.pth
```

For example:

```text
ccp_best_model.pth  →  best_model.pth
```

Place each renamed checkpoint inside the directory corresponding to that dataset. Do not place all renamed `best_model.pth` files in the same directory, as they would overwrite one another.

Similarly, place each manually downloaded dataset in the appropriate dataset directory expected by the project.

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{resmatching2025,
  title={ResMatching: Noise-Resilient Computational Super-Resolution via Guided Conditional Flow Matching},
  author={Anirban Ray and Vera Galinova and Florian Jug},
  journal={2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  year={2026}
}
```

## License

MIT
