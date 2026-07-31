import os
import zipfile
import warnings
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pooch
import torch
from microssim import MicroMS3IM
from tifffile import imread, TiffFile
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm
import typer

from resmatching.ra_psnr import RangeInvariantPsnr
from resmatching.utils import (
    lpips,
    fid_score,
    FSIM,
    extract_patches_inner_metrics,
    GMSD,
    entropy,
)

warnings.filterwarnings("ignore")

SUBSETS = ["ccp", "er", "factin", "mt", "mt_noisy"]
SPLITS = ["test", "val"]
PAPER_RESULTS_URL = (
    "https://zenodo.org/records/21721986/files/"
    "{zenodo_subset}_test_val_result_samples.zip?download=1"
)
PAPER_RESULTS_SUBSET_NAMES = {"mt_noisy": "mtNoisy"}

app = typer.Typer()


def _has_tifs(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.tif"))


def _safe_extract(zip_file: zipfile.ZipFile, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    for member in zip_file.infolist():
        target = (output_dir / member.filename).resolve()
        if output_root != target and output_root not in target.parents:
            raise ValueError(f"Unsafe zip member path: {member.filename}")
    zip_file.extractall(output_dir)


def _download_paper_result_samples(
    subset: str,
    data_dir: Path,
    paper_results_dir: Optional[Path],
) -> Path:
    output_dir = paper_results_dir or data_dir / subset / "paper_result_samples"
    test_dir = output_dir / "test_result_samples"
    val_dir = output_dir / "val_result_samples"

    if _has_tifs(test_dir) and _has_tifs(val_dir):
        typer.echo(f"Using cached paper result samples: {output_dir}")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    zenodo_subset = PAPER_RESULTS_SUBSET_NAMES.get(subset, subset)
    filename = f"{zenodo_subset}_test_val_result_samples.zip"
    typer.echo(f"Downloading paper result samples from Zenodo ({filename})...")
    archive_path = pooch.retrieve(
        url=PAPER_RESULTS_URL.format(zenodo_subset=zenodo_subset),
        known_hash=None,
        fname=filename,
        path=output_dir,
        progressbar=True,
    )

    typer.echo(f"Extracting paper result samples to {output_dir}...")
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        _safe_extract(zip_file, output_dir)

    missing = [
        folder.name
        for folder in (test_dir, val_dir)
        if not _has_tifs(folder)
    ]
    if missing:
        typer.echo(
            f"Error: downloaded archive did not create expected folder(s): {missing}",
            err=True,
        )
        raise typer.Exit(1)

    return output_dir


def _prediction_samples_from_stack(
    stack: np.ndarray, n_samples: int, image_path: Path
) -> np.ndarray:
    if stack.ndim < 3:
        raise ValueError(f"{image_path} must contain sample stacks, got {stack.shape}.")
    if stack.shape[0] < n_samples:
        raise ValueError(
            f"{image_path} contains {stack.shape[0]} samples, "
            f"but n_samples={n_samples} was requested."
        )

    samples = stack[:n_samples]
    while samples.ndim > 3:
        if samples.shape[1] == 1:
            samples = np.squeeze(samples, axis=1)
        else:
            samples = samples[:, -1]

    if samples.ndim != 3:
        raise ValueError(
            f"{image_path} must reduce to (samples, H, W), got {samples.shape}."
        )
    return samples.astype("float32", copy=False)


@app.command()
def compute_metrics(
    subset: Annotated[str, typer.Argument(help=f"Dataset subset. One of: {SUBSETS}")],
    results_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory containing inference .tif results. Defaults to <data_dir>/<subset>/<split>_results/."
        ),
    ] = None,
    fid_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory of FID reference crops. Defaults to <data_dir>/<subset>/train_crops_fid_filtered/."
        ),
    ] = None,
    data_dir: Annotated[
        Path, typer.Option(help="Root data directory (used to resolve defaults).")
    ] = Path("data"),
    split: Annotated[
        str, typer.Option(help=f"Dataset split to evaluate. One of: {SPLITS}.")
    ] = "test",
    n_samples: Annotated[
        int, typer.Option(help="Number of samples to average for MMSE prediction.")
    ] = 50,
    paper_results: Annotated[
        bool,
        typer.Option(
            "--paper-results",
            help=(
                "Download/use the exact Zenodo result sample stacks used for the "
                "paper metrics."
            ),
        ),
    ] = False,
    paper_results_dir: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "Where Zenodo paper result samples are stored. Defaults to "
                "<data_dir>/<subset>/paper_result_samples/."
            )
        ),
    ] = None,
):
    if subset not in SUBSETS:
        typer.echo(f"Error: subset must be one of {SUBSETS}", err=True)
        raise typer.Exit(1)
    if split not in SPLITS:
        typer.echo(f"Error: split must be one of {SPLITS}", err=True)
        raise typer.Exit(1)
    if n_samples < 1:
        typer.echo("Error: n_samples must be at least 1", err=True)
        raise typer.Exit(1)
    if paper_results_dir is not None and not paper_results:
        typer.echo("Error: --paper-results-dir requires --paper-results", err=True)
        raise typer.Exit(1)

    subset_dir = data_dir / subset
    if paper_results:
        paper_root = _download_paper_result_samples(
            subset=subset,
            data_dir=data_dir,
            paper_results_dir=paper_results_dir,
        )
        results_dir = paper_root / f"{split}_result_samples"
    elif results_dir is None:
        results_dir = subset_dir / f"{split}_results"
    if fid_dir is None:
        fid_dir = subset_dir / "train_crops_fid_filtered"
    gt_dir = subset_dir / split

    for label, path in (
        ("results", results_dir),
        ("FID reference crops", fid_dir),
        ("ground truth", gt_dir),
    ):
        if not path.is_dir():
            typer.echo(f"Error: {label} directory not found: {path}", err=True)
            raise typer.Exit(1)

    micros_ms3im = MicroMS3IM()

    # ── Load FID reference crops ─────────────────────────────────────────────
    fid_files = sorted(f for f in os.listdir(fid_dir) if f.endswith(".tif"))
    fid_crops = []
    for fid_file in tqdm(fid_files, desc="Loading FID crops", leave=False):
        with TiffFile(fid_dir / fid_file) as tif:
            fid_crops.append(tif.asarray())
    if not fid_crops:
        typer.echo(f"Error: no .tif files found in {fid_dir}", err=True)
        raise typer.Exit(1)
    fid_crops = np.concatenate(fid_crops, axis=0)
    fid_crops_gts = torch.from_numpy(fid_crops).unsqueeze(1)
    typer.echo(f"Using {fid_crops.shape[0]} crops for FID.")

    image_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".tif"))
    if not image_files:
        typer.echo(f"Error: no .tif files found in {results_dir}", err=True)
        raise typer.Exit(1)

    psnr_values, ms_ssim_scores, micro3_ssim_scores = [], [], []
    gts, outputs, gts_full, outputs_full = [], [], [], []
    ind_fsims, ind_lpips, ind_fids, ind_gmsd = [], [], [], []

    typer.echo(
        f"Computing metrics over {len(image_files)} images (MMSE n={n_samples})..."
    )

    for image_file in tqdm(image_files, desc="Images", leave=False):
        with TiffFile(results_dir / image_file) as tif:
            image = tif.asarray()

        try:
            image_pred = _prediction_samples_from_stack(
                image, n_samples=n_samples, image_path=results_dir / image_file
            )
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        gt_path = gt_dir / image_file
        if not gt_path.is_file():
            typer.echo(f"Error: ground-truth file not found: {gt_path}", err=True)
            raise typer.Exit(1)
        image_gt = imread(gt_dir / image_file).astype("float32")[0:1]  # (1, H, W)
        mmse_pred = np.mean(image_pred, axis=0, keepdims=True)

        # PSNR + MS-SSIM
        psnr_values.append(RangeInvariantPsnr(image_gt, mmse_pred))
        ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
            kernel_size=3, data_range=1.0, betas=(0.0448, 0.2856, 0.3001)
        )
        ms_ssim_scores.append(
            ms_ssim_metric(
                torch.from_numpy(mmse_pred).unsqueeze(0),
                torch.from_numpy(image_gt).unsqueeze(0),
            )
        )

        mmse_patches, _ = extract_patches_inner_metrics(mmse_pred, 64)
        gt_patches, _ = extract_patches_inner_metrics(image_gt, 64)
        gts.append(torch.from_numpy(gt_patches))
        outputs.append(torch.from_numpy(mmse_patches))
        gts_full.append(torch.from_numpy(image_gt).unsqueeze(1))
        outputs_full.append(torch.from_numpy(mmse_pred).unsqueeze(1))

        # Per-sample perceptual metrics
        torch_gt_patches = torch.from_numpy(gt_patches)
        valid = [
            i for i in range(torch_gt_patches.shape[0]) if torch_gt_patches[i].max() > 0
        ]
        torch_gt_patches = torch_gt_patches[valid]

        image_fsims, image_lpips_, image_fids_, image_gmsd_ = [], [], [], []
        for j in range(image_pred.shape[0]):
            pred_patches, _ = extract_patches_inner_metrics(image_pred[j : j + 1], 64)
            torch_pred = torch.from_numpy(pred_patches)[valid]
            image_fsims.append(FSIM(torch_pred, torch_gt_patches))
            image_lpips_.append(lpips(torch_gt_patches, torch_pred))
            image_fids_.append(fid_score(fid_crops_gts, torch_pred))
            image_gmsd_.append(GMSD(torch_pred, torch_gt_patches))

        if image_fsims:
            ind_fsims.append(torch.mean(torch.stack(image_fsims)))
            ind_lpips.append(torch.mean(torch.tensor(image_lpips_)))
            ind_fids.append(torch.mean(torch.tensor(image_fids_)))
            ind_gmsd.append(torch.mean(torch.stack(image_gmsd_)))

    # ── Aggregate ────────────────────────────────────────────────────────────
    gts = torch.cat(gts, dim=0)
    outputs = torch.cat(outputs, dim=0)
    gts_full = torch.cat(gts_full, dim=0)
    outputs_full = torch.cat(outputs_full, dim=0)

    average_psnr = sum(psnr_values) / len(psnr_values)
    std_psnr = torch.std(torch.stack(psnr_values))
    average_ms_ssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
    std_ms_ssim = torch.std(torch.stack(ms_ssim_scores))

    fsim_scores = FSIM(outputs, gts)
    fsim_mean = torch.mean(fsim_scores)
    lpips_score = lpips(gts, outputs)
    fid = fid_score(fid_crops_gts, outputs)
    gmsd_scores = GMSD(outputs, gts)
    entropy_scores = entropy(outputs)

    average_ind_fsim = torch.mean(torch.stack(ind_fsims))
    std_ind_fsim = torch.std(torch.stack(ind_fsims))
    average_ind_lpips = torch.mean(torch.tensor(ind_lpips))
    std_ind_lpips = torch.std(torch.tensor(ind_lpips))
    average_ind_fid = torch.mean(torch.tensor(ind_fids))
    std_ind_fid = torch.std(torch.tensor(ind_fids))
    average_ind_gmsd = torch.mean(torch.stack(ind_gmsd))
    std_ind_gmsd = torch.std(torch.stack(ind_gmsd))

    # MicroMS3IM
    gts_np = gts_full.numpy()
    outs_np = outputs_full.numpy()
    micros_ms3im.fit(gts_np[:, 0], outs_np[:, 0])
    micro3_ssim_scores = [
        micros_ms3im.score(gts_np[i, 0], outs_np[i, 0], betas=(0.0448, 0.2856, 0.3001))
        for i in range(gts_np.shape[0])
    ]
    average_micro3_ssim = np.mean(micro3_ssim_scores)
    std_micro3_ssim = np.std(micro3_ssim_scores)

    # ── Print ────────────────────────────────────────────────────────────────
    source = "paper results" if paper_results else "current results"
    typer.echo(f"\n=== {subset.upper()} {split} ({source}, n={n_samples}) ===")
    typer.echo(f"PSNR:         {average_psnr.item():.4f} ± {std_psnr.item():.4f}")
    typer.echo(f"MicroMS3IM:   {average_micro3_ssim:.4f} ± {std_micro3_ssim:.4f}")
    typer.echo(f"LPIPS (MMSE): {lpips_score:.4f}")
    typer.echo(
        f"LPIPS (Ind):  {average_ind_lpips.item():.4f} ± {std_ind_lpips.item():.4f}"
    )
    typer.echo(f"FID   (MMSE): {fid:.4f}")
    typer.echo(f"FID   (Ind):  {average_ind_fid.item():.4f} ± {std_ind_fid.item():.4f}")

    # LaTeX rows
    name = "\\textbf{ResMatching}"
    typer.echo("\n--- LaTeX (MMSE + Ind, supplemental) ---")
    typer.echo(
        f"& {name} & "
        f"\\makecell{{{average_psnr.item():.2f} \\\\ {std_psnr.item():.3f}}} & "
        f"\\makecell{{{average_micro3_ssim:.3f} \\\\ {std_micro3_ssim:.4f}}} & "
        f"\\makecell{{{lpips_score:.3f}}} & "
        f"\\makecell{{{fid:.3f}}} & "
        f"\\makecell{{{average_ind_lpips.item():.3f} \\\\ {std_ind_lpips.item():.4f}}} & "
        f"\\makecell{{{average_ind_fid.item():.3f} \\\\ {std_ind_fid.item():.4f}}} \\\\ \\cline{{2-7}}"
    )


if __name__ == "__main__":
    app()
