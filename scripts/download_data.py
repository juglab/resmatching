"""Download BioSR datasets used in ResMatching.

Usage:
    uv run python scripts/download_data.py               # download all subsets
    uv run python scripts/download_data.py --data-dir /my/path
    uv run python scripts/download_data.py --subset ccp --subset er
"""

import zipfile
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import pooch
import typer

BASE_URL = "https://download.fht.org/jug/resmatching/data/"


DATASETS = {
    "ccp": (
        "ccp.zip",
        "13eff82a3cdd155b4c3b553e067c2d58bc2d8218918532efe88c14ae5d4dbb56",
    ),
    "er": (
        "er.zip",
        "959a930728b131d1783ba60b39a7ed71b1d2cfc873bdaa911a0466e11f985d9d",
    ),
    "factin": (
        "factin.zip",
        "4401482869d4f6af741c1880e8b9c63d7182172cf0b0cae5ff0f882f01bdf641",
    ),
    "mt": (
        "mt.zip",
        "b164261bd2b737cb7f1b72d585c20ade56d8e18ffc8bbb8df5bb0753b8f72d4a",
    ),
    "mt_noisy": (
        "mt_noisy.zip",
        "1006261fded568f683a1f4da2639ab8e6a630e164234575b7dfa472aafaf9222",
    ),
}

DESCRIPTIONS = {
    "ccp": "Clathrin-Coated Pits",
    "er": "Endoplasmic Reticulum",
    "factin": "F-actin",
    "mt": "Microtubules",
    "mt_noisy": "Microtubules (noisy input)",
}

Subset = Enum("Subset", {k: k for k in DATASETS})

app = typer.Typer()


def _download_subset(key: str, data_dir: Path) -> None:
    filename, known_hash = DATASETS[key]
    typer.echo(f"Downloading {DESCRIPTIONS[key]} ({filename}) ...")

    path = pooch.retrieve(
        url=BASE_URL + filename,
        known_hash=known_hash,
        fname=filename,
        path=data_dir,
        progressbar=True,
    )

    typer.echo(f"  Extracting to {data_dir} ...")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(data_dir)
    Path(path).unlink()
    typer.echo(f"  Done -> {data_dir / filename.replace('.zip', '')}")


@app.command()
def main(
    data_dir: Annotated[
        Path, typer.Option(help="Directory to download data into.")
    ] = Path("data"),
    subset: Annotated[
        Optional[list[Subset]],
        typer.Option(
            help="Subset(s) to download. Repeat to select multiple. Default: all."
        ),
    ] = None,
):
    keys = [s.value for s in subset] if subset else list(DATASETS)
    data_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        _download_subset(key, data_dir)

    typer.echo("\nAll done. Data layout:")
    for key in keys:
        typer.echo(f"  {data_dir / key}/")


if __name__ == "__main__":
    app()
