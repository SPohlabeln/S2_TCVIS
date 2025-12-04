from pathlib import Path

import typer
from tqdm import tqdm

from utils.mask import mask_scene

app = typer.Typer(help="Apply cloud masks to Sentinel-2 scenes")


@app.command()
def main(
    input_dir: Path = typer.Option(
        Path("data/coverage70/scenes_raw/2025"),
        "--input-dir",
        "-i",
        help="Directory with raw scene .tif files",
    ),
    output_dir: Path = typer.Option(
        Path("data/coverage70/scenes_masked/2025"),
        "--output-dir",
        "-o",
        help="Directory to write masked scenes",
    ),
    pattern: str = typer.Option(
        "*.tif", "--pattern", "-p", help="Glob pattern for input files"
    ),
    device: str = typer.Option(
        "cuda:0", "--device", "-d", help="Device for OMC inference (e.g. cuda:0 or cpu)"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing masked files"
    ),
):
    """Mask scenes in INPUT_DIR and write results to OUTPUT_DIR."""
    if not input_dir.exists() or not input_dir.is_dir():
        typer.secho(f"Input directory does not exist: {input_dir}", fg="red", err=True)
        raise typer.Exit(code=1)

    files = sorted(list(input_dir.glob(pattern)))
    if not files:
        typer.secho(f"No files matching '{pattern}' in {input_dir}", fg="yellow")
        raise typer.Exit(code=0)

    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failures = 0

    for in_file in tqdm(files, desc="Masking scenes", unit="file"):
        out_file = output_dir / in_file.name
        if out_file.exists() and not overwrite:
            typer.echo(f"Skipping (exists): {out_file}")
            continue

        try:
            typer.echo(f"Masking {in_file} -> {out_file}")
            mask_scene(
                input_image=in_file,
                output_image=out_file,
                omc_kwargs=dict(inference_device=device, mosaic_device=device),
            )
            success += 1
        except Exception as exc:
            typer.secho(f"Failed: {in_file} ({exc})", fg="red", err=True)
            failures += 1

    typer.secho(
        f"Done. Success: {success}, Failures: {failures}",
        fg="green" if failures == 0 else "yellow",
    )


if __name__ == "__main__":
    app()
