import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

from utils.download import BAND_ORDER, download_year

# =========================================
# CONFIG
# =========================================
app = typer.Typer()


@app.command()
def main(
    years_start: int = typer.Option(
        2024, "--years-start", "-ys", help="Start year for download"
    ),
    years_end: int = typer.Option(
        2025, "--years-end", "-ye", help="End year for download"
    ),
    month_start: str = typer.Option(
        "08-01", "--month-start", "-ms", help="Month start date (MM-DD)"
    ),
    month_end: str = typer.Option(
        "08-31", "--month-end", "-me", help="Month end date (MM-DD)"
    ),
    grid: str = typer.Option("MGRS-05WMU", "--grid", "-g", help="MGRS grid identifier"),
    max_cloud_cover: int = typer.Option(
        70, "--max-cloud-cover", "-cc", help="Maximum cloud cover percentage"
    ),
    bbox_west: float = typer.Option(
        -153.5, "--bbox-west", "-bw", help="Bounding box west coordinate"
    ),
    bbox_south: float = typer.Option(
        70.5, "--bbox-south", "-bs", help="Bounding box south coordinate"
    ),
    bbox_east: float = typer.Option(
        -153.0, "--bbox-east", "-be", help="Bounding box east coordinate"
    ),
    bbox_north: float = typer.Option(
        71.0, "--bbox-north", "-bn", help="Bounding box north coordinate"
    ),
    out_dir: Path = typer.Option(
        Path("data/coverage70/scenes_raw"),
        "--output-dir",
        "-o",
        help="Output directory",
    ),
    aws_access_key: str = typer.Option(None, "--aws-access-key", help="AWS access key"),
    aws_secret_key: str = typer.Option(None, "--aws-secret-key", help="AWS secret key"),
    aws_region: str = typer.Option(
        "eu-central-1", "--aws-region", "-ar", help="AWS region"
    ),
    aws_endpoint: str = typer.Option(
        "eodata.dataspace.copernicus.eu",
        "--aws-endpoint",
        "-ae",
        help="AWS S3 endpoint",
    ),
    n_parallel: int = typer.Option(
        1, "--n-parallel", "-np", help="Number of parallel downloads"
    ),
):
    """Download Sentinel-2 scenes with configurable parameters."""

    YEARS = list(range(years_start, years_end + 1))
    MONTH_START_END = (month_start, month_end)
    BBOX_LL = (bbox_west, bbox_south, bbox_east, bbox_north)

    # determine AWS credentials from args or environment
    provided_access = aws_access_key or os.environ.get("AWS_ACCESS_KEY_ID")
    provided_secret = aws_secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not (provided_access and provided_secret):
        typer.secho(
            "AWS credentials not provided. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "in the environment or pass --aws-access-key and --aws-secret-key.",
            fg="red",
            err=True,
        )
        raise typer.Exit(code=1)

    if aws_access_key is not None:
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
    elif "AWS_ACCESS_KEY_ID" not in os.environ:
        typer.secho(
            "AWS credentials not provided. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "in the environment or pass --aws-access-key and --aws-secret-key.",
            fg="red",
            err=True,
        )
        raise typer.Exit(code=1)

    if aws_secret_key is not None:
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
    elif "AWS_SECRET_ACCESS_KEY" not in os.environ:
        typer.secho(
            "AWS credentials not provided. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "in the environment or pass --aws-access-key and --aws-secret-key.",
            fg="red",
            err=True,
        )
        raise typer.Exit(code=1)
    os.environ["AWS_REGION"] = aws_region
    os.environ["AWS_S3_ENDPOINT"] = aws_endpoint
    os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"

    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================
    # RUN ALL YEARS IN PARALLEL
    # =========================================
    def download_year_wrapper(yr):
        """Wrapper function for downloading a single year."""
        n = download_year(
            yr,
            grid,
            max_cloud_cover,
            bbox_ll=BBOX_LL,
            band_order=BAND_ORDER,
            out_dir=out_dir,
            month_start_end=MONTH_START_END,
        )
        return yr, n

    all_counts = {}
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {executor.submit(download_year_wrapper, yr): yr for yr in YEARS}

        for future in as_completed(futures):
            yr, n = future.result()
            all_counts[yr] = n
            typer.echo(f"✅ Year {yr}: {n} scenes downloaded")

    typer.echo("\n✅ Done. Scenes written per year:")
    for yr in sorted(all_counts.keys()):
        typer.echo(f" • {yr}: {all_counts[yr]} scenes")


if __name__ == "__main__":
    app()
