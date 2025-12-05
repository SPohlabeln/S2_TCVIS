
import warnings
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
import typer
from dask.diagnostics import ProgressBar

from utils.spatial.raster import compute_tasseled_cap

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*coordinate precision.*"
)


app = typer.Typer(help="Compute tasseled-cap mosaics from yearly median mosaics")


@app.command()
def main(
    median_dir: Path = typer.Option(
        Path("data/coverage70/medians"),
        "--median-dir",
        "-m",
        help="Directory with yearly median mosaics (median_{year}.tif)",
    ),
    tc_dir: Path = typer.Option(
        Path("data/coverage70/tc"),
        "--tc-dir",
        "-o",
        help="Output directory for tasseled-cap mosaics",
    ),
    years_start: int = typer.Option(2017, "--years-start", "-ys", help="Start year"),
    years_end: int = typer.Option(2025, "--years-end", "-ye", help="End year"),
    scheduler: str = typer.Option(
        "threads",
        "--scheduler",
        "-s",
        help="Dask scheduler for compute ('threads' or 'processes')",
    ),
):
    """Compute tasseled-cap (TCB/TCG/TCW) mosaics from yearly median rasters."""
    typer.echo(f"📍 Median dir: {median_dir}")
    typer.echo(f"📍 Output dir: {tc_dir}")
    typer.echo(f"📅 Years: {years_start} .. {years_end}")
    tc_dir.mkdir(parents=True, exist_ok=True)

    # default chunking chosen internally (no CLI argument)
    chunks = {"x": 1024, "y": 1024}

    years = list(range(years_start, years_end + 1))


    for year in years:
        in_file = Path(median_dir) / f"median_{year}.tif"
        out_file = Path(tc_dir) / f"tc_median_{year}.tif"

        if not in_file.exists():
            typer.secho(f"❌ Missing median mosaic for {year}: {in_file}", fg="yellow")
            continue
        if out_file.exists():
            typer.echo(f"⏭️ Already exists, skipping {out_file}")
            continue

        typer.echo(f"✅ Loading: {in_file}")
        da = rioxarray.open_rasterio(in_file, chunks=chunks, masked=True)

        # compute tasseled-cap using the new helper
        tc_stack = compute_tasseled_cap(da)

        # Ensure NaNs are preserved as nodata and correct dtype
        tc_stack = tc_stack.astype("float32").rio.write_nodata(np.nan)

        typer.echo(f"💾 Saving tasseled cap mosaic: {out_file}")
        with ProgressBar():
            (
                tc_stack.compute(scheduler=scheduler)
                .transpose("band", "y", "x")
                .rio.to_raster(
                    out_file,
                    driver="GTiff",
                    tiled=True,
                    compress="deflate",
                    BIGTIFF="IF_SAFER",
                    predictor=3,
                    blockxsize=1024,
                    blockysize=1024,
                )
            )

    typer.secho("✅ All tasseled cap mosaics saved.", fg="green")


if __name__ == "__main__":
    app()