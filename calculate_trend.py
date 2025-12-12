# =========================================
# TREND CALCULATION FOR TC STACKS
# =========================================

import logging
from pathlib import Path

import dask
import numpy as np
import rioxarray
import typer
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from utils.spatial.trend import create_tc_stack

app = typer.Typer(help="Create Trend Data")


@app.command()
def main(
    input_dir: Path = typer.Option(
        Path("data/coverage70/tc"),
        "--input-dir",
        "-i",
        help="Base directory with masked scene TIFFs",
    ),
    output_file: Path = typer.Option(
        Path("data/coverage70/trends/s2_tc_trend_2017-2025.tif"),
        "--output-file",
        "-o",
        help="Output directory for median mosaics",
    ),
    chunksize: int = typer.Option(
        512, "--dask-chunksize", "-dcs", help="Chunk size for x/y dimensions"
    ),
    n_workers: int = typer.Option(8, "--dask-n-workers", "-dnw", help="Number of Dask workers"),
    threads_per_worker: int = typer.Option(
        1, "--dask-threads-per-worker", "-dtpw", help="Threads per worker"
    ),
    memory_limit: str = typer.Option(
        "12GB", "--dask-memory-limit", "-dml", help="Memory limit per worker"
    ),
):
    # -----------------------------------------
    # CONFIG
    # -----------------------------------------

    years = list(range(2017, 2026))
    bands_tc = ["TCB", "TCG", "TCW"]

    typer.echo(f"📍 Input directory: {input_dir}")
    typer.echo(f"📍 Output file: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Start Dask cluster
    typer.echo(f"\n🚀 Starting Dask cluster with {n_workers} workers...")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,
    )
    client = Client(cluster)
    typer.echo(client)

    # -----------------------------------------
    # 1. LOAD ALL TASSELED CAP MOSAICS
    # -----------------------------------------
    arrays = []

    for year in years:
        fp = input_dir / f"tc_median_{year}.tif"
        if not fp.exists():
            print(f"❌ Missing {fp}")
            continue

        print(f"✅ Loading {fp}")
        da = rioxarray.open_rasterio(fp, chunks={"x": chunksize, "y": chunksize})

        # Assign TC band names
        da = da.assign_coords(band=bands_tc)

        # Add numeric time coordinate
        da = da.expand_dims(time=[np.datetime64(f"{year}-07-15")])

        arrays.append(da)

    if not arrays:
        raise RuntimeError("No tasseled cap mosaics found!")

    # Concatenate stack
    stack = xr.concat(arrays, dim="time").transpose("time", "band", "y", "x")
    stack = stack.chunk({"time": -1, "x": chunksize, "y": chunksize})
    stack.name = "tc"

    print(f"🧩 Stack shape: {stack.shape} (time, band, y, x)")

    # -----------------------------------------
    # 2. FIX THE TIME AXIS FOR REGRESSION
    # -----------------------------------------
    # Convert datetime64 → integer years
    years_numeric = stack["time"].dt.year

    # Replace time dim with 'year'
    stack = stack.assign_coords(year=("time", years_numeric.data))
    stack = stack.swap_dims({"time": "year"})

    print(f"📅 Using year values for regression: {list(years_numeric.values)}")

    stack = create_tc_stack(input_dir, years, client=client, chunksize=chunksize, bands_tc=bands_tc)
    dask.distributed.wait(stack)

    # -----------------------------------------
    # 3. TREND REGRESSION (PER YEAR)
    # -----------------------------------------
    results = []

    for band in bands_tc:
        print(f"📈 Computing trend for {band}...")

        sub = stack.sel(band=band)

        # Fit a first-degree polynomial across the 'year' axis
        fit = sub.to_dataset(name="tc").polyfit(dim="year", deg=1)

        # Extract slope (degree 1 coefficient)
        slope = fit["tc_polyfit_coefficients"].sel(degree=1)

        # OPTIONAL —
        # match GEE visualization intensity (your GEE script did "*10")
        slope = slope * 10

        slope = slope.expand_dims(band=[f"{band}_slope"])
        results.append(slope)

    # Combine all slope bands
    trend = xr.concat(results, dim="band")
    trend.rio.write_crs(stack.rio.crs, inplace=True)

    # trend = client.persist(trend)
    # dask.distributed.wait(trend)

    # -----------------------------------------
    # 4. COMPUTE THE ARRAY
    # -----------------------------------------
    out_path = output_file
    print(f"💾 Saving trend raster: {out_path}")

    # Threaded Dask scheduler
    # dask.config.set(scheduler="threads")
    logging.getLogger("tornado.application").setLevel(logging.ERROR)
    logging.getLogger("tornado.general").setLevel(logging.ERROR)

    with ProgressBar(dt=30.0):
        trend = trend.compute()

    trend_vis = trend.clip(-0.3, 0.3)
    trend_vis = ((trend_vis + 0.3) / 0.6 * 255).astype("uint8")

    # -----------------------------------------
    # 5. SAVE TO GEOTIFF
    # -----------------------------------------
    trend_vis.transpose("band", "y", "x").rio.to_raster(
        out_path,
        driver="COG",
        # tiled=True,
        compress="deflate",
        BIGTIFF="IF_SAFER",
        # predictor=2,
        # blockxsize=1024,
        # blockysize=1024,
    )

    print("✅ Trend image saved successfully.")

    typer.echo("\n✅ Done. Closing Dask client/cluster.")
    client.close()
    cluster.close()


if __name__ == "__main__":
    app()
