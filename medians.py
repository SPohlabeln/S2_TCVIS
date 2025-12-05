from dask.distributed import Client, LocalCluster
import dask
from pathlib import Path
import rioxarray
import xarray as xr
from utils.spatial.raster import da_to_uint16, raster_kwargs, create_median_mosaic, save_to_to_tif
from utils.mask import BAND_LABELS
import typer


app = typer.Typer(help="Create median mosaics for Sentinel-2 scenes")


@app.command()
def main(
    years_start: int = typer.Option(
        2017, "--year-start", "-ys", help="Start year for processing"
    ),
    years_end: int = typer.Option(
        2025, "--year-end", "-ye", help="End year for processing"
    ),
    tif_dir: Path = typer.Option(
        Path("data/coverage70/scenes_masked"),
        "--tif-dir",
        "-i",
        help="Base directory with masked scene TIFFs",
    ),
    out_dir: Path = typer.Option(
        Path("data/coverage70/medians"),
        "--output-dir",
        "-o",
        help="Output directory for median mosaics",
    ),
    chunksize: int = typer.Option(
        512, "--dask-chunksize", "-dcs", help="Chunk size for x/y dimensions"
    ),
    n_workers: int = typer.Option(
        8, "--dask-n-workers", "-dnw", help="Number of Dask workers"
    ),
    threads_per_worker: int = typer.Option(
        1, "--dask-threads-per-worker", "-dtpw", help="Threads per worker"
    ),
    memory_limit: str = typer.Option(
        "12GB", "--dask-memory-limit", "-dml", help="Memory limit per worker"
    ),
):
    """Create median mosaics for Sentinel-2 scenes across multiple years."""

    typer.echo(f"📍 Input directory: {tif_dir}")
    typer.echo(f"📍 Output directory: {out_dir}")
    typer.echo(f"📅 Years: {years_start} to {years_end}")

    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Process all years
    YEARS = list(range(years_start, years_end + 1))
    results = {}

    for year in YEARS:
        median_u16 = create_median_mosaic(
            year=year,
            tif_dir=tif_dir,
            out_dir=out_dir,
            client=client,
            chunksize=chunksize,
        )
        if median_u16 is not None:
            save_to_to_tif(year, median_u16, out_dir)
            results[year] = "✅"
        else:
            results[year] = "⏭️/❌"

    # Print summary
    typer.echo("\n" + "=" * 60)
    typer.secho("📊 SUMMARY", bold=True)
    typer.echo("=" * 60)
    for yr in sorted(results.keys()):
        typer.echo(f"{results[yr]} {yr}")

    typer.echo("\n✅ Done. Closing Dask client/cluster.")
    client.close()
    cluster.close()
    
if __name__ == "__main__":
    app()