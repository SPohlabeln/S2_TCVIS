import xarray as xr
from pathlib import Path

from dask.distributed import Client
import dask
from pathlib import Path
import rioxarray
import xarray as xr
from utils.mask import BAND_LABELS

import warnings
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
import typer
from dask.diagnostics import ProgressBar

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*coordinate precision.*"
)



def create_median_mosaic(
    year: int,
    tif_dir: Path | str,
    out_dir: Path | str,
    client: Client,
    chunksize: int = 512,
):
    """
    Create a median mosaic for a given year.

    Parameters
    ----------
    year : int
        Year to process
    tif_dir : Path | str
        Base directory containing masked scenes (e.g., "data/coverage70/scenes_masked")
    out_dir : Path | str
        Output directory for median mosaics
    client : Client
        Dask client for distributed computing
    chunksize : int, optional
        Chunk size for x/y dimensions (default 512)

    Returns
    -------
    xr.DataArray
        Median mosaic DataArray (uint16, persisted in cluster memory)
    """
    tif_dir = Path(tif_dir) / str(year)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"median_{year}.tif"

    if out_path.exists():
        print(f"⏭️ Year {year}: Output already exists, skipping {out_path}")
        return None

    if not tif_dir.exists():
        print(f"❌ Year {year}: Input directory does not exist: {tif_dir}")
        return None

    # Load TIFFs
    tif_files = sorted(list(tif_dir.glob("*.tif")))
    if not tif_files:
        print(f"⚠️ Year {year}: No TIFFs found in {tif_dir}")
        return None

    print(f"\n📅 Year {year} | 🗂 Found {len(tif_files)} TIFFs")

    scenes = []
    for f in tif_files:
        try:
            ds = rioxarray.open_rasterio(
                f, masked=True, chunks={"x": chunksize, "y": chunksize}
            )
            if "band" not in ds.coords:
                ds = ds.assign_coords(band=range(1, ds.sizes["band"] + 1))
            if len(BAND_LABELS) == ds.sizes["band"]:
                ds = ds.assign_coords(band=BAND_LABELS)
            scenes.append(ds.expand_dims(time=[f.name]))
            print(f"   ✓ Loaded (lazy) {f.name}")
        except Exception as e:
            print(f"   ⚠️ Failed to load {f.name}: {e}")

    if not scenes:
        print(f"❌ Year {year}: No scenes could be loaded.")
        return None

    # Concatenate (still lazy)
    stack = xr.concat(scenes, dim="time")
    print(f"📊 Stack shape: {stack.shape}")

    # Re-chunk for reasonable task size
    stack = stack.chunk({"time": -1, "x": chunksize, "y": chunksize})
    stack.name = "median_stack"

    # Persist stack to cluster memory
    stack = client.persist(stack)
    dask.distributed.wait(stack)
    print("🧠 Stack persisted to Dask cluster")

    # Compute median
    median_img = stack.median(dim="time", skipna=True)

    # Persist median result
    median_img = client.persist(median_img)
    dask.distributed.wait(median_img)
    print("✅ Median computed / persisted in cluster memory")

    # Convert to uint16
    median_img_u16 = da_to_uint16(median_img)

    print(f"✅ Year {year} median ready.\n")
    return median_img_u16

raster_kwargs = dict(
driver="GTiff",
compress="deflate",
tiled=True,
predictor=2,
BIGTIFF="IF_SAFER",
blockxsize=512,
blockysize=512,
)

def da_to_uint16(da: xr.DataArray) -> xr.DataArray:
    da_uint16 = (
        da
        .clip(0, 10000)
        .fillna(0)
        .astype("uint16")
        .rio.write_nodata(0)
    )
    
    return da_uint16

def save_da_to_tif(
    year: int,
    median_img_u16: xr.DataArray,
    out_dir: Path | str,
):
    """
    Save a median mosaic to GeoTIFF.

    Parameters
    ----------
    year : int
        Year (for naming)
    median_img_u16 : xr.DataArray
        Median mosaic DataArray (uint16)
    out_dir : Path | str
        Output directory for median mosaics
    """
    out_dir = Path(out_dir)
    out_path = out_dir / f"median_{year}.tif"

    print(f"💾 Saving median image to {out_path}")
    median_img_u16.rio.to_raster(out_path, **raster_kwargs)
    print(f"✅ Year {year} saved.\n")
    
    



def compute_tasseled_cap(da: xr.DataArray) -> xr.DataArray:
    """
    Compute tasseled-cap (TCB, TCG, TCW) from a 6-band DataArray.

    Input:
      - da: DataArray with bands [Blue, Green, Red, NIR, SWIR1, SWIR2] (or numeric band order).
            The function will assign band names, scale from 0..10000 to 0..1 and treat 0 as nodata.

    Output:
      - tc_stack: DataArray with band dimension ['TCB','TCG','TCW'] (float32), same CRS as input.
    """
    # Tasseled-cap coefficients for Sentinel-2
    coeffs = {
        "tcb": dict(
            Blue=0.3037, Green=0.2793, Red=0.4743, NIR=0.5585, SWIR1=0.5082, SWIR2=0.1863
        ),
        "tcg": dict(
            Blue=-0.2848,
            Green=-0.2435,
            Red=-0.5436,
            NIR=0.7243,
            SWIR1=0.0840,
            SWIR2=-0.1800,
        ),
        "tcw": dict(
            Blue=0.1509, Green=0.1973, Red=0.3279, NIR=0.3406, SWIR1=-0.7112, SWIR2=-0.4572
        ),
    }

    # assign band names if missing and scale reflectance
    try:
        da = da.assign_coords(band=["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])
    except Exception:
        # if band coord exists but names differ, assume correct order
        pass

    da = da.astype("float32") / 10000.0
    da = da.where(da != 0)  # treat zeros as nodata

    blue, green, red, nir, swir1, swir2 = da.sel(
        band=["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
    )

    def tc(c):
        return (
            c["Blue"] * blue
            + c["Green"] * green
            + c["Red"] * red
            + c["NIR"] * nir
            + c["SWIR1"] * swir1
            + c["SWIR2"] * swir2
        )

    tcb = tc(coeffs["tcb"])
    tcg = tc(coeffs["tcg"])
    tcw = tc(coeffs["tcw"])

    tc_stack = xr.concat([tcb, tcg, tcw], dim="band")
    tc_stack = tc_stack.assign_coords(band=["TCB", "TCG", "TCW"])
    tc_stack = tc_stack.rio.write_crs(da.rio.crs)

    return tc_stack

