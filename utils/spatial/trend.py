from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from dask.distributed import Client


def create_tc_stack(
    input_dir: Path,
    years: list,
    client: Client,
    chunksize: int = 512,
    bands_tc: list = ["TCB", "TCG", "TCW"],
) -> xr.DataArray:
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

    # Persist stack to cluster memory
    stack = client.persist(stack)
    # dask.distributed.wait(stack)
    print("🧠 Stack persisted to Dask cluster")
    print(f"📅 Using year values for regression: {list(years_numeric.values)}")

    return stack
