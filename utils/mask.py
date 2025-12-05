from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from omnicloudmask import predict_from_array


BAND_ORDER = ["B02_10m", "B03_10m", "B04_10m", "B08_10m", "B11_20m", "B12_20m"]
BAND_LABELS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]


def mask_scene(
    input_image: Path | str, output_image: Path | str, omc_kwargs: dict = None
):
    scene = rioxarray.open_rasterio(input_image)

    if "band" not in scene.coords:
        scene = scene.assign_coords(band=range(1, scene.sizes["band"] + 1))
    scene = scene.assign_coords(band=BAND_LABELS)
    red = scene.sel(band="Red").values
    green = scene.sel(band="Green").values
    nir = scene.sel(band="NIR").values
    input_array = np.stack([red, green, nir], axis=0)

    pred_mask = predict_from_array(input_array, **omc_kwargs)

    # Handle shape (1, H, W) or (3, H, W)
    if pred_mask.ndim == 3:
        if pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]
        elif pred_mask.shape[0] == 3:
            pred_mask = pred_mask[1]  # assume class 1 = cloud

    # Ensure mask shape matches (y, x)
    expected_shape = (scene.sizes["y"], scene.sizes["x"])
    if pred_mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {pred_mask.shape} does not match scene shape {expected_shape}"
        )

    # Keep only pixels where class == 0
    mask_keep = pred_mask == 0

    mask_da = xr.DataArray(
        mask_keep,
        dims=("y", "x"),
        coords={"y": scene.coords["y"], "x": scene.coords["x"]},
    )

    # return mask_da

    scene_masked = scene.where(mask_da)

    scene_u16 = (
        scene_masked.fillna(0).clip(0, 10000).astype("uint16").rio.write_nodata(0)
    )

    print("   💾 Saving →", output_image)
    scene_u16.transpose("band", "y", "x").rio.to_raster(
        output_image,
        driver="GTiff",
        compress="deflate",
        tiled=True,
        predictor=2,
        BIGTIFF="IF_SAFER",
        blockxsize=512,
        blockysize=512,
    )
    return 0
