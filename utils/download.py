import math
import os
from pathlib import Path

import numpy as np
import rasterio as rio
import rioxarray
import xarray as xr
from omnicloudmask import predict_from_array
from pyproj import Transformer
from pystac import Item
from pystac_client import Client
from rasterio.enums import Resampling
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform


BAND_ORDER  = ["B02_10m","B03_10m","B04_10m","B08_10m","B11_20m","B12_20m"]
BAND_LABELS = ["Blue","Green","Red","NIR","SWIR1","SWIR2"]

def search_s2_stac(
    start_date: str, end_date: str, grid: str, max_cloud_cover: int = 100
) -> list[Item]:
    cat = Client.open("https://stac.dataspace.copernicus.eu/v1/")
    search = cat.search(
        collections=["sentinel-2-l2a"],
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lte": max_cloud_cover}, "grid:code": {"eq": grid}},
    )
    items = list(search.items())
    print(f"  🔎 Found {len(items)} items")
    return items


def prefer_s3_assets(items):
    out = []
    for it in items:
        it = it.clone()
        for a in it.assets.values():
            s3_href = None
            extra = getattr(a, "extra_fields", None) or {}
            alt = extra.get("alternate") or extra.get("alternates")
            if isinstance(alt, dict):
                s3_href = (alt.get("s3") or alt.get("S3") or {}).get("href")
            elif isinstance(alt, list):
                for d in alt:
                    href = d.get("href")
                    if href and href.startswith("s3://"):
                        s3_href = href
                        break
            if s3_href:
                a.href = s3_href
        out.append(it)
    return out


def detect_epsg_and_bounds(items, bbox_ll_override=None):
    if not items:
        raise ValueError("No items")

    if bbox_ll_override is None:
        bbs = [it.bbox for it in items]
        minx = min(b[0] for b in bbs)
        miny = min(b[1] for b in bbs)
        maxx = max(b[2] for b in bbs)
        maxy = max(b[3] for b in bbs)
        bbox_ll = (minx, miny, maxx, maxy)
    else:
        bbox_ll = bbox_ll_override

    epsg = None
    for it in items:
        if "proj:epsg" in it.properties:
            epsg = int(it.properties["proj:epsg"])
            break
    if epsg is None:
        lon = (bbox_ll[0] + bbox_ll[2]) / 2.0
        lat = (bbox_ll[1] + bbox_ll[3]) / 2.0
        zone = int(math.floor((lon + 180) / 6) + 1)
        epsg = 32600 + zone if lat >= 0 else 32700 + zone

    tx = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x1, y1 = tx.transform(bbox_ll[0], bbox_ll[1])
    x2, y2 = tx.transform(bbox_ll[2], bbox_ll[3])
    bounds_proj = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return epsg, bbox_ll, bounds_proj


def projected_intersection_ratio(item_geom, aoi_bounds, epsg_out):
    # Transform AOI bbox (in lon/lat) to projected coords
    tx = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_out}", always_xy=True)
    aoi_proj = shapely_transform(tx.transform, box(*aoi_bounds))

    # Get item's footprint and project it too
    geom = shape(item_geom)
    geom_proj = shapely_transform(tx.transform, geom)

    inter = geom_proj.intersection(aoi_proj)

    if inter.is_empty:
        return 0.0

    return inter.area / aoi_proj.area


def process_year(
    year: int, grid: str, max_cloud: int, bbox_ll: tuple|list, band_order: list, out_dir: str|Path, month_start_end: tuple|list
):
    print(f"\n==== Year {year} | grid={grid} | clouds≤{max_cloud}% ====")
    start_date = f"{year}-{month_start_end[0]}"
    end_date = f"{year}-{month_start_end[1]}"

    items = search_s2_stac(start_date, end_date, grid, max_cloud_cover=max_cloud)
    if not items:
        print("  ⚠️ No items for this year.")
        return 0

    items_s3 = prefer_s3_assets(items)
    epsg_out, bbox_ll_used, bounds_out = detect_epsg_and_bounds(
        items, bbox_ll_override=bbox_ll
    )
    print(f"  EPSG={epsg_out} | bounds_proj={tuple(round(v, 2) for v in bounds_out)}")

    bands_10m = [b for b in band_order if b.endswith("10m")]
    bands_20m = [b for b in band_order if b.endswith("20m")]

    out_dir = Path(out_dir) / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0

    with rasterio_env():
        for it_s3, it_orig in zip(items_s3, items):
            scene_date = it_orig.properties.get("datetime", "").split("T")[0]
            print(f"\n→ Scene {it_orig.id} ({scene_date})")
            # ------------------------------------------------
            # AOI intersection check (BEFORE loading bands)
            # ------------------------------------------------
            coverage_ratio = projected_intersection_ratio(
                item_geom=it_orig.geometry,
                aoi_bounds=bbox_ll,  # in lon/lat!
                epsg_out=epsg_out,  # detected for the tile
            )

            print(f"   ℹ️ AOI intersection coverage: {coverage_ratio:.2%}")

            if coverage_ratio < 0.4:  # Example: require 5% coverage
                print("   ⚠️ Scene skipped due to low AOI coverage.")
                continue

            try:
                ref = None
                pieces = []

                for bname in bands_10m:
                    if bname not in it_s3.assets:
                        continue
                    href = it_s3.assets[bname].href
                    da = rioxarray.open_rasterio(href, masked=True).squeeze(
                        "band", drop=True
                    )
                    if da.rio.crs is None:
                        da = da.rio.write_crs(f"EPSG:{epsg_out}")
                    da = da.rio.clip_box(*bounds_out)
                    if ref is None:
                        ref = da
                    pieces.append(da.expand_dims("band"))

                for bname in bands_20m:
                    if bname not in it_s3.assets:
                        continue
                    href = it_s3.assets[bname].href
                    da20 = rioxarray.open_rasterio(href, masked=True).squeeze(
                        "band", drop=True
                    )
                    if da20.rio.crs is None:
                        da20 = da20.rio.write_crs(f"EPSG:{epsg_out}")
                    da20 = da20.rio.clip_box(*bounds_out)
                    da20u = da20.rio.reproject_match(
                        ref, resampling=Resampling.bilinear
                    )
                    pieces.append(da20u.expand_dims("band"))

                if not pieces:
                    print("   ⚠️ No usable bands.")
                    continue

                scene = xr.concat(pieces, dim="band")
                scene = scene.assign_coords(band=BAND_LABELS)
                if scene.rio.crs is None:
                    scene = scene.rio.write_crs(f"EPSG:{epsg_out}")

                # MASKING
                red = scene.sel(band="Red").values
                green = scene.sel(band="Green").values
                nir = scene.sel(band="NIR").values
                input_array = np.stack([red, green, nir], axis=0)

                try:
                    pred_mask = predict_from_array(input_array)

                    # Handle shape (1, H, W) or (3, H, W)
                    if pred_mask.ndim == 3:
                        if pred_mask.shape[0] == 1:
                            pred_mask = pred_mask[0]
                        elif pred_mask.shape[0] == 3:
                            pred_mask = pred_mask[1]  # assume class 1 = cloud

                    # Ensure mask shape matches (y, x)
                    if pred_mask.shape != (scene.sizes["y"], scene.sizes["x"]):
                        raise ValueError(
                            f"❌ Mask shape {pred_mask.shape} does not match scene shape {(scene.sizes['y'], scene.sizes['x'])}"
                        )

                    # Keep only pixels where class == 0
                    mask_keep = pred_mask == 0

                    mask_da = xr.DataArray(
                        mask_keep,
                        dims=("y", "x"),
                        coords={"y": scene.coords["y"], "x": scene.coords["x"]},
                    )

                    scene = scene.where(mask_da)
                    print("   ✔ Cloud mask applied.")

                except Exception as e:
                    print("   ⚠️ Cloud mask failed:", e)

                scene_u16 = (
                    scene.fillna(0)
                    .clip(0, 10000)
                    .astype("uint16")
                    .rio.write_nodata(0)
                    .rio.write_crs(f"EPSG:{epsg_out}")
                )

                out_path = out_dir / f"{it_orig.id}_{scene_date}_masked.tif"
                print("   💾 Saving →", out_path)
                scene_u16.transpose("band", "y", "x").rio.to_raster(
                    out_path,
                    driver="GTiff",
                    compress="deflate",
                    tiled=True,
                    predictor=2,
                    BIGTIFF="IF_SAFER",
                    blockxsize=512,
                    blockysize=512,
                )
                n_ok += 1

            except Exception as e:
                print("   ❌ Scene failed:", e)

    return n_ok


def rasterio_env():
    return rio.Env(
        AWS_S3_ENDPOINT=os.environ["AWS_S3_ENDPOINT"],
        AWS_REGION=os.environ["AWS_REGION"],
        AWS_VIRTUAL_HOSTING=os.environ["AWS_VIRTUAL_HOSTING"],
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="tif,gtiff,jp2,xml",
    )
