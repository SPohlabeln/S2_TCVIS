from pathlib import Path

import rioxarray
import xarray as xr
from rasterio.enums import Resampling

from .online.rio import rasterio_env
from .online.stac import prefer_s3_assets, search_s2_stac
from .spatial.geom import detect_epsg_and_bounds, projected_intersection_ratio

BAND_ORDER = ["B02_10m", "B03_10m", "B04_10m", "B08_10m", "B11_20m", "B12_20m"]
BAND_LABELS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]


def download_year(
    year: int,
    grid: str,
    max_cloud: int,
    bbox_ll: tuple | list,
    band_order: list,
    out_dir: str | Path,
    month_start_end: tuple | list,
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

            # check if outfile exists
            out_path = out_dir / f"{it_orig.id}_{scene_date}.tif"
            if out_path.exists():
                print("   ⚠️ Scene skipped, output file already exists.")
                continue
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

                print("   ⬇️ Downloading and processing bands...")

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

                print("   💾 Saving →", out_path)
                scene.transpose("band", "y", "x").rio.to_raster(
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
