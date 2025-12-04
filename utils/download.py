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
    n_parallel: int = 1,
):
    """
    Download and process all scenes for a given year.

    If n_parallel > 1, scenes are downloaded/processed in parallel using threads.
    """
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

    def process_scene(it_s3, it_orig):
        """Process a single scene (download bands, reproject, save). Runs inside rasterio_env."""
        scene_date = it_orig.properties.get("datetime", "").split("T")[0]
        print(f"\n→ Scene {it_orig.id} ({scene_date})")

        out_path = out_dir / f"{it_orig.id}_{scene_date}.tif"
        if out_path.exists():
            print("   ⚠️ Scene skipped, output file already exists.")
            return False

        # AOI intersection check (BEFORE loading bands)
        coverage_ratio = projected_intersection_ratio(
            item_geom=it_orig.geometry,
            aoi_bounds=bbox_ll,  # in lon/lat!
            epsg_out=epsg_out,  # detected for the tile
        )
        print(f"   ℹ️ AOI intersection coverage: {coverage_ratio:.2%}")

        if coverage_ratio < 0.4:
            print("   ⚠️ Scene skipped due to low AOI coverage.")
            return False

        try:
            ref = None
            pieces = []

            print("   ⬇️ Downloading and processing bands...")

            # 10m bands
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

            # 20m bands (resample to match ref)
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
                if ref is None:
                    # If no 10m reference exists, use this 20m as ref (no upsampling here)
                    ref = da20
                    pieces.append(da20.expand_dims("band"))
                else:
                    da20u = da20.rio.reproject_match(
                        ref, resampling=Resampling.bilinear
                    )
                    pieces.append(da20u.expand_dims("band"))

            if not pieces:
                print("   ⚠️ No usable bands.")
                return False

            scene = xr.concat(pieces, dim="band")
            scene = scene.assign_coords(band=BAND_LABELS[: len(scene.band)])
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
            return True

        except Exception as e:
            print("   ❌ Scene failed:", e)
            return False

    # If running sequentially, keep the original behaviour
    if n_parallel is None or n_parallel <= 1:
        with rasterio_env():
            for it_s3, it_orig in zip(items_s3, items):
                success = process_scene(it_s3, it_orig)
                if success:
                    n_ok += 1
    else:
        # Parallel execution using threads. Each worker creates its own rasterio_env
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def run_process_in_env(s3, orig):
            try:
                with rasterio_env():
                    return process_scene(s3, orig)
            except Exception as e:
                print("   ❌ Scene failed (executor):", e)
                return False

        futures = []
        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            for it_s3, it_orig in zip(items_s3, items):
                futures.append(executor.submit(run_process_in_env, it_s3, it_orig))

            for fut in as_completed(futures):
                try:
                    success = fut.result()
                except Exception as e:
                    print("   ❌ Scene failed (executor result):", e)
                    success = False
                if success:
                    n_ok += 1

    return n_ok
