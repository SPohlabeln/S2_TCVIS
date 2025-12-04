import math

from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform


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
