
from pystac import Item
from pystac_client import Client


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
