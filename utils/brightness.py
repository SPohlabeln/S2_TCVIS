
#---- Apply ice mask-------
# masked_median = ice_mask(
#     scene,
#     threshold=None,
#     band_names=(1, 2, 3),
# )


def ice_mask(
    scene,
    thresholds=None,
    band_names=("Band1", "Band2", "Band3")
):
    """
    Mask out pixels that are too bright in the specified bands
    (e.g., ice or snow). If all selected bands exceed their
    respective threshold, the pixel is set to NaN.

    Parameters
    ----------
    scene : xarray.DataArray
        Multiband Sentinel-2 scene (dims: band, y, x).
    thresholds : dict or None
        Dictionary mapping band_name -> threshold value.
        Example:
            {"Band1":2700, "Band2":2500, "Band3":2300}
        If None, defaults are used.
    band_names : tuple of str
        Names of the 3 bands to check for brightness.

    Returns
    -------
    masked_scene : xarray.DataArray
        Scene with ice pixels masked (set to NaN).
    """

    # Ensure bands exist
    missing = [b for b in band_names if b not in scene.band.values]
    if missing:
        raise ValueError(f"Scene is missing required bands: {missing}")

    # Default thresholds (sensible RGB defaults)
    if thresholds is None:
        thresholds = {
            band_names[0]: 2500,   # e.g. Blue
            band_names[1]: 2500,   # e.g. Green
            band_names[2]: 2300,   # e.g. Red
        }

    # Safety check
    for b in band_names:
        if b not in thresholds:
            raise ValueError(f"No threshold provided for band '{b}'")

    # Extract relevant bands
    b1 = scene.sel(band=band_names[0])
    b2 = scene.sel(band=band_names[1])
    b3 = scene.sel(band=band_names[2])

    # Apply band-specific thresholds
    ice_mask = (
        (b1 > thresholds[band_names[0]]) &
        (b2 > thresholds[band_names[1]]) &
        (b3 > thresholds[band_names[2]])
    )

    # Invert mask: keep only non-ice pixels
    keep_mask = ~ice_mask

    return scene.where(keep_mask)

