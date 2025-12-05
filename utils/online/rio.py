import os

import rasterio as rio


def rasterio_env():
    return rio.Env(
        AWS_S3_ENDPOINT=os.environ["AWS_S3_ENDPOINT"],
        AWS_REGION=os.environ["AWS_REGION"],
        AWS_VIRTUAL_HOSTING=os.environ["AWS_VIRTUAL_HOSTING"],
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="tif,gtiff,jp2,xml",
    )
