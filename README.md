# Sentinel-2 TCVIS

Cloudmasking_tests_Phidown.ipynb : Cloud masking tests with Omnicloudmask & Phidown Downloads
S2_workflow_1.ipynb : TCVIS trend calculations with unsufficient masking & long loading (stac_load)
S2_workflow_1_omc.ipynb: TCVIS trend calculations with omnicloudmask integrated

## Installation

* recommended to use uv

`uv sync`

## Usage

### Scripts

#### Usage example

##### Download images

`uv run download_scenes.py --month-start 07-01 --month-end 08-31 --years-start 2017 --years-end 2025 --n-parallel 4 --output-dir data/coverage70/scenes_raw`

#### Mask images

##### running with CPU (ca 1:45 min per image)
`uv run mask_scenes.py --device cpu --input-dir data/coverage70/scenes_raw/2017`

##### running on cuda device (ca. 0:10 min per image)
`uv run mask_scenes.py --input-dir data/coverage70/
scenes_raw/2017 --output-dir data/coverage70/scenes_masked/2017 --device cuda:2`

#### Create mosaics



#### Calculate Indices


#### Calculate Trends
