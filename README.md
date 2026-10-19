# SAR Flood Mapping in Python: Arakawa River, August 2022

Flood extent mapping from Sentinel-1 radar for the August 2022 Arakawa River flood
(Murakami / Sakamachi, Niigata), checked pixel by pixel against the GSI
(Geospatial Information Authority of Japan) inundation estimate.

A Python rewrite of my earlier Google Earth Engine workflow.

## Why radar

Heavy cloud covered the flood, so optical imagery could not see it. Radar works through
cloud. Calm floodwater reflects the radar pulse away from the satellite, so flooded land
appears dark in the after image compared with the before image.

## Method

1. **Data:** Sentinel-1 IW GRD, VH and VV, descending relative orbit 46, sigma0.
   Before: 22 Jul 2022. After: 3 Aug 2022.
2. **Common grid:** all layers resampled to one 10 m grid (UTM 54N) covering the GSI
   reference area plus 3 km.
3. **Speckle filter:** 50 m circular moving average, applied to linear power.
4. **Change detection:** before/after ratio above 1.25 (the UN-SPIDER value) is flagged
   as a backscatter drop.
5. **Permanent water mask:** OSM water polygons and JRC Global Surface Water
   (water at least 10 months per year).
6. **Terrain mask:** pixels more than 15 m above the nearest drainage (HAND) are removed.
   Elevation from Copernicus DEM.
7. **Clean-up:** patches of 20 connected pixels or fewer are removed.
8. **Validation:** IoU, precision and recall against GSI, within 300 m of the reference
   area, plus recall per ESA WorldCover land-cover class.

Outputs: flood mask (GeoTIFF), flood polygons (GeoPackage), change ratio (GeoTIFF),
metrics (CSV) and a quick-look map. All parameters are in
[`config/arakawa_2022.yaml`](config/arakawa_2022.yaml).

## Run it

```bash
conda env create -f environment.yml
conda activate sarflood

python -m sarflood download --config config/arakawa_2022.yaml   # once, saved to data/raw/
python -m sarflood run      --config config/arakawa_2022.yaml
pytest
```

The download step needs a free [Copernicus Data Space](https://dataspace.copernicus.eu)
account. The DEM, JRC water and WorldCover layers come from Microsoft Planetary Computer
and need no account. The pipeline reads local GeoTIFFs from `data/raw/`, so exports from
another source (for example GEE) with the same file names also work.

## Results

Threshold 1.25, compared with the GSI survey within 300 m of the reference area:

| | IoU | Precision | Recall | Cropland recall | Built-up recall |
|---|---|---|---|---|---|
| **VH** (Python) | 0.295 | 0.408 | 0.515 | 67.5 % | 5.5 % |
| VV (Python) | 0.246 | 0.425 | 0.370 | 50.2 % | 2.4 % |
| VH (earlier GEE run) | 0.30 | 0.41 | 0.52 | - | 5.6 % |

The results are close to the GEE run, despite the differences listed below.

![VH detection vs GSI reference](docs/flood_map_VH.png)

Flooded cropland behaves like open water and is mostly detected. Flooded built-up areas
(the large red area around Fujisawa) are mostly missed: water next to walls creates a
double-bounce reflection that raises backscatter instead of lowering it, so a method that
looks for darkening cannot find it.

## Implementation notes

- All layers are resampled to the same grid before any pixel comparison, so every array
  covers exactly the same ground.
- Speckle filtering and the change ratio use linear power rather than dB, because speckle
  noise is multiplicative.
- No-data pixels are ignored by the filter instead of being treated as zero, which would
  otherwise create dark, flood-like edges.
- HAND is approximated as elevation minus the elevation of the nearest drainage pixel.
  A full HAND calculation would follow flow paths over the DEM.
- Metrics are only computed near the GSI reference, because the GSI map is a sample and
  areas outside it are not confirmed dry.

## Differences from the GEE version

| | GEE | Python |
|---|---|---|
| SAR | `COPERNICUS/S1_GRD` (sigma0, dB) | openEO `SENTINEL1_GRD` (sigma0, linear) |
| DEM | SRTM 30 m | Copernicus DEM GLO-30 |
| HAND drainage | MERIT Hydro and JRC | OSM and JRC |
| HAND method | minimum elevation within 3 km, 90 m | nearest drainage pixel, 10 m |

## Limitations and next steps

- Built-up flooding is largely missed (see Results).
- The Sentinel-1 pass and the GSI survey were not at the same time.
- Next: train a random forest on the same scenes to see whether it improves built-up recall.

## Project layout

```
config/            run settings
data/reference/    GSI flood reference and OSM water polygons
src/sarflood/      code
tests/             tests on small synthetic scenes
docs/              figures for this README
```

## Data sources

- Inundation reference: Geospatial Information Authority of Japan (GSI), 浸水推定図
- Sentinel-1: Copernicus Sentinel data 2022, via Copernicus Data Space Ecosystem
- Copernicus DEM GLO-30: © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA
- JRC Global Surface Water: Pekel et al. (2016), European Commission Joint Research Centre
- ESA WorldCover 2021: © ESA WorldCover project, CC BY 4.0
- Permanent water polygons: © OpenStreetMap contributors, ODbL
