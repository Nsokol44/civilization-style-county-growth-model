#!/usr/bin/env python3
"""Download the raw public inputs for the county panel.

The processed files in this directory (county_panel.csv, county_static.csv,
dist_km.npy) are the files the results were generated from. Their SHA-256
hashes are in SHA256SUMS.txt. This script downloads the raw source files into
raw/. The processing steps (Virginia combined-area harmonisation and so on)
are described in DATA_MANIFEST.md.

Sources (downloaded 2026-10-06):
  BEA CAGDP1  https://apps.bea.gov/regional/zip/CAGDP1.zip
  BEA CAINC1  https://apps.bea.gov/regional/zip/CAINC1.zip
  BEA CAGDP2  https://apps.bea.gov/regional/zip/CAGDP2.zip
  Census Gazetteer 2020
              https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip
  USDA ERS Natural Amenities Scale
              https://www.ers.usda.gov/media/6172/natural-amenities-scale-including-the-6-components-for-us-counties.xls
  FEMA NRI v1.20 (used only in the sensitivity check)
              https://www.fema.gov/about/reports-and-data/openfema/nri/v120/NRI_Table_Counties.zip
  Census cartographic boundaries 2020 (500k), source of
  data/county_geometries.geojson (region counties only; Virginia's 51
  components dissolved onto the 23 BEA combined areas; projected to
  EPSG:5070 and simplified at 400 m). Not downloaded by this script:
              https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip
"""
import hashlib
import urllib.request
from pathlib import Path

URLS = {
    "CAGDP1.zip": "https://apps.bea.gov/regional/zip/CAGDP1.zip",
    "CAINC1.zip": "https://apps.bea.gov/regional/zip/CAINC1.zip",
    "CAGDP2.zip": "https://apps.bea.gov/regional/zip/CAGDP2.zip",
    "2020_Gaz_counties_national.zip":
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip",
    "natural_amenities.xls":
        "https://www.ers.usda.gov/media/6172/natural-amenities-scale-including-the-6-components-for-us-counties.xls",
    "NRI_Table_Counties.zip":
        "https://www.fema.gov/about/reports-and-data/openfema/nri/v120/NRI_Table_Counties.zip",
}

if __name__ == "__main__":
    raw = Path(__file__).resolve().parent / "raw"
    raw.mkdir(exist_ok=True)
    for name, url in URLS.items():
        dst = raw / name
        if dst.exists():
            print("exists, skipped:", dst)
            continue
        print("downloading", url)
        urllib.request.urlretrieve(url, dst)
        print("  sha256:", hashlib.sha256(dst.read_bytes()).hexdigest())
    print("Raw files in", raw, "- the processed tables county_panel.csv and county_static.csv are already in this directory; DATA_MANIFEST.md lists their sources.")
