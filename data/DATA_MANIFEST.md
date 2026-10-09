# Data manifest

Region: TN, KY, NC, SC, GA, AL, MS, VA, FL. The modelling universe is 841 counties and county-equivalents on BEA geography (Virginia is handled as described in section 7). All 841 have a complete BEA panel for 2001-2019, 15,979 county-years in total, with no county-year missing core GDP or population. Raw files were downloaded on 2026-10-06 from the public URLs below, with no account or credentials.

## Files in this directory

SHA-256 hashes for these files are in `SHA256SUMS.txt`.

| File | Contents |
|---|---|
| county_panel.csv | County-year panel: real GDP, personal income, population, per-capita income, sector shares (sections 1 to 3) |
| county_static.csv | County attributes: Census Gazetteer 2020, USDA ERS Natural Amenities Scale, FEMA National Risk Index v1.20 (sections 4 to 6) |
| dist_km.npy | County-to-county distance matrix in km, haversine between county centroids (Gazetteer internal points) |
| county_geometries.geojson | County boundaries for the maps, derived from the 2020 cartographic boundary file (section 6) |
| build_report.json | Counts and the Virginia component map from the build (section 7) |
| v1_hindcast_county.csv, v1_hindcast_metrics.csv | Archived model hindcast output by county and its error metrics, read by `code/gate_repro.py` |
| fetch_data.py | Downloads the raw source files listed below into `raw/` |

## Sources

| Table or file | Source | URL | Raw file |
|---|---|---|---|
| CAGDP1 | BEA Regional Economic Accounts, county GDP summary | https://apps.bea.gov/regional/zip/CAGDP1.zip | CAGDP1.zip, 1,928,241 bytes |
| CAINC1 | BEA, county personal income summary | https://apps.bea.gov/regional/zip/CAINC1.zip | CAINC1.zip, 3,467,441 bytes |
| CAGDP2 | BEA, GDP by industry in current dollars | https://apps.bea.gov/regional/zip/CAGDP2.zip | CAGDP2.zip, 15,288,721 bytes |
| Natural Amenities Scale | USDA ERS | https://www.ers.usda.gov/media/6172/natural-amenities-scale-including-the-6-components-for-us-counties.xls | natural_amenities.xls, 1,918,464 bytes |
| National Risk Index v1.20 (December 2025), county table | FEMA | https://www.fema.gov/about/reports-and-data/openfema/nri/v120/NRI_Table_Counties.zip | NRI_Table_Counties.zip, 24,966,535 bytes |
| 2020 Gazetteer, counties | US Census Bureau | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip | 2020_Gaz_counties_national.zip, 141,524 bytes |
| 2020 cartographic boundary file, 500k | US Census Bureau | https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip | cb_2020_us_county_500k.zip, 12,587,934 bytes |

`python fetch_data.py` downloads the first six files into `data/raw/` and prints their SHA-256. The boundary file is not downloaded by the script. The scripts that turn the raw files into the processed files are not included; the steps are described in sections 1 to 7.

## 1. BEA county real GDP (CAGDP1)

LineCode 1, real GDP in thousands of chained 2017 dollars, 2001-2019 (the file spans 2001-2024). Member used: `CAGDP1__ALL_AREAS_2001_2024.csv`, 9,565 data rows. The CSV is latin-1 encoded and GeoFIPS values have a leading space. The values (D) and (NA) are read as missing. Real GDP has no suppressed values in the region panel.

## 2. BEA county personal income and population (CAINC1)

LineCode 1 personal income (thousands of dollars), LineCode 2 population, LineCode 3 per-capita personal income, 2001-2019 (the file spans 1969-2024). Member used: `CAINC1__ALL_AREAS_1969_2024.csv`, 9,631 data rows. Personal income is kept in the panel but is not simulated separately.

## 3. BEA GDP by industry (CAGDP2)

Member used: `CAGDP2__ALL_AREAS_2001_2024.csv`, 108,362 data rows. Sector shares are current-dollar shares of the all-industry total (LineCode 1): goods-producing is LineCode 91, services-providing is LineCode 92, government is LineCode 83. Farm output is inside goods-producing. Real GDP by industry (CAGDP9) is not used.

Industry detail is partly suppressed at county level. In 2010, 728 of the 841 counties (86.6%) have all three sector shares observed. The rest are filled for simulation initialisation by within-county forward and backward fill, then the regional median. The sector-drift regressions use only the observed county-years (5,625 of 6,728).

## 4. USDA ERS Natural Amenities Scale

County file with the six underlying measures (January and July temperature, January sun, July humidity, topography scale, water area), standardised scores, the composite scale and its rank. Climate normals are 1941-70 and topography is the US DOI Land Surface Form scale (1-21). The file is read with xlrd 2.0.2. Sheet `NATAMENF` has about 100 rows of documentation before the header, so county rows are found by their 5-digit FIPS pattern: 3,111 county rows nationally and 872 in the region states before harmonisation to BEA geography.

The file has no row for Miami-Dade County, FL (12086), which it lists under the older Dade code 12025. Miami-Dade's standardised topography, January temperature and amenity values are set to the regional mean (z = 0) in estimation. It is the only universe county without amenity data (840 of 841 have it).

## 5. FEMA National Risk Index

County table `NRI_Table_Counties.csv` (24,639,216 bytes, 3,232 county rows, 465 columns). Variables used: `RISK_SCORE`, `EAL_VALT`, `SOVI_SCORE`, `RESL_SCORE`, the hazard risk scores (`*_RISKS`, 0-100) for drought, hurricane, inland flooding, tornado, hail, strong wind, wildfire, heat wave and coastal flooding, and `DRGT_AFREQ`. The hazard composite is the mean of the eight available `*_RISKS` scores divided by 100. The NRI variables are used only in the sensitivity check.

A county is flagged coastal when its coastal-flooding score is positive (`CFLD_RISKS` > 0), which gives 153 of the 841 counties. No coastline geometry is used.

## 6. Census Gazetteer 2020 and boundary file

The Gazetteer counties file (`2020_Gaz_counties_national.txt`, 3,221 rows, 869 of them county-equivalents in the nine region states) supplies land area and water area (ALAND, AWATER, in m2), internal-point latitude and longitude, and the water share AWATER / (ALAND + AWATER). Distances in `dist_km.npy` are haversine distances between these points.

`county_geometries.geojson` comes from the 500k cartographic boundary file: region counties only, Virginia's 51 component areas dissolved onto the 23 BEA combined areas, projected to EPSG:5070 and simplified at 400 m. It is used only for the hindcast-error maps.

## 7. Virginia combined areas

BEA does not report most Virginia independent cities separately. It publishes 23 combined county and city areas under pseudo-FIPS 519xx, with the combination named in the BEA GeoName (for example `51901 "Albemarle + Charlottesville, VA*"`). The Gazetteer, ERS and NRI files list the 51 components separately, so they are aggregated onto the BEA areas: areas are summed, climate, topography and amenity values are land-area-weighted, NRI percentile scores are population-weighted (EAL is summed), centroids are land-area-weighted, and coastal is the maximum over components. The component map for all 23 areas is in `build_report.json`. The 23 areas use exactly the 51 Gazetteer region rows that have no separate BEA row. Other Virginia independent cities (for example Richmond, Virginia Beach, Norfolk, Alexandria) are reported by BEA on their own and enter the universe as themselves.

County count: 841 BEA region counties = 869 Gazetteer region rows - 51 absorbed components + 23 combined areas.
