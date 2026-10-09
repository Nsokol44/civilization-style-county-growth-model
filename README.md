# Can a Civilization-style county growth model beat "no change"? A frozen-hindcast test

Code for a recursive spatial panel model of county real GDP and population
growth in nine southeastern states (TN, KY, NC, SC, GA, AL, MS, VA, FL; 841
counties, 2001-2019). The model is estimated on a calibration window ending at
a forecast origin and iterated forward to 2019 without further updating. The
hindcasts are compared with persistence, trend, and shrunk-trend baselines and
with a graph neural network.

Nicholas J. Sokol, Department of Geography & Sustainability, University of
Tennessee, Knoxville (nsokol@utk.edu).

## Contents

- `code/model.py`: the panel model (GDP and population growth equations, goods
  and services share drift equations; pooled, state FE, state x year FE and
  county FE specifications; recursive hindcasts).
- `code/evaluation.py`: baselines, shrinkage selection, state-clustered
  bootstrap intervals, Diebold-Mariano statistics.
- `code/gate_repro.py`: baseline comparison at the 2010 origin for the archived
  v1 ensemble hindcast.
- `code/neural.py`: GraphSAGE-style graph neural network (NumPy/SciPy) and an
  aspatial MLP (scikit-learn).
- `code/scenarios.py`: S1 and S2 sensitivity grids.
- `code/figures.py`: the four figures.
- `code/run_all.py`: runs everything.
- `data/`: input files (see below).
- `results/`: output tables, figures, and `run_manifest.json`.
- `figures/`: copy of `results/figures/`.

## Install and run

Python 3.12 with the pinned packages:

```
pip install -r requirements.txt
cd code
python run_all.py
```

The run takes about 3 minutes and rewrites the files in `results/`.

## Data

All inputs are in `data/`. SHA-256 hashes are in `data/SHA256SUMS.txt`, and
`data/fetch_data.py` lists the download URLs.

- `county_panel.csv`: real GDP, population and sector shares by county and
  year (BEA CAGDP1, CAINC1, CAGDP2).
- `county_static.csv`: county attributes (Census Gazetteer 2020, USDA ERS
  Natural Amenities Scale, FEMA National Risk Index v1.20). The NRI variables
  are used only in the sensitivity check.
- `dist_km.npy`: county-to-county distance matrix.
- `county_geometries.geojson`: county boundaries for the maps (Census
  cartographic boundary files, 2020).
- `v1_hindcast_county.csv`, `v1_hindcast_metrics.csv`: archived output of an
  earlier stochastic version of the model (ensemble median), used by
  `gate_repro.py`.
- `DATA_MANIFEST.md`, `build_report.json`: notes on how the files were built.

## Output

`run_all.py` writes these to `results/`:

- `gate2010_*.csv`, `gate2010_summary.json`: 2010-origin comparison for the v1 hindcast.
- `rolling_metrics.csv`, `rolling_inference.csv`: hindcasts from origins 2006,
  2008, 2010 and 2012 to 2019, for each specification and year-effect policy,
  with bootstrap and Diebold-Mariano results.
- `lambda_validation_all_origins.csv`, `coefficients_2010.csv`.
- `neural_cv.csv`, `hindcast_county_2010_neural.csv`: neural benchmark.
- `hindcast_county_2010_headline_{mean,conditional}.csv`: county hindcasts.
- `nri_sensitivity.csv`, `spatial_diagnostic.csv`.
- `S1_grid.csv`, `S2_band.csv`, `S1_grid2d_*.csv`, `S2_grid2d.csv`: scenario grids.
- `figures/`: figures 1 to 4 as PDF and PNG.
- `run_manifest.json`: package versions, seeds, data hashes.

## Headline numbers

Taken from the files in `results/`.

2010 origin, v1 ensemble hindcast, 2010-2019 population growth
(`gate2010_bootstrap_ci.csv`): RMSE 0.0739 for the simulation against 0.0957
for the best shrunk trend (difference -0.0219, state-clustered 95% interval
-0.0339 to -0.0072). For GDP growth the simulation RMSE is 0.2059 against
0.2068 (difference -0.0009, interval -0.0119 to 0.0126).

Rolling origins, demeaned RMSE of the state x year FE simulation (conditional
policy), the graph neural network (conditional policy), and baselines
(`rolling_metrics.csv`):

| Outcome | Origin | Simulation | GNN | Best shrunk trend | Persistence |
|---|---|---|---|---|---|
| Population | 2006 | 0.1001 | 0.1016 | 0.1163 | 0.1241 |
| Population | 2008 | 0.0775 | 0.0785 | 0.0887 | 0.1054 |
| Population | 2010 | 0.0686 | 0.0664 | 0.0711 | 0.0920 |
| Population | 2012 | 0.0527 | 0.0510 | 0.0541 | 0.0754 |
| GDP | 2006 | 0.2120 | 0.1989 | 0.2160 | 0.2232 |
| GDP | 2008 | 0.1963 | 0.1981 | 0.2044 | 0.2082 |
| GDP | 2010 | 0.1882 | 0.1966 | 0.2022 | 0.2045 |
| GDP | 2012 | 0.1575 | 0.1606 | 0.1643 | 0.1665 |

NRI sensitivity (`nri_sensitivity.csv`, 2010 origin, state x year FE,
conditional policy): population demeaned RMSE 0.0686 without the NRI variables
and 0.0691 with them; GDP 0.1882 and 0.1895.

County FE specification, 2006 origin, mean policy (`rolling_metrics.csv`): GDP RMSE 243.6 and
population RMSE 0.848, because the recursion diverges.
