# Data

Download the **DOHMH New York City Restaurant Inspection Results** dataset from NYC Open Data
(https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j)
via *Export -> CSV* and save it as:

    data/DOHMH_New_York_City_Restaurant_Inspection_Results.csv

The CSV is not committed to the repo (large file, updated daily).

To smoke-test the pipeline without the real data, generate a synthetic file with the same schema:

    python -m src.make_synthetic --out data/synthetic_inspections.csv
