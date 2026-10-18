# HyperImpute

This project reproduces Table 2 of the HyperImpute paper (Jarrett et al., ICML 2022) on three UCI datasets and tests five of the method's limitations.

## Where to look

- [`docs/Report.pdf`](docs/Report.pdf): the report with the results and findings
- [`reproduction.ipynb`](reproduction.ipynb): runs every experiment and shows all tables and figures
- [`docs/HyperImpute.pdf`](docs/HyperImpute.pdf): the original paper
- `experiments.py` and `config.py`: the code and settings behind the notebook
- `learners/`: CatBoost with 1000 trees and a learning rate of 0.03, used by `ice_cb_fixed`
- `results/`: the cached results, one CSV per experiment

## Running

You need Python 3.11 and an internet connection, because the datasets are downloaded from UCI.

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then open `reproduction.ipynb` with this environment and run all cells. The results are cached in `results/`, so this takes about a minute. To rerun an experiment, delete its CSV. A full rerun takes several hours.

## Notes

- pandas must be older than 2.3. With newer versions, every imputer silently falls back to mean imputation.
- The paper's `column_naive` is called `hyperimpute_naive` in the code.
