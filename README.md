# HyperImpute

This project reproduces Table 2 of the HyperImpute paper (Jarrett et al., ICML 2022) on three UCI datasets, reruns the key methods under hyperimpute 0.1.5, and adds a few tests of the method's limitations.

## Where to look

```text
├── reproduction.ipynb            runs every experiment and shows all tables and figures
├── docs/
│   ├── Report.pdf                the report with the results and findings
│   └── HyperImpute.pdf           the original paper
├── hyperreimpute/                the code behind the notebook
│   ├── config.py                 datasets, seeds, methods compared and paper reference values
│   ├── experiments.py            imputation methods, experiments, paper comparison and paired tests
│   ├── version_check.py          reruns hyperimpute, column_naive_paper and ice_cb under hyperimpute 0.1.5
│   └── learners/                 CatBoost with 1000 trees and a learning rate of 0.03, used by ice_cb_fixed
└── results/                      the cached results, one CSV per experiment
    ├── main.csv                  the Table 2 methods
    ├── ice_cb_fixed.csv          ICE with the fixed CatBoost
    ├── ablation_extra.csv        column_naive_paper and the paper's other Table 2 rows
    ├── column_naive_cb_fixed.csv column_naive with the fixed CatBoost in its pool
    ├── version_015.csv           the same methods and masks under hyperimpute 0.1.5
    ├── selections.csv            learners and hyperparameters chosen per column, airfoil seed 0
    ├── masks/                    the masks of the version check, saved under 0.1.17
    ├── mechanisms.csv            limitation test: MCAR, MAR and MNAR
    └── features.csv              limitation test: few features
```

## Running

You need Python 3.11 and an internet connection, because the datasets are downloaded from UCI.

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then open `reproduction.ipynb` with this environment and run all cells. The results are cached in `results/`, so this takes about a minute. To rerun an experiment, delete its CSV. A full rerun takes several hours. Without the notebook, `python -m hyperreimpute.experiments` runs all experiments.

### Version check with hyperimpute 0.1.5

The metadata of hyperimpute 0.1.5 contains an invalid pin (`loguru (==.0.6.0)`), which pip 24.1 and later reject. Install it without dependencies, with an older pip, in a separate environment:

```
python -m venv venv_015
venv_015\Scripts\activate
python -m pip install "pip<24.1"
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.1.3 scipy==1.9.3 xgboost==1.6.2 catboost==1.2 optuna==2.10.1 "sqlalchemy<2" loguru==0.6.0 "pydantic<2" ipython xlrd packaging joblib tqdm
pip install --no-deps hyperimpute==0.1.5
```

Run `python -m hyperreimpute.version_check` from this folder once in the main environment (saves the masks and the 0.1.17 selections) and then once in `venv_015` (writes `version_015.csv` and the 0.1.5 selections).

## Notes

- pandas must be older than 2.3. With newer versions, every imputer silently falls back to mean imputation.
- `column_naive_paper` is the paper's `column_naive`. `hyperimpute_naive` is column_naive plus adaptivity, as it may still switch models between iterations.
