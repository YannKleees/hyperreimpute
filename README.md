# HyperImpute

This project reproduces Table 2 of the HyperImpute paper (Jarrett et al., ICML 2022) on three UCI datasets and adds five tests of the method's limitations.

- Report: [`docs/Report.pdf`](docs/Report.pdf)
- Paper: [`docs/HyperImpute.pdf`](docs/HyperImpute.pdf)
- All tables and figures: [`reproduction.ipynb`](reproduction.ipynb)

## Key results

RMSE × 10 on the masked cells, MAR with 30% missingness, mean over 10 seeds. Lower is better.

| Method       | airfoil (paper) | airfoil (ours) | compression (paper) | compression (ours) | wine_white (paper) | wine_white (ours) |
| ------------ | --------------: | -------------: | ------------------: | -----------------: | -----------------: | ----------------: |
| ice_lr       |            2.35 |           1.95 |                1.43 |               1.49 |               0.92 |              0.86 |
| ice_rf       |            2.37 |           1.91 |                1.62 |               1.68 |               0.99 |              0.90 |
| ice_cb       |            2.08 |           2.55 |                1.25 |               2.29 |               0.88 |              1.10 |
| column_naive |            1.86 |           1.36 |                1.09 |               1.15 |               0.85 |              0.79 |
| HyperImpute  |            1.48 |           1.80 |                1.01 |               1.48 |               0.80 |              0.79 |
| ice_cb_fixed |               – |       **1.25** |                   – |           **1.04** |                  – |          **0.75** |

- Per column model selection (`column_naive`) lowers RMSE by 8% to 29% compared with the best single learner, so the central claim holds.
- Full HyperImpute with hyperparameter tuning matches the paper only on wine_white. Tuning makes it worse than `column_naive` and 3 to 18 times slower.
- `ice_cb` behaves like mean imputation, because hyperimpute 0.1.17 gives CatBoost 10 trees with a learning rate of 0.001. With 1000 trees and a learning rate of 0.03 (`ice_cb_fixed`), CatBoost has the lowest RMSE on all three datasets.

## Repository layout

```text
├── reproduction.ipynb     runs every experiment and shows all tables and figures
├── experiments.py         imputation methods, experiments and paper comparison
├── config.py              datasets, seeds, missingness settings and paper reference values
├── plugins/               CatBoost learners with sensible settings, used by ice_cb_fixed
├── results/               cached results, one CSV per experiment
├── docs/                  report and paper
└── requirements.txt
```

## Installation

You need Python 3.11 and an internet connection, because the datasets are downloaded from the UCI repository when the experiments run.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Running

Open `reproduction.ipynb`, select the environment you created above as the kernel, and run all cells.

The results are already in `results/`, so the notebook finishes in about a minute. To rerun an experiment from scratch, delete its CSV file. A full rerun takes several hours, because every method is run on every dataset with ten seeds. Each job is saved as soon as it finishes, so an interrupted run continues where it stopped.

You can also run the experiments without the notebook.

```
python experiments.py
```

## Experiments

| Notebook section  | Question                                                              | Results file       |
| ----------------- | --------------------------------------------------------------------- | ------------------ |
| 1. Reproduction   | All methods on airfoil, compression and wine_white, MAR 30%, 10 seeds | `main.csv`         |
| CatBoost baseline | ICE with CatBoost at 1000 trees and a learning rate of 0.03           | `ice_cb_fixed.csv` |
| Test 1            | Does hyperparameter tuning add anything to model choice alone?        | `main.csv`         |
| Test 2            | How much runtime does each method cost for its accuracy?              | `main.csv`         |
| Test 3            | How do MCAR, MAR and MNAR missingness compare?                        | `mechanisms.csv`   |
| Test 4            | What happens with few features?                                       | `features.csv`     |
| Test 5            | Do the imputed values keep the spread of the true values?             | `main.csv`         |

`table2_comparison.csv` compares RMSE with Table 2 of the paper. `paper_comparison.csv` compares RMSE (Table 2) and Wasserstein distance (Table 7), with z = (ours − paper) / paper std.

## Notes

- pandas must be older than 2.3. With newer versions, every imputer silently falls back to mean imputation.
- HyperImpute adds neural networks to its pool of learners only when PyTorch finds a CUDA GPU. Results for `hyperimpute` and `hyperimpute_naive` therefore depend on whether a GPU is present. All results here were computed on a CPU.
- `hyperimpute_naive` in the code corresponds to `column_naive` in the paper. Unlike the paper's version, it can still switch models between iterations.
