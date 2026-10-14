# HyperImpute: Reproduction and Limitation Tests

This project reproduces Table 2 of the HyperImpute paper (Jarrett et al., 2022) on three UCI datasets and adds five tests of the method's limitations.

## Requirements

You need Python 3.11 and an internet connection, because the datasets are downloaded from the UCI repository when the experiments run.

## Installation

Create a separate environment and install the packages from `requirements.txt` into it.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

pandas must be older than 2.3. With newer versions, every imputer silently falls back to mean imputation.

HyperImpute adds neural networks to its pool of learners only when PyTorch finds a CUDA GPU. Results for `hyperimpute` and `hyperimpute_naive` therefore depend on whether a GPU is present.

## Running

Open `reproduction.ipynb`, select the environment you created above as the kernel, and run all cells. The notebook runs the experiments and shows every table and figure.

The first run takes several hours, because every method is run on every dataset with ten seeds. A progress bar shows how far each experiment is. Each experiment is saved when it finishes, so later runs take seconds.

You can also run the experiments without the notebook.

```
python experiments.py
```

## Files

- `config.py` contains all settings, such as the datasets, seeds, and missingness rate.
- `experiments.py` runs the imputation methods and saves the results.
- `reproduction.ipynb` runs everything and shows the tables and figures.
- `plugins/` holds the CatBoost learners with sensible settings used by `ice_cb_fixed`.

## Output

The folder `results/` holds one CSV file per experiment, plus two comparisons with the paper. `table2_comparison.csv` compares RMSE with Table 2, and `paper_comparison.csv` compares both RMSE (Table 2) and Wasserstein distance (Table 7), with z = (ours − paper) / paper std. `ice_cb_fixed.csv` holds the CatBoost run with sensible settings. To rerun an experiment from scratch, delete its CSV file.

## Experiments

- **Reproduction.** All methods are compared on airfoil, compression, and wine_white with 30% of values missing at random.
- **Test 1.** Hyperparameter tuning is compared with model choice alone.
- **Test 2.** Runtime is compared with accuracy.
- **Test 3.** The missingness mechanisms MCAR, MAR, and MNAR are compared.
- **Test 4.** The number of features is reduced.
- **Test 5.** The spread of the imputed values is compared with the spread of the true values.
- **CatBoost baseline.** In hyperimpute 0.1.17, the CatBoost learners default to 10 trees with a learning rate of 0.001, so `ice_cb` barely moves away from its starting values and behaves like mean imputation. `ice_cb_fixed` reruns ICE with CatBoost at 1000 trees and a learning rate of 0.03, on the same three datasets and ten seeds.
