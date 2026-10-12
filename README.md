# HyperImpute: Reproduction and Limitation Tests

This project reproduces Table 2 of the HyperImpute paper (Jarrett et al., 2022) on three UCI datasets and adds five tests of the method's limitations.

## Requirements

You need Python 3.11 and an internet connection, because the datasets are downloaded from the UCI repository when the experiments run.

## Installation

Create a separate environment and install the packages into it.

```
python -m venv .venv
.venv\Scripts\activate
pip install hyperimpute "pandas<2.3" xlrd seaborn ipykernel
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

pandas must be older than 2.3. With newer versions, every imputer silently falls back to mean imputation.

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

## Output

The folder `results/` holds one CSV file per experiment, plus `table2_comparison.csv`, which compares our numbers with the paper. To rerun an experiment from scratch, delete its CSV file.

## Experiments

- **Reproduction.** All methods are compared on airfoil, compression, and wine_white with 30% of values missing at random.
- **Test 1.** Hyperparameter tuning is compared with model choice alone.
- **Test 2.** Runtime is compared with accuracy.
- **Test 3.** The missingness mechanisms MCAR, MAR, and MNAR are compared.
- **Test 4.** The number of features is reduced.
- **Test 5.** The spread of the imputed values is compared with the spread of the true values.
