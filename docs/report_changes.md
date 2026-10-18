# Replacement text for Part 2

Each block below replaces the section of the report with the same heading. One sentence in Part 1 also changes, given at the end.

## Part 2: Reproduction

Code: https://github.com/YannKleees/hyperreimpute. I use the authors' hyperimpute package [2], version 0.1.17, with the settings of their notebook experiments_04_gain_of_function.ipynb, and version 0.1.5, released with the paper, for a check. All runs were on CPU.

```
git clone https://github.com/YannKleees/hyperreimpute.git && cd hyperreimpute
pip install -r requirements.txt
python experiments.py      # writes results/*.csv, then open reproduction.ipynb
python version_check.py    # once here and once in venv_015, see README
```

### 1. Selected Experiment

I reproduce the source of gains ablation (Table 2), which tests the paper's central claim that per column model selection improves iterative imputation, under MAR with 30% missingness and 10 seeds on airfoil, compression and wine_white; california and letter were infeasible on a laptop. Two controls use a correctly configured CatBoost (Section 4).

### 2. Comparison with the Paper

Table 1. RMSE x 10 on the masked cells under MAR with 30% missingness, averaged over 10 seeds. Lower is better.

| Method                | airfoil Paper | airfoil Ours | compression Paper | compression Ours | wine_white Paper | wine_white Ours |
| --------------------- | ------------: | -----------: | ----------------: | ---------------: | ---------------: | --------------: |
| ice_lr                |          2.35 |         1.95 |              1.43 |             1.49 |             0.92 |            0.86 |
| ice_rf                |          2.37 |         1.91 |              1.62 |             1.68 |             0.99 |            0.90 |
| ice_cb                |          2.08 |         2.55 |              1.25 |             2.29 |             0.88 |            1.10 |
| global_search         |          1.60 |         1.92 |              1.03 |             1.45 |             0.85 |            0.79 |
| column_naive          |          1.86 |         1.32 |              1.09 |             1.14 |             0.85 |            0.79 |
| wo_flexibility_cb     |          1.59 |         2.16 |              1.08 |             2.00 |             0.88 |            1.10 |
| wo_adaptivity         |          1.66 |         1.92 |              1.05 |             1.58 |             0.89 |            0.79 |
| HyperImpute           |          1.48 |         1.80 |              1.01 |             1.48 |             0.80 |            0.79 |
| ice_cb_fixed          |           n/a |     **1.25** |               n/a |         **1.04** |              n/a |        **0.75** |
| column_naive_cb_fixed |           n/a |     **1.25** |               n/a |         **1.04** |              n/a |        **0.75** |

The paper reports standard deviations between 0.29 and 0.53 on airfoil and between 0.03 and 0.22 on the other datasets. My column_naive follows Table 3 of the paper: each column chooses its model once, with default hyperparameters. z scores below use the standard error of the paper's mean (std / √10), and p values come from paired Wilcoxon signed rank tests over the 10 seeds, which share the same masks.

The baselines reproduce only in part. On airfoil, ice_lr, ice_rf and column_naive are 17%, 19% and 29% below the paper (z = -2.6 to -3.9), all in the same direction, which suggests different masks. On compression they lie within 1.6 standard errors, on wine_white 6% to 8% lower (z = -2.2 to -7.7). The Wasserstein distance deviates more, for example for ice_rf on compression, at 1.60 against 1.01 (z = 47).

Per column selection clearly beats ICE with linear models and random forests: column_naive lowers RMSE by 31%, 24% and 9% compared with the better of the two (p = 0.002 on each dataset). It does not beat the best single learner, however. ice_cb_fixed is 5% to 9% better than column_naive and 6% to 31% better than HyperImpute on all three datasets (p ≤ 0.004). With the same CatBoost in its pool, column_naive performs like ice_cb_fixed (mean differences below 0.01, p ≥ 0.64). The paper's gain over the best single learner therefore does not reproduce here.

Full HyperImpute reproduces only on wine_white. On airfoil and compression its RMSE exceeds the reported value by 22% and 46% (z = 3.4 and 10.1), and its Wasserstein distance is about 2.7 and 2.8 times the reported 0.29. Tuning makes it 37% and 30% worse than column_naive (p = 0.002) and raises the runtime from under a minute to five to six minutes per run. The tuned ablation rows global_search and wo_adaptivity are also worse than column_naive on these two datasets, where the paper reports them as better, and wo_flexibility_cb behaves like ice_cb.

### 3. Parameters

Table 2. Parameters used in all runs.

| Parameter              | Value                                                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| *Setup*                |                                                                                                                              |
| Software               | hyperimpute 0.1.17, pandas below 2.3, Python 3.11, CPU only                                                                  |
| Version check          | hyperimpute 0.1.5 installed with `--no-deps`, Python 3.11, numpy 1.23.5, pandas 1.5.3, scikit-learn 1.1.3, xgboost 1.6.2, catboost 1.2, torch 2.0.1 |
| Datasets               | airfoil, compression, wine_white, min max scaled                                                                             |
| Missingness            | MAR, rate 0.3, `column_limit=10`, seeds 0 to 9; version check on airfoil and compression, seeds 0 to 4, same masks          |
| Metrics                | RMSE on masked cells, Wasserstein distance summed over columns, paired Wilcoxon signed rank tests over seeds                |
| *Methods*              |                                                                                                                              |
| HyperImpute            | `optimizer="hyperband"`, `select_lazy=False`                                                                                 |
| column_naive           | `optimizer="simple"`, `select_model_by_column=True`, `select_model_by_iteration=False`, `select_lazy=True`                   |
| global_search          | `optimizer="hyperband"`, `select_model_by_column=False`, `select_model_by_iteration=False`, `select_lazy=True`               |
| wo_adaptivity          | as global_search, with `select_model_by_column=True`                                                                         |
| wo_flexibility_cb      | as global_search, with CatBoost as the only learner                                                                          |
| ICE variants           | `optimizer="simple"`, `select_model_by_column=False`, `select_model_by_iteration=False`, one fixed learner                   |
| ice_cb_fixed           | ICE with CatBoost, `n_estimators=1000`, `learning_rate=0.03`                                                                 |
| column_naive_cb_fixed  | column_naive with this CatBoost in place of the default one                                                                  |
| *Package defaults*     | 0.1.17, with 0.1.5 in brackets where it differs                                                                              |
| Learner pool           | linear and logistic regression, random forest, XGBoost, CatBoost                                                             |
| Imputation             | mean initialisation, random column order                                                                                     |
| Search                 | `n_inner_iter=40` (50), `select_patience=5` (3), `optimize_thresh=5000` (1000)                                               |

All other settings keep the package defaults. Neural networks join the learner pool only with a GPU, so they were not used.

### 4. Reasons for the Differences

The package changed after publication. Version 0.1.5 already searched CatBoost between 10 and 100 trees, and Table 5 of the paper lists only depth, n_estimators and grow_policy for CatBoost. Later versions added the learning rate range [0.01, 0.04], l2_leaf_reg, random_strength and min_data_in_leaf to the search and fixed the default learning rate at 0.001, which 0.1.5 left to CatBoost. With 10 trees at 0.001 the predictions barely leave the column mean, which explains ice_cb, and the tuned range of at most 100 trees at 0.04 keeps wo_flexibility_cb weak. Version 0.1.17 also searches a wider XGBoost space, raises the random forest default from 10 to 100 trees and lowers n_inner_iter from 50 to 40. The column order is random in both versions.

To test this explanation, I reran HyperImpute, column_naive and ice_cb under version 0.1.5 on airfoil and compression, with seeds 0 to 4 and the same masks. ice_cb improves from 2.67 to 1.82 and from 2.28 to 1.30, close to the reported 1.25 on compression. HyperImpute improves on every seed, by 19% and 21%, and its Wasserstein distance falls to 0.30 and 0.33, close to the reported 0.29. Yet it remains 5% and 3% worse than column_naive, which hardly changes. With five seeds the smallest possible Wilcoxon p value is 0.06, so these results are consistent across seeds but not formally significant. The version therefore explains ice_cb and the gap in Wasserstein distance, but not the paper's claim that tuning improves on column_naive. On airfoil, HyperImpute chose random forests of depth 3 for every column under 0.1.17 and XGBoost of depth 8 and 9 under 0.1.5. Runtimes come from parallel jobs on one laptop CPU, so only their ratios are meaningful.

### 5. Conclusion

Per column model selection clearly outperforms iterative imputation with a single linear or forest model, but a correctly configured single CatBoost is at least as good, so the paper's gain over the best single learner does not reproduce on these datasets. Full HyperImpute reproduces only on wine_white. Under version 0.1.5, later changes to CatBoost explain the failure of ice_cb and the gap in Wasserstein distance, but HyperImpute still does not improve on column_naive. Reproductions of this paper should fix the package version and check the configuration of every learner in the pool.

## Part 1, Strengths, first bullet

It removes the manual model choice that limits iterative imputation, and it inherits the properties of iterative imputation under MAR.
