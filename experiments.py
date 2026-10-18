from hyperimpute.utils.benchmarks import *  # first, so the imports below win (it exports its own `time`)
import time
import warnings
from functools import cache
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from packaging.version import Version
from scipy import stats
from tqdm.auto import tqdm
from hyperimpute.plugins.prediction import classifiers, regression
import config as cfg
warnings.filterwarnings("ignore")  # hyperimpute floods the output with warnings
assert Version(pd.__version__) < Version("2.3"), "with pandas>=2.3 every imputer becomes mean imputation"


@cache
def load(dataset):
    """Download a UCI dataset."""
    url = cfg.URLS[dataset]
    if url.endswith(".xls"):
        return pd.read_excel(url)
    return pd.read_csv(url, **cfg.CSV_OPTIONS[dataset])


def imputer(name="hyperimpute", **settings):
    return lambda seed: Imputers().get(name, random_state=seed, **settings)


def ice(regressor, classifier):
    """HyperImpute with one fixed learner for all columns"""
    return imputer(
        regression_seed=[regressor],
        classifier_seed=[classifier],
        optimizer="simple",
        select_model_by_column=False,
        select_model_by_iteration=False,
        select_lazy=True,
    )


METHODS = {
    "hyperimpute": imputer(optimizer="hyperband", select_lazy=False),
    # column_naive plus adaptivity: it may still switch models between iterations
    "hyperimpute_naive": imputer(optimizer="simple"),
    "ice_lr": ice("linear_regression", "logistic_regression"),
    "ice_rf": ice("random_forest_regressor", "random_forest"),
    "ice_cb": ice("catboost_regressor", "catboost"),
}
for name in ["sklearn_ice", "sklearn_missforest", "mean", "softimpute", "gain", "miwae"]:
    METHODS[name] = imputer(name)

# CatBoost with 1000 trees and learning rate 0.03, to check whether ice_cb is weak only because of
# the hyperimpute defaults. hyperimpute only knows the learners in its lists of file paths
# (`regression.plugins`, `classifiers.plugins`), so we append ours there. Each file name must
# match the learner's name(), because hyperimpute looks learners up by file name.
LEARNERS = Path(__file__).parent / "learners"
regression.plugins.append(str(LEARNERS / "catboost_regressor_fixed.py"))
classifiers.plugins.append(str(LEARNERS / "catboost_fixed.py"))
# Kept out of METHODS, because the order of METHODS defines the job numbers cached in main.csv,
# and a new entry there would shift them and force a rerun.
EXTRA_METHODS = {"ice_cb_fixed": ice("catboost_regressor_fixed", "catboost_fixed")}

# Ablation rows with the settings of the authors' experiments_04_gain_of_function.ipynb.
# column_naive_paper follows Table 3: one model per column, chosen once with default settings.
EXTRA_METHODS |= {
    "column_naive_paper": imputer(
        optimizer="simple", select_model_by_column=True, select_model_by_iteration=False, select_lazy=True
    ),
    "global_search": imputer(
        optimizer="hyperband", select_model_by_column=False, select_model_by_iteration=False, select_lazy=True
    ),
    "wo_adaptivity": imputer(
        optimizer="hyperband", select_model_by_column=True, select_model_by_iteration=False, select_lazy=True
    ),
    # the authors' seeds_only_cb, which also selects globally and once, unlike Table 3's description
    "wo_flexibility_cb": imputer(
        optimizer="hyperband",
        regression_seed=["catboost_regressor"],
        classifier_seed=["catboost"],
        select_model_by_column=False,
        select_model_by_iteration=False,
        select_lazy=True,
    ),
}

# column_naive_paper with the fixed CatBoost in its pool, so that ice_cb_fixed is compared with a
# column_naive that has the same CatBoost available
EXTRA_METHODS["column_naive_cb_fixed"] = imputer(
    optimizer="simple",
    regression_seed=["random_forest_regressor", "linear_regression", "xgboost_regressor", "catboost_regressor_fixed"],
    classifier_seed=["random_forest", "logistic_regression", "xgboost", "catboost_fixed"],
    select_model_by_column=True,
    select_model_by_iteration=False,
    select_lazy=True,
)


def make_missing(dataset, seed, mechanism=cfg.MECHANISM, columns=None):
    """This function returns the scaled true data, a copy with missing cells and the mask of those cells"""
    enable_reproducible_results(seed)
    data = load(dataset)
    if columns is not None:
        data = data[columns]
    truth, with_missing, mask = ampute(
        scale_data(data), mechanism, cfg.MISSING_RATE, column_limit=cfg.COLUMN_LIMIT
    )
    return truth, with_missing, mask.to_numpy() == 1


def impute(method, seed, with_missing):
    return (METHODS | EXTRA_METHODS)[method](seed).fit_transform(with_missing).to_numpy()


def variance_ratio(imputed, truth, mask):
    """Std of imputed vs true values in the missing cells, averaged over columns."""
    ratios = []
    for column in range(truth.shape[1]):
        missing = mask[:, column]
        if missing.sum() > 1:
            ratios.append(imputed[missing, column].std() / truth[missing, column].std())
    return np.mean(ratios)


def evaluate(dataset, seed, methods, mechanism=cfg.MECHANISM, columns=None, missing=None):
    """missing: (truth, with_missing, mask) saved earlier, so that another hyperimpute version sees the same masks"""
    truth, with_missing, mask = make_missing(dataset, seed, mechanism, columns) if missing is None else missing
    rows = []
    for method in methods:
        start = time.time()
        imputed = impute(method, seed, with_missing)
        row = {
            "dataset": dataset,
            "seed": seed,
            "mechanism": mechanism,
            "n_features": truth.shape[1],
            "method": method,
            "seconds": time.time() - start,
            "rmse": RMSE(imputed, truth.to_numpy(), mask),
            "wasserstein": ws_score(imputed, truth),
            "variance_ratio": variance_ratio(imputed, truth.to_numpy(), mask),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_cached(name, jobs):
    """Run the jobs and append each result to results/<name>.csv as soon as it is finished.

    Jobs that are already in the file are skipped, so an interrupted run continues where it stopped.
    """
    path = cfg.RESULTS / f"{name}.csv"
    cfg.RESULTS.mkdir(exist_ok=True)
    done = set(pd.read_csv(path)["job"]) if path.exists() else set()
    todo = [number for number in range(len(jobs)) if number not in done]
    finished = Parallel(n_jobs=cfg.N_JOBS, return_as="generator")(jobs[number] for number in todo)
    progress = tqdm(zip(todo, finished), total=len(jobs), initial=len(done), desc=name)
    for number, result in progress:
        result.assign(job=number).to_csv(path, mode="a", header=not path.exists(), index=False)
    return pd.read_csv(path).sort_values("job", ignore_index=True)


def main_experiment():
    jobs = []
    for dataset in cfg.DATASETS:
        for seed in cfg.SEEDS:
            for method in METHODS:
                jobs.append(delayed(evaluate)(dataset, seed, [method]))
    return run_cached("main", jobs)


def mechanism_experiment():
    methods = ["mean", *cfg.COMPARED]
    jobs = []
    for dataset in cfg.DATASETS:
        for seed in cfg.SEEDS_EXTRA:
            for mechanism in cfg.MECHANISMS:
                jobs.append(delayed(evaluate)(dataset, seed, methods, mechanism))
    return run_cached("mechanisms", jobs)


def cb_fixed_experiment(n_seeds=len(cfg.SEEDS)):
    """ICE with CatBoost at sensible settings. Seeds are the outer loop, so running fewer seeds keeps job numbers."""
    jobs = []
    for seed in cfg.SEEDS[:n_seeds]:
        for dataset in cfg.DATASETS:
            jobs.append(delayed(evaluate)(dataset, seed, ["ice_cb_fixed"]))
    return run_cached("ice_cb_fixed", jobs)


def ablation_experiment(n_seeds=len(cfg.SEEDS)):
    """The remaining Table 2 rows. Seeds are the outer loop, so a partial run covers every dataset and method."""
    jobs = []
    for seed in cfg.SEEDS[:n_seeds]:
        for dataset in cfg.DATASETS:
            for method in cfg.ABLATION:
                jobs.append(delayed(evaluate)(dataset, seed, [method]))
    return run_cached("ablation_extra", jobs)


def cb_pool_experiment(n_seeds=len(cfg.SEEDS)):
    """column_naive with the fixed CatBoost in its pool, the fair comparison for ice_cb_fixed."""
    jobs = []
    for seed in cfg.SEEDS[:n_seeds]:
        for dataset in cfg.DATASETS:
            jobs.append(delayed(evaluate)(dataset, seed, ["column_naive_cb_fixed"]))
    return run_cached("column_naive_cb_fixed", jobs)


def selections(method, seed, with_missing):
    """Learner and hyperparameters that each column uses in the final iteration."""
    model = (METHODS | EXTRA_METHODS)[method](seed)
    model.fit_transform(with_missing)
    rows = []
    for column, learner in model.models().items():
        # each column's optimizer keeps the learner and hyperparameters it chose last ({} = defaults).
        # With select_lazy a column may instead copy the model of an earlier column with the same learner,
        # and then its own optimizer was never asked, so its record does not describe the model.
        candidate = model.model.column_to_optimizer[column].candidate
        params = str(candidate["params"]) if candidate["name"] == learner.name() else "copied from another column"
        rows.append({"method": method, "column": column, "learner": learner.name(), "params": params})
    return pd.DataFrame(rows)


def random_columns(dataset, k, seed):
    all_columns = load(dataset).columns
    return list(np.random.RandomState(100 + seed).choice(all_columns, k, replace=False))


def feature_experiment():
    jobs = []
    for k in cfg.FEATURE_COUNTS:
        for seed in cfg.SEEDS_EXTRA:
            columns = random_columns("wine_white", k, seed)
            jobs.append(delayed(evaluate)("wine_white", seed, cfg.COMPARED, columns=columns))
    return run_cached("features", jobs)


def density_example(seed=0):
    """Imputed vs true values in the wine_white column with the most missing cells."""
    truth, with_missing, mask = make_missing("wine_white", seed)
    column = mask.sum(axis=0).argmax()
    missing = mask[:, column]
    values = {"true": truth.to_numpy()[missing, column]}
    for method in ["hyperimpute_naive", "sklearn_missforest"]:
        values[method] = impute(method, seed, with_missing)[missing, column]
    return truth.columns[column], values


def paper_comparison(results):
    """Our RMSE and WD (x 10) next to Tables 2 and 7 of the paper.

    z = (ours - paper) / paper_std compares with the spread of single runs, z_se = (ours - paper) / paper_se
    with the uncertainty of the paper's mean over 10 seeds, paper_se = paper_std / sqrt(10).
    """
    columns = ["dataset", "paper_method", "paper", "paper_std"]
    paper = pd.concat([
        pd.DataFrame(cfg.PAPER_TABLE2, columns=columns).assign(metric="rmse"),
        pd.DataFrame(cfg.PAPER_TABLE7, columns=columns).assign(metric="wasserstein"),
    ])
    long = results.melt(["dataset", "method"], ["rmse", "wasserstein"], "metric")
    ours = 10 * long.groupby(["dataset", "method", "metric"]).value.agg(ours="mean", ours_std="std")
    ours = ours.reset_index()
    ours["paper_method"] = ours.method.replace(cfg.PAPER_NAMES)
    table = paper.merge(ours)
    table["diff_%"] = 100 * (table.ours / table.paper - 1)
    table["z"] = (table.ours - table.paper) / table.paper_std
    table["paper_se"] = table.paper_std / np.sqrt(len(cfg.SEEDS))
    table["z_se"] = (table.ours - table.paper) / table.paper_se
    order = ["metric", "dataset", "method", "paper_method", "paper", "paper_std", "paper_se", "ours", "ours_std",
             "diff_%", "z", "z_se"]
    return table[order]


def table2(results):
    return paper_comparison(results).query("metric == 'rmse'").drop(columns="metric").reset_index(drop=True)


def paired_test(results, method, baselines):
    """RMSE (x 10) of method minus baseline per seed, on the same masks, for each dataset.

    The baseline is the one of `baselines` with the lowest mean RMSE on that dataset. Returns the mean
    difference, a 95% t interval and the Wilcoxon signed rank p value (with 10 seeds at least 0.002).
    """
    rmse = 10 * results.pivot_table("rmse", ["dataset", "seed"], "method")
    rows = []
    for dataset in cfg.DATASETS:
        baseline = rmse.loc[dataset][baselines].mean().idxmin()
        diff = (rmse.loc[dataset][method] - rmse.loc[dataset][baseline]).dropna()
        half = stats.t.ppf(0.975, len(diff) - 1) * diff.std() / np.sqrt(len(diff))
        rows.append({
            "dataset": dataset,
            "method": method,
            "baseline": baseline,
            "seeds": len(diff),
            "mean_diff": diff.mean(),
            "ci_low": diff.mean() - half,
            "ci_high": diff.mean() + half,
            "p_wilcoxon": stats.wilcoxon(diff).pvalue,
        })
    return pd.DataFrame(rows)


def paired_tests(results):
    return pd.concat([paired_test(results, method, baselines) for method, baselines in cfg.PAIRED], ignore_index=True)


if __name__ == "__main__":
    results = pd.concat([main_experiment(), cb_fixed_experiment()])
    table2(results).round(3).to_csv(cfg.RESULTS / "table2_comparison.csv", index=False)
    paper_comparison(results).round(3).to_csv(cfg.RESULTS / "paper_comparison.csv", index=False)
    mechanism_experiment()
    feature_experiment()
    extended = pd.concat([results, ablation_experiment(), cb_pool_experiment()])
    table2(extended).round(3).to_csv(cfg.RESULTS / "table2_extended.csv", index=False)
    paired_tests(extended).round(4).to_csv(cfg.RESULTS / "paired_tests.csv", index=False)
