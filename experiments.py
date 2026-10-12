from hyperimpute.utils.benchmarks import *  # first, so the imports below win (it exports its own `time`)
import time
import warnings
from functools import cache
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import config as cfg
warnings.filterwarnings("ignore")  # hyperimpute floods the output with warnings
assert pd.__version__ < "2.3", "with pandas>=2.3 every imputer becomes mean imputation"


@cache
def load(dataset):
    """Download a UCI dataset. The target column is kept, as in the authors notebooks"""
    url = cfg.URLS[dataset]
    if url.endswith(".xls"):
        return pd.read_excel(url)
    return pd.read_csv(url, **cfg.CSV_OPTIONS[dataset])


def imputer(name="hyperimpute", **settings):
    return lambda seed: Imputers().get(name, random_state=seed, **settings)


def ice(regressor, classifier):
    """HyperImpute with one fixed learner for all columns, as in the authors' notebook."""
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
    "hyperimpute_naive": imputer(optimizer="simple"),
    "ice_lr": ice("linear_regression", "logistic_regression"),
    "ice_rf": ice("random_forest_regressor", "random_forest"),
    "ice_cb": ice("catboost_regressor", "catboost"),
}
for name in ["sklearn_ice", "sklearn_missforest", "mean", "softimpute", "gain", "miwae"]:
    METHODS[name] = imputer(name)


def make_missing(dataset, seed, mechanism=cfg.MECHANISM, columns=None):
    """Return the scaled true data, a copy with missing cells, and the mask of those cells."""
    enable_reproducible_results(seed)
    data = load(dataset)
    if columns is not None:
        data = data[columns]
    truth, with_missing, mask = ampute(
        scale_data(data), mechanism, cfg.MISSING_RATE, column_limit=cfg.COLUMN_LIMIT
    )
    return truth, with_missing, mask.to_numpy() == 1


def impute(method, seed, with_missing):
    return METHODS[method](seed).fit_transform(with_missing).to_numpy()


def variance_ratio(imputed, truth, mask):
    """Std of imputed vs true values in the missing cells, averaged over columns."""
    ratios = []
    for column in range(truth.shape[1]):
        missing = mask[:, column]
        if missing.sum() > 1:
            ratios.append(imputed[missing, column].std() / truth[missing, column].std())
    return np.mean(ratios)


def evaluate(dataset, seed, methods, mechanism=cfg.MECHANISM, columns=None):
    truth, with_missing, mask = make_missing(dataset, seed, mechanism, columns)
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
            for method in METHODS:  # one job per method, so the progress bar moves often
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


def table2(main):
    paper = pd.DataFrame(cfg.PAPER_TABLE2, columns=["dataset", "method", "paper", "paper_std"])
    ours = 10 * main.groupby(["dataset", "method"]).rmse.agg(ours="mean", ours_std="std")
    table = paper.merge(ours.reset_index())
    table["diff_%"] = 100 * (table.ours / table.paper - 1)
    return table


if __name__ == "__main__":
    table2(main_experiment()).round(3).to_csv(cfg.RESULTS / "table2_comparison.csv", index=False)
    mechanism_experiment()
    feature_experiment()
