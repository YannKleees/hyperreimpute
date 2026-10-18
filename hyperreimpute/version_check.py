"""Rerun hyperimpute, column_naive_paper and ice_cb under hyperimpute 0.1.5, on the masks of 0.1.17.

Run `python -m hyperreimpute.version_check` once in the main environment (saves the masks and the 0.1.17
selections), then once in venv_015 (writes results/version_015.csv and the 0.1.5 selections). See README.
"""
from importlib.metadata import version
import numpy as np
import pandas as pd
from joblib import delayed
from . import config as cfg
from . import experiments as ex

VERSION = version("hyperimpute")
DATASETS = ["airfoil", "compression"]
SEEDS = range(5)
METHODS = ["hyperimpute", "column_naive_paper", "ice_cb"]
MASKS = cfg.RESULTS / "masks"


def mask_path(dataset, seed):
    return MASKS / f"{dataset}_{seed}.npz"


def save_masks():
    """Save the 0.1.17 masks, so that 0.1.5 imputes exactly the same cells even if its numpy or torch draw differently."""
    MASKS.mkdir(exist_ok=True)
    for dataset in DATASETS:
        for seed in SEEDS:
            truth, with_missing, mask = ex.make_missing(dataset, seed)
            np.savez(mask_path(dataset, seed), truth=truth.to_numpy(), mask=mask, columns=truth.columns.to_numpy(str))


def load_masks(dataset, seed):
    saved = np.load(mask_path(dataset, seed))
    truth = pd.DataFrame(saved["truth"], columns=saved["columns"])
    return truth, truth.mask(saved["mask"]), saved["mask"]


def same_masks():
    """Whether this version's own make_missing draws the masks that were saved under 0.1.17."""
    return all(
        (ex.make_missing(dataset, seed)[2] == load_masks(dataset, seed)[2]).all()
        for dataset in DATASETS
        for seed in SEEDS
    )


def version_experiment():
    """Seeds are the outer loop, so a partial run covers both datasets and all three methods."""
    jobs = []
    for seed in SEEDS:
        for dataset in DATASETS:
            for method in METHODS:
                jobs.append(delayed(ex.evaluate)(dataset, seed, [method], missing=load_masks(dataset, seed)))
    return ex.run_cached("version_015", jobs)


def selection_log():
    """Append this version's selections on airfoil seed 0 to results/selections.csv, unless they are already there."""
    path = cfg.RESULTS / "selections.csv"
    if path.exists() and VERSION in set(pd.read_csv(path, dtype={"version": str}).version):
        return
    with_missing = load_masks("airfoil", 0)[1]
    rows = pd.concat([ex.selections(method, 0, with_missing) for method in ["hyperimpute", "column_naive_paper"]])
    rows.insert(0, "version", VERSION)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    if VERSION == "0.1.5":
        print("0.1.5 draws the same masks as 0.1.17:", same_masks())
        version_experiment()
    else:
        save_masks()
    selection_log()
