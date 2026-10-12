from pathlib import Path

RESULTS = Path("results")

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"
URLS = {
    "airfoil": f"{UCI}/00291/airfoil_self_noise.dat",
    "compression": f"{UCI}/concrete/compressive/Concrete_Data.xls",
    "wine_white": f"{UCI}/wine-quality/winequality-white.csv",
}
CSV_OPTIONS = {"airfoil": dict(sep="\t", header=None), "wine_white": dict(sep=";")}
DATASETS = list(URLS)

# main setting of the paper: 30% missing at random, 10 seeds
MECHANISM, MISSING_RATE = "MAR", 0.3
SEEDS = range(10)
COLUMN_LIMIT = 10  # columns that may receive missing values, as in the authors' code

# limitation tests: fewer seeds, three methods
SEEDS_EXTRA = range(5)
MECHANISMS = ["MCAR", "MAR", "MNAR"]
FEATURE_COUNTS = [2, 3, 4, 6, 8, 12]
COMPARED = ["sklearn_ice", "sklearn_missforest", "hyperimpute_naive"]

N_JOBS = 2

# Table 2 of the paper: RMSE x 10, MAR 30%, mean and std over 10 seeds
PAPER_TABLE2 = [
    ("airfoil", "ice_lr", 2.349, 0.483),
    ("airfoil", "ice_rf", 2.365, 0.533),
    ("airfoil", "ice_cb", 2.078, 0.482),
    ("airfoil", "hyperimpute", 1.479, 0.294),
    ("compression", "ice_lr", 1.429, 0.215),
    ("compression", "ice_rf", 1.615, 0.141),
    ("compression", "ice_cb", 1.251, 0.204),
    ("compression", "hyperimpute", 1.013, 0.145),
    ("wine_white", "ice_lr", 0.919, 0.060),
    ("wine_white", "ice_rf", 0.987, 0.034),
    ("wine_white", "ice_cb", 0.877, 0.054),
    ("wine_white", "hyperimpute", 0.801, 0.053),
]
