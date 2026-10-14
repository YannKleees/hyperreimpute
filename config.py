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
    ("airfoil", "column_naive", 1.861, 0.446),
    ("airfoil", "hyperimpute", 1.479, 0.294),
    ("compression", "ice_lr", 1.429, 0.215),
    ("compression", "ice_rf", 1.615, 0.141),
    ("compression", "ice_cb", 1.251, 0.204),
    ("compression", "column_naive", 1.094, 0.167),
    ("compression", "hyperimpute", 1.013, 0.145),
    ("wine_white", "ice_lr", 0.919, 0.060),
    ("wine_white", "ice_rf", 0.987, 0.034),
    ("wine_white", "ice_cb", 0.877, 0.054),
    ("wine_white", "column_naive", 0.845, 0.081),
    ("wine_white", "hyperimpute", 0.801, 0.053),
]

# Table 7 of the paper: Wasserstein distance x 10, MAR 30%, mean and std over 10 seeds
PAPER_TABLE7 = [
    ("airfoil", "ice_lr", 1.066, 0.307),
    ("airfoil", "ice_rf", 1.135, 0.307),
    ("airfoil", "ice_cb", 0.790, 0.272),
    ("airfoil", "column_naive", 0.516, 0.278),
    ("airfoil", "hyperimpute", 0.291, 0.103),
    ("compression", "ice_lr", 0.781, 0.219),
    ("compression", "ice_rf", 1.010, 0.040),
    ("compression", "ice_cb", 0.537, 0.053),
    ("compression", "column_naive", 0.315, 0.030),
    ("compression", "hyperimpute", 0.289, 0.041),
    ("wine_white", "ice_lr", 0.584, 0.093),
    ("wine_white", "ice_rf", 0.648, 0.047),
    ("wine_white", "ice_cb", 0.491, 0.093),
    ("wine_white", "column_naive", 0.390, 0.172),
    ("wine_white", "hyperimpute", 0.426, 0.141),
]

# our method -> paper row it is compared with.
# hyperimpute_naive also keeps per iteration selection on (select_model_by_iteration=True),
# so it is column_naive plus adaptivity.
PAPER_NAMES = {"hyperimpute_naive": "column_naive", "ice_cb_fixed": "ice_cb"}
