from pathlib import Path

# next to this folder, so that the notebook and the scripts find the cache from any working directory
RESULTS = Path(__file__).parent.parent / "results"

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

# the remaining Table 2 rows of the paper, plus column_naive as Table 3 defines it
ABLATION = ["column_naive_paper", "global_search", "wo_adaptivity", "wo_flexibility_cb"]

# paired per seed tests: (method, baselines), where the best baseline per dataset is used
PAIRED = [
    ("column_naive_paper", ["ice_lr", "ice_rf"]),
    ("ice_cb_fixed", ["column_naive_paper"]),
    ("ice_cb_fixed", ["hyperimpute"]),
    ("hyperimpute", ["column_naive_paper"]),
    ("ice_cb_fixed", ["column_naive_cb_fixed"]),
]

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
    ("airfoil", "global_search", 1.599, 0.322),
    ("airfoil", "wo_flexibility_cb", 1.594, 0.367),
    ("airfoil", "wo_adaptivity", 1.665, 0.360),
    ("compression", "global_search", 1.031, 0.175),
    ("compression", "wo_flexibility_cb", 1.082, 0.157),
    ("compression", "wo_adaptivity", 1.047, 0.176),
    ("wine_white", "global_search", 0.853, 0.049),
    ("wine_white", "wo_flexibility_cb", 0.876, 0.050),
    ("wine_white", "wo_adaptivity", 0.890, 0.074),
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
PAPER_NAMES = {
    "hyperimpute_naive": "column_naive",
    "column_naive_paper": "column_naive",
    "column_naive_cb_fixed": "column_naive",
    "ice_cb_fixed": "ice_cb",
}
