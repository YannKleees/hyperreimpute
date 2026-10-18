from hyperimpute.plugins.prediction.classifiers.plugin_catboost import CatBoostPlugin


# We subclass hyperimpute's CatBoost classifier instead of passing settings to it, because ice()
# picks learners by name only, so the settings have to be part of a learner of their own.
class CatBoostFixed(CatBoostPlugin):
    """catboost classifier with sensible settings instead of the 0.1.17 defaults (10 trees, learning rate 1e-3)."""

    # 1000 trees is CatBoost's own default and 0.03 a usual learning rate for it. The hyperimpute
    # defaults move the predictions so little that ice_cb ends up as mean imputation.
    def __init__(self, n_estimators=1000, learning_rate=0.03, **settings):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, **settings)

    # hyperimpute looks a learner up by its file name and then registers it under name(),
    # so the two have to match. A new name also keeps the original "catboost" for ice_cb.
    @staticmethod
    def name():
        return "catboost_fixed"


# hyperimpute loads the class from a module variable that must be called `plugin`
plugin = CatBoostFixed
