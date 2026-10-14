from hyperimpute.plugins.prediction.regression.plugin_catboost_regressor import CatBoostRegressorPlugin


class CatBoostRegressorFixed(CatBoostRegressorPlugin):
    """catboost_regressor with sensible settings instead of the 0.1.17 defaults (10 trees, learning rate 1e-3)."""

    def __init__(self, n_estimators=1000, learning_rate=0.03, **settings):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, **settings)

    @staticmethod
    def name():
        return "catboost_regressor_fixed"


plugin = CatBoostRegressorFixed
