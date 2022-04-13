from .propensityScoreMatching import PropensityScoreMatching
import numpy as np

class Metric():
    def __init__(self):
        self.ate = None
        self.pehe = None

    def get_pehe(self, ite_pred, ite_true):
        return np.mean(np.power((ite_pred - ite_true), 2))

    def get_epsilon_ate(self, ite_pred, ite_true):
        return np.absolute(np.mean(ite_pred - ite_true))

