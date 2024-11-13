import torch
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class MLModelEnsemble:
    def __init__(self, configs):
        available_models = {
            'ridge': Ridge(alpha=configs['ridge_alpha']),
            'rf': RandomForestRegressor(
                n_estimators=configs['rf_n_estimators'],
                max_depth=configs['rf_max_depth']),
            'svr': SVR(
                kernel=configs['svr_kernel'],
                C=configs['svr_C']),
            'dt': DecisionTreeRegressor(
                max_depth=configs['dt_max_depth']),
            'gpr': GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
                alpha=1e-10,
                n_restarts_optimizer=0)
        }

        selected_models = configs['selected_models']
        self.models = [(name, model) for name, model in available_models.items() if name in selected_models]
        
        self.model = VotingRegressor(estimators=self.models)