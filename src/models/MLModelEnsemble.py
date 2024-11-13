from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class MLModelEnsemble:
    def __init__(self, configs):
        # Define available models with verbose if supported
        available_models = {
            'ridge': Ridge(alpha=configs['ridge_alpha']),  # Ridge doesn't support verbose
            'rf': RandomForestRegressor(
                n_estimators=configs['rf_n_estimators'],
                max_depth=configs['rf_max_depth'],
                verbose=1  # Enable logging for RandomForestRegressor
            ),
            'svr': SVR(
                kernel=configs['svr_kernel'],
                C=configs['svr_C']
                # SVR doesn't support verbose directly
            ),
            'dt': DecisionTreeRegressor(
                max_depth=configs['dt_max_depth']
                # DecisionTree doesn't support verbose
            ),
            'gpr': GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
                alpha=1e-10,
                n_restarts_optimizer=0
                # GaussianProcessRegressor doesn't support verbose
            )
        }

        # Select models based on configuration
        selected_models = configs['selected_models']
        self.models = [(name, model) for name, model in available_models.items() if name in selected_models]
        
        # Initialize VotingRegressor
        self.model = VotingRegressor(estimators=self.models)
        
    def fit(self, X, y):
        print("Starting VotingRegressor fit...")
        
        for name, model in self.models:
            print(f"Fitting model: {name}")
            model.fit(X, y)  # Individual model's fit
            print(f"Model {name} fitting completed.")
        
        print("Fitting VotingRegressor...")
        self.model.fit(X, y)  # Fit the VotingRegressor
        print("VotingRegressor fit completed.")