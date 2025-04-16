from tabpfn import TabPFNRegressor

class MLTabPFN():
    def __init__(self):
        self.model = TabPFNRegressor()

    def fit(self, X_train, y_train):
        print("Fitting TabPFN...")
        self.model.fit(X_train, y_train)
        print("Fitting TabPFN Done!")

    def predict(self, X):
        return self.model.predict(X)

    def __call__(self, X):
        return self.predict(X)  # 사용 시: model(X)