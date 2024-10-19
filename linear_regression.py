import numpy as np

class MyLinearRegression:
    def __init__(self, alpha=.001, n_iterations=1000):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.mse_list = []

    def gradient(self, errors):
        return -(2/len(errors))*((errors).sum())
    
    def mse(self, y, y_pred):
        y = np.array(y)
        y_pred = np.array(y_pred)
        return ((1/len(y))*(y - y_pred)**2).mean()
    
    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        self.theta0 = np.random.rand()
        self.theta1 = np.random.rand()

        for _ in range(self.n_iterations):
            # Тут в пред много значений числом len(X)
            pred = self.theta0 + self.theta1 * X

            # Тоже много знач
            errors = Y - pred

            self.theta0 -= (self.alpha*self.gradient(errors, 1))
            self.theta1 -= (self.alpha*self.gradient(errors, X))

            self.mse_list.append(self.mse(Y, pred))

    def predict(self, X):
        X = np.array(X)
        return self.theta0 + self.theta1 * X
    




