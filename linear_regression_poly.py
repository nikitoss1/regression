import numpy as np

class LinearRegressionPoly:

    def __init__(self, alpha=.001, n_iterations=1000):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.mse_list = []

    def mse(self, y, y_pred):
        return (1/len(y))*(y - y_pred) ** 2
    
    def fit(self, X, Y):
        # Добавляем колонку единиц для свободного члена
        self.X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.Y = np.array(Y)

        # Добавляем вектор theta с рандомными значениями
        self.theta = np.random.rand(self.X.shape[1])

        for _ in range(self.n_iterations):
            # Вычисляем предсказание
            pred = self.X @ self.theta

            # Вычисляем значение ошибки
            self.mse_list.append(self.mse(self.Y, pred))

            # Вычисляем ошибку
            errors = self.Y - pred

            # Находим градиент
            grad = -(2/self.X.shape[0]) * (self.X.T @ errors)

            # Вычитаем из текущих theta значение градиента, умноженного на альфа
            self.theta -= (self.alpha * grad)
    
    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self.X @ self.theta
