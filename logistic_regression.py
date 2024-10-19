import numpy as np

class LogisticRegressionPoly:

    def __init__(self, alpha=.001, n_iterations=1000):
        self.alpha=alpha
        self.n_iterations = n_iterations

    def __sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)

        # Добавим столбец единиц для свободного члена
        self.X = np.hstack([np.ones((self.X.shape[0], 1)), self.X])

        # Задаем рандомные значения весов
        self.theta = np.random.rand(len(self.X[0]))

        for _ in range(self.n_iterations):
            # вычисляем линейную комбинацию
            lin_comb = self.X @ self.theta

            # Вычисляем вероятность получения 1 при наших данных,
            # используя сигмоидальную функцию
            # (тут по сути предсказания)
            P = []
            for i in range(len(lin_comb)):
                P.append(self.__sigmoid(lin_comb[i]))
            P = np.array(P)

            # Вычисление функции ошибки (функции потерь)
            L = -(self.Y * np.log(P) + (1 - self.Y) * np.log(1 - P)).mean()

            # Вычисляем градиент
            errors = P - self.Y

            grad = self.X.T @ errors

            # Обновляем веса
            self.theta -= (self.alpha * grad)

    def predict(self, X, threshold=0.6):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        lin_comb = X @ self.theta
        P = self.__sigmoid(lin_comb) 
        return (P >= threshold).astype(int)
