import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Создайте массивы данных для X и Y:
X = np.array([5, 17, 9, 13, 12, 7, 18, 20, 22, 24]).reshape(-1, 1)
Y = np.array([12, 45, 18, 25, 23, 18, 43, 36, 66, 78])

# Обучите модель линейной регрессии:
model = LinearRegression()
model.fit(X, Y)

# Определите выборочное среднее для X и Y:
mean_X = np.mean(X)
mean_Y = np.mean(Y)

# Найдите коэффициенты O1 (наклон) и O2 (пересечение) модели:
O1 = model.coef_[0]
O2 = model.intercept_

# Оцените точность модели, вычислив R^2 статистику:
Y_pred = model.predict(X)
r_squared = r2_score(Y, Y_pred)

# Выведите результаты:
print("Выборочное среднее X:", mean_X)
print("Выборочное среднее Y:", mean_Y)
print("Коэффициент O1:", O1)
print("Коэффициент O2:", O2)
print("R^2 статистика:", r_squared)
