import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Данные
data = {
    'X': [22, 19, 11, 7, 13, 20, 8, 12, 15, 23],
    'Y': [45, 42, 23, 23, 23, 39, 19, 21, 28, 65]
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Вычисляем средние значения
mean_X = df['X'].mean()
mean_Y = df['Y'].mean()

# Обучаем модель линейной регрессии
model = LinearRegression()
model.fit(df[['X']], df['Y'])

# Коэффициенты модели
b1 = model.coef_[0]
b0 = model.intercept_

# Прогнозируем значения
predictions = model.predict(df[['X']])

# Вычисляем R^2
r2 = r2_score(df['Y'], predictions)

mean_X, mean_Y, round(b1, 2), round(b0, 2), round(r2, 2)

# Выводим результаты
print(f'1. Выборочное среднее X: {mean_X}')
print(f'2. Выборочное среднее Y: {mean_Y}')
print(f'3. Коэффициент b1: {round(b1, 2)}')
print(f'4. Коэффициент b0: {round(b0, 2)}')
print(f'5. Статистика R^2: {round(r2, 2)}')
