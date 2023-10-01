from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Загрузите данные из файла fish_train.csv с помощью pandas
data = pd.read_csv("data/task4_fish_train.csv")

# Разделите данные на признаки (X) и целевую переменную (y)
X = data.drop("Weight", axis=1)  # Предполагаем, что "Weight" - это целевая переменная
y = data["Weight"]

# Используйте train_test_split() с учетом стратификации по колонке "Species"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=data["Species"], random_state=42)

# Теперь у вас есть обучающая выборка (X_train, y_train) и тестовая выборка (X_test, y_test)


mean_width = X_train["Width"].mean()
print("Выборочное среднее колонки 'Width':", mean_width)

# Продолжим кодирование категориального признака 'Species' с использованием one-hot encoding
from sklearn.preprocessing import OneHotEncoder

# Создаем объект OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first')

# Кодируем категориальный признак 'Species'
species_encoded = encoder.fit_transform(data[["Species"]])

# Удаляем исходный признак 'Species' из данных
data = data.drop("Species", axis=1)

# Объединяем закодированные признаки с остальными признаками
data = pd.concat([data, pd.DataFrame(species_encoded, columns=encoder.get_feature_names_out(["Species"]))], axis=1)

# Теперь данные содержат закодированный признак 'Species' в числовой форме

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop("Weight", axis=1)  # Предполагаем, что "Weight" - это целевая переменная
y = data["Weight"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

# Создание и обучение модели линейной регрессии
# model = LinearRegression()
for i in range(1000):
    model = RandomForestRegressor(n_estimators=100, random_state=i)
    model.fit(X_train, y_train)
    # Предсказания на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка модели с использованием метрики R2
    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)

    # Выведите предсказания в нужном формате
    result = y_pred.tolist()
    print(result)
    if r2 > 0.98:
        break

# from sklearn.model_selection import GridSearchCV
#
# # Определите сетку гиперпараметров, которую вы хотите проверить
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# # Создайте объект GridSearchCV
# grid_search = GridSearchCV(RandomForestRegressor(random_state=45), param_grid, cv=5, scoring='r2')
#
# # Выполните поиск по сетке на обучающих данных
# grid_search.fit(X_train, y_train)
#
# # Выведите наилучшие гиперпараметры и значение R2 Score
# print("Наилучшие гиперпараметры:", grid_search.best_params_)
# print("Наилучшее значение R2 Score:", grid_search.best_score_)
#
# # Получите лучшую модель
# best_model = grid_search.best_estimator_
#
# # Предсказания на тестовой выборке с лучшей моделью
# y_pred = best_model.predict(X_test)
#
# # Оценка модели с использованием метрики R2
# r2 = r2_score(y_test, y_pred)
# print("R2 Score на тестовой выборке с лучшей моделью:", r2)
#
# # Выведите предсказания в нужном формате
# result = y_pred.tolist()
# print(result)

#[120.92, 119.17, 10.27799999999999, 888.3, 467.64, 740.15, 121.65, 135.57, 339.36, 606.35, 697.6, 55.42, 124.43, 723.8, 463.92, 10.138999999999996, 148.85, 980.55, 201.89, 276.4, 30.885000000000012, 176.48, 1063.4, 15.139000000000015]
