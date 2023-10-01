import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Загрузите ваш обучающий датасет
train_data = pd.read_csv('data/task4_fish_train.csv')
# Предполагаем, что ваши признаки находятся в колонках 'feature1', 'feature2', ..., а метки в колонке 'target'
X_train = train_data[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y_train = train_data['Weight']

# Загрузите ваш тестовый датасет
test_data = pd.read_csv('data/task5_fish_reserved.csv')
X_test = test_data[['Length1', 'Length2', 'Length3', 'Height', 'Width']]

# Создайте и обучите модель
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Сделайте предсказания
predictions = model.predict(X_test)

# Выведите предсказания в нужном формате
result = predictions.tolist()
print(result)
