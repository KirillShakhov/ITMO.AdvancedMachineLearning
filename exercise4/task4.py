import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Загрузка обучающего и тестового датасетов
train_data = pd.read_csv('data/titanic_train.csv')
test_data = pd.read_csv('data/titanic_reserved.csv')

# Подготовка данных
# Преобразование категориальных признаков, заполнение пропущенных значений, и т. д.

# Объединение обучающего и тестового датасетов для обработки признака "sex"
combined_data = pd.concat([train_data, test_data], axis=0)

# Применяем LabelEncoder к столбцу "sex" для создания числового признака
label_encoder = LabelEncoder()
combined_data['sex_encoded'] = label_encoder.fit_transform(combined_data['sex'])

# Применяем OneHotEncoder к числовому признаку "sex_encoded"
one_hot_encoder = OneHotEncoder()
sex_encoded = one_hot_encoder.fit_transform(combined_data[['sex_encoded']]).toarray()

# Создаем DataFrame на основе OneHotEncoder
sex_encoded_df = pd.DataFrame(sex_encoded, columns=['sex_female', 'sex_male'])

# Объединяем DataFrame с данными и DataFrame с OneHot-кодированным признаком
combined_data = pd.concat([combined_data, sex_encoded_df], axis=1)

# Разделение обработанных данных обратно на обучающий и тестовый датасеты
train_data = combined_data[:len(train_data)]
test_data = combined_data[len(train_data):]

# Разделение обучающего датасета на признаки (X_train) и целевую переменную (y_train)
X_train = train_data.drop(columns=['survived', 'name', 'sex', 'sex_encoded'])
y_train = train_data['survived']

# Создание и обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Подготовка данных тестового датасета
# Преобразование категориальных признаков, заполнение пропущенных значений, и т. д.

# Извлечение истинных меток из тестового датасета
true_labels = test_data['survived']

# Выполнение предсказаний на тестовом датасете
X_test = test_data.drop(columns=['survived', 'name', 'sex', 'sex_encoded'])  # Здесь важно, чтобы X_test был подготовлен аналогично X_train
predictions = model.predict(X_test)

# Рассчет f1_score
f1 = f1_score(true_labels, predictions)

print("F1 Score:", f1)
