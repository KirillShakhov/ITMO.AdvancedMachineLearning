import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # Импортирована библиотека для построения графиков
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
data = pd.read_csv('data/task4_fish_train.csv')

# Преобразование категориального признака "Species" в числовые значения с использованием Label Encoding
label_encoder = LabelEncoder()
data['Species'] = label_encoder.fit_transform(data['Species'])

# Разбиение набора данных на обучающую и тестовую выборки с учетом стратификации
X = data.drop(columns=['Weight'])
Y = data['Weight']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=data['Species'])

mean_width_train = X_train['Width'].mean()

# Построение базовой модели линейной регрессии
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
r2_baseline = r2_score(Y_test, Y_pred)

# Построение матрицы корреляций признаков
corr_matrix = X_train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Применение метода главных компонент (PCA)
pca = PCA(n_components=3, svd_solver='full')
pca.fit(X_train)
explained_variance_ratio = pca.explained_variance_ratio_[0]

# Замена трех наиболее коррелированных признаков на новый признак Lengths
X_train_pca = pca.transform(X_train)
X_train['Lengths'] = X_train_pca[:, 0]

# Применение преобразования PCA к тестовым данным и замена признаков
X_test_pca = pca.transform(X_test)
X_test['Lengths'] = X_test_pca[:, 0]

# Обучение модели линейной регрессии после преобразования PCA
model_pca = LinearRegression()
model_pca.fit(X_train, Y_train)
Y_pred_pca = model_pca.predict(X_test)
r2_pca = r2_score(Y_test, Y_pred_pca)

# Преобразование признаков в куб
X_train[['Width', 'Height', 'Lengths']] = X_train[['Width', 'Height', 'Lengths']] ** 3
mean_width_cubed = X_train['Width'].mean()

# Обучение модели линейной регрессии после преобразования признаков в куб
model_cubed = LinearRegression()
model_cubed.fit(X_train, Y_train)
Y_pred_cubed = model_cubed.predict(X_test)
r2_cubed = r2_score(Y_test, Y_pred_cubed)

# One-hot кодирование категориальных признаков
X_train_encoded = pd.get_dummies(X_train, columns=['Species'])
X_test_encoded = pd.get_dummies(X_test, columns=['Species'])

# Обучение модели линейной регрессии с кодированными признаками
model_encoded = LinearRegression()
model_encoded.fit(X_train_encoded, Y_train)
Y_pred_encoded = model_encoded.predict(X_test_encoded)
r2_encoded = r2_score(Y_test, Y_pred_encoded)

# One-hot кодирование категориальных признаков с drop_first=True
X_train_encoded_drop_first = pd.get_dummies(X_train, columns=['Species'], drop_first=True)
X_test_encoded_drop_first = pd.get_dummies(X_test, columns=['Species'], drop_first=True)

# Обучение модели линейной регрессии с кодированными признаками и drop_first
model_encoded_drop_first = LinearRegression()
model_encoded_drop_first.fit(X_train_encoded_drop_first, Y_train)
Y_pred_encoded_drop_first = model_encoded_drop_first.predict(X_test_encoded_drop_first)
r2_encoded_drop_first = r2_score(Y_test, Y_pred_encoded_drop_first)

print("Не работает!!!")
print("Вычислите выборочное среднее колонки Width полученной тренировочной выборки:", mean_width_train)
print("Введите r2_score() полученной модели:", r2_baseline)
print("Замените тройку наиболее коррелированных признаков на полученный признак Lengths, значения которого совпадают со счетами первой главной компоненты. Обучите модель линейной регрессии. Введите r2_score() полученной модели.:", r2_pca)
print("Используя полученный на предыдущем этапе набор данных, возведите в куб значения признаков Width, Height, Lengths. Введите выборочное среднее колонки Width тренировочного набора данных после возведения в куб:", mean_width_cubed)
print("Обучите модель линейнной регрессии. Введите r2_score() полученной модели:", r2_cubed)
print("Добавьте к набору данных, полученному на предыдущем этапе, ранее исключенные категориальные признаки, предварительно произведя one-hot кодирование при помощи pd.get_dummies(). Обучите модель регрессии. Введите r2_score() полученной модели:", r2_encoded)
print("Закодируйте категориальные признаки при помощи pd.get_dummies(drop_first=True). Введите r2_score() модели после избавления от коррелированности:", r2_encoded_drop_first)
