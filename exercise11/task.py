import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score

import tensorflow_datasets as tfds

from tensorflow.python.keras import regularizers

ds_train_tf, ds_validation_tf, ds_test_tf = tfds.load(
    name='titanic',
    split=['train[:70%]', 'train[70%:80%]', 'train[80%:90%]'],
    as_supervised=True
)

# Предобработка данных
# Загрузка данных
df_train = tfds.as_dataframe(ds_train_tf, ds_info=tfds.load('titanic', split='train[:70%]').info)
df_validation = tfds.as_dataframe(ds_validation_tf)
df_test = tfds.as_dataframe(ds_test_tf)

# Объединение тренировочной и валидационной выборки
df_train = pd.concat([df_train, df_validation], ignore_index=True)

# Заполнение пропущенных значений
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_train['Fare'].fillna(df_train['Fare'].median(), inplace=True)

df_test['Age'].fillna(df_train['Age'].median(), inplace=True)
df_test['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test['Fare'].fillna(df_train['Fare'].median(), inplace=True)

# Приведение столбцов к единому виду
df_train['Sex'] = df_train['Sex'].map({'male': 0, 'female': 1})
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Удаление ненужных столбцов
df_train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Удаление nan и infinity значений
df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
df_train.dropna(inplace=True)

df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test.dropna(inplace=True)

# Нормализация данных относительно среднего и медианы
scaler = StandardScaler()
df_train[['Age', 'Fare']] = scaler.fit_transform(df_train[['Age', 'Fare']])
df_test[['Age', 'Fare']] = scaler.transform(df_test[['Age', 'Fare']])

# Инженерия признаков
# Добавление признака "Семейный размер"
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_train['IsAlone'] = 0
df_train.loc[df_train['FamilySize'] == 1, 'IsAlone'] = 1

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
df_test['IsAlone'] = 0
df_test.loc[df_test['FamilySize'] == 1, 'IsAlone'] = 1

# Обучение модели
# Выбор модели
model = Sequential()
model.add(Dense(64, activation='relu',
                kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# Обучение модели
history = model.fit(ds_train_tf, epochs=50, validation_data=ds_validation_tf)

# Оценка модели на тестовом наборе данных
test_loss, test_accuracy = model.evaluate(ds_test_tf)

# Вывод метрик на тестовом наборе данных
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Предсказание для данных из зарезервированной части
df_reserved = pd.read_csv('DL_Task_3_Titain_reserved.csv')

# Предобработка данных
df_reserved['Age'].fillna(df_reserved['Age'].median(), inplace=True)
df_reserved['Embarked'].fillna(df_reserved['Embarked'].mode()[0], inplace=True)
df_reserved.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

df_reserved['Sex'] = df_reserved['Sex'].apply(lambda x: 1 if x == 'male' else 0)
df_reserved['Embarked'] = df_reserved['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Нормализация данных
df_reserved = (df_reserved - df_reserved.mean()) / df_reserved.std()

# Предсказание вероятности спасения
predictions = model.predict(df_reserved)

# Конвертация вероятностей в метки классов (0 или 1)
predictions_classes = [1 if p > 0.5 else 0 for p in predictions]

# Вывод меток классов
print(predictions_classes)
