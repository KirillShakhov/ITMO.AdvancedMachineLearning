import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# загружаем данные
df = pd.read_csv('360T.csv', delimiter=',')

# разбиваем на тренировочный и тестовый наборы данных
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.3, random_state=65, stratify=df.iloc[:,-1])

# стандартизируем признаки тренировочных данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# обучаем MLPClassifier
clf = MLPClassifier(random_state=65, hidden_layer_sizes=(31, 10), activation='logistic', max_iter=1000)
clf.fit(X_train, y_train)

# делаем предсказания на тестовых данных
y_pred = clf.predict(X_test)

# вычисляем метрики
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Precision (macro avg):", precision)
print("Recall (macro avg):", recall)
print("F-score (macro avg):", f1)

# загружаем тестовые данные
X_test_new = pd.read_csv('DL_Task_2_test_file_166.csv', delimiter=',')

# стандартизируем тестовые данные с помощью scaler, полученного на тренировочных данных
X_test_new_scaled = scaler.transform(X_test_new)

# делаем предсказания на тестовых данных
y_pred = clf.predict(X_test_new_scaled)

print("Predictions:", y_pred)