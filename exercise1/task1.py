import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Чтение данных из файла
data = np.genfromtxt('task1_data.csv', delimiter=',')

# Создание объекта PCA с 2 главными компонентами
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)

# Вычисление координаты первого объекта относительно первой и второй главных компонент
first_object_coordinates = principal_components[0]

# Вычисление доли объясненной дисперсии при использовании первых двух главных компонент
explained_variance_ratio = sum(pca.explained_variance_ratio_)

# Определение минимального количества главных компонент для достижения доли объясненной дисперсии более 0.85
min_components_for_variance = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.85)

# Кластеризация с использованием KMeans
kmeans = KMeans(n_clusters=5)  # Замените на желаемое количество кластеров
kmeans.fit(principal_components)

# Количество кластеров
num_clusters = kmeans.n_clusters

# Определение минимального количества главных компонент для достижения доли объясненной дисперсии более 0.85
pca = PCA()
principal_components = pca.fit_transform(data)
min_components_for_variance = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.85) + 1

# Вывод результатов
print("Координата первого объекта относительно первой главной компоненты:", round(first_object_coordinates[0], 3))
print("Координата первого объекта относительно второй главной компоненты:", round(first_object_coordinates[1], 3))
print("Доля объясненной дисперсии при использовании первых двух главных компонент:", round(explained_variance_ratio, 3))
print("Минимальное количество главных компонент для доли объясненной дисперсии > 0.85:", min_components_for_variance)
print("Количество групп объектов при использовании первых двух главных компонент:", num_clusters)
