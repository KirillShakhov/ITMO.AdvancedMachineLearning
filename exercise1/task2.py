import numpy as np
import matplotlib.pyplot as plt

# Загрузите данные из файлов
loadings = np.genfromtxt('data/task2_loadings_441.csv', delimiter=';')
reduced_data = np.genfromtxt('data/task2_X_reduced_441.csv', delimiter=';')

# Выберите первые десять загрузок
first_ten_loadings = loadings[:, :10]

# Восстановите исходное изображение
restored_image = np.dot(reduced_data, first_ten_loadings.T)

# Отобразите изображение
plt.imshow(restored_image, cmap='gray')
plt.title('Восстановленное изображение')
plt.show()
