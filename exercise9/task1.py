# Определим все необходимые параметры
gamma = 0.8
max_iterations = 1000

# Вероятности выбора действий
pi_a1_s1 = 0.4
pi_a2_s1 = 0.6
pi_a2_s2 = 1.0
pi_a1_s3 = 0.5
pi_a3_s3 = 0.5
pi_a1_s4 = 0.5
pi_a3_s4 = 0.5

# Начальные значения ценности состояний
V = {
    's1': 0,
    's2': 0,
    's3': 0,
    's4': 0
}

# Основной итерационный цикл
threshold = 0.001
delta = float('inf')
iteration = 0
while delta > threshold and iteration < max_iterations:
    V_prev = V.copy()

    V['s1'] = gamma * (pi_a1_s1 * (2.0 * V_prev['s2'] + 3.0 * V_prev['s3'] + 1.0) + pi_a2_s1 * (2.0 * V_prev['s2'] + 1.0))
    V['s2'] = gamma * V_prev['s1']
    V['s3'] = gamma * (pi_a1_s3 * (1.0 * V_prev['s2'] - 3.0 * V_prev['s4'] + 1.0) + pi_a3_s3 * (0.2 * V_prev['s2'] + 0.8 * V_prev['s3'] + 1.0 * V_prev['s4'] + 6.0))
    V['s4'] = gamma * (pi_a1_s4 * (0.6 * V_prev['s1'] + 0.4 * V_prev['s3'] + 2.0) + pi_a3_s4 * (1.0 * V_prev['s3'] - 3.0))

    # Вычисляем изменение для проверки сходимости
    deltas = [abs(V[s] - V_prev[s]) for s in V]
    delta = max(deltas)
    iteration += 1

# Выводим полученные значения
print(f"V(s1) = {V['s1']:.3f}")
print(f"V(s2) = {V['s2']:.3f}")
print(f"V(s3) = {V['s3']:.3f}")
print(f"V(s4) = {V['s4']:.3f}")
