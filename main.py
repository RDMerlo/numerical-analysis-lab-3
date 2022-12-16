import numpy as np
import math

# Функция печати в консоль матрицы
def print_matrix(matrix, str='', before=8, after=4):
    # Печать числа с настройкой чисел до и после точки
    f = f'{{: {before}.{after}f}}'
    print(str)
    print('\n'.join([f''.join(f.format(el)
                              for el in row)
                     for row in matrix]) + '\n')


# Система нелинейных уравнений
def func(params):
    x, y, z = params
    return np.array([
        x + x ** 2 - 2 * y * z - 0.1,
        y - y ** 2 + 3 * x * z + 0.2,
        z + z ** 2 + 2 * x * y - 0.3
    ])

# Матрица Якоби
def matrix_jacobian(params):
    x, y, z = params
    return np.array([
        [1 + 2*x, -2*z, -2*y],
        [3*z, 1-2*y, 3*x],
        [2*y, 2*x, 1+2*z]
    ], float)

# Метод Ньютона с точностью 1.0e-6
def method_Newton(dot):
    eps = 1.0e-6

    f = func(dot)

    steps = 0
    while True:
      Jf = matrix_jacobian(dot).reshape(3,3)
      # print("jf=",Jf)
      dot_new = dot - np.linalg.inv(Jf) @ f
      
      f = func(dot_new)

      dot_list = dot.tolist()
      dot1 = np.zeros(3, dtype=float)
      k = 0
      for row in dot_list:
        dot1[k] = float(row[0])
        k+=1

      dot_list = dot_new.tolist()
      dot0 = np.zeros(3, dtype=float)
      k = 0
      for row in dot_list:
        dot0[k] = float(row[0])
        k+=1

      if get_norm_vector(dot0, dot1) < eps:
        break

      dot = dot_new
      steps += 1
      
    return dot, steps
  
# Метод Зейделя
def method_seidel(x, y, z):
    # Точки Xo,Уо,Zo для начального приближения
    eps = 0.0001
    n = 10000
    for k in range(n):
     xk_1 = (x ** 2 - 2 * y * z - 0.1)
     yk_1 = (-y ** 2 + 3 * xk_1 * z + 0.2)
     zk_1 = (z ** 2 + 2 * xk_1 * yk_1 - 0.3)
     if (abs(xk_1-x)<eps and abs(yk_1-y)< eps and abs(zk_1-z)<eps):
        zz1 = xk_1
        zz2 = yk_1
        zz3 = zk_1
        break
     x = xk_1
     y = yk_1
     z = zk_1
    arr = np.array([[-zz1],[-zz2],[-zz3]])
    aa= np.round(arr,decimals=16)
    return aa, k


# Достаточное условие сходимости
def dost_usl(dot):
  
    dot1 = np.zeros(3)

    norm_X = np.zeros((3, 1))
    for i in range(3):
      norm_X[i] = (abs(dot1[i] - dot[i]))**2
    max_sum = math.sqrt(sum(norm_X))

    if (max_sum < 1):
        print("Достаточное условие сходимости выполняется\n")
    else:
        print("\nДостаточное условие сходимости не выполняется\n")


# Вектор Невязки
def vector_nevyazki(f, dot):
    return np.array(f) - func(np.array(dot))

def get_norm_vector(x_new, X): 
  norm_X = np.zeros((3, 1))
  for i in range(3):
    norm_X[i] = (abs(x_new[i] - X[i]))**2
  return(math.sqrt(sum(norm_X)))

matrix_X0_dot_arr = [[0.1],[-0.2],[0.3]]

# Точка Xo для начального приближения
matrix_X0_dot = np.array(matrix_X0_dot_arr, float)
# 1) Решение системы нелинейных уравнений f(x) = 0 методом простых итераций с точностью 1e-3
arr, steps = method_seidel(matrix_X0_dot_arr[0][0], matrix_X0_dot_arr[1][0], matrix_X0_dot_arr[2][0])
print("Метод Зейделя")
print_matrix(arr)
print("Количество шагов\n",steps)
# Вектор невязки
print("\nВектор невязки")
print_matrix(vector_nevyazki([[0], [0], [0]], arr.copy()), "", 4, 16)
# 2) Проверка выполнения достаточного условия сходимости
dost_usl([0.1, -0.2, 0.3])

matrix_dot_solved, steps = method_Newton(matrix_X0_dot.copy())

print('Метод Ньютона')
print_matrix(matrix_dot_solved)
print('Шагов метода:', steps)

# Вектор невязки
print("\nВектор невязки")
print_matrix(vector_nevyazki([[0], [0], [0]], matrix_dot_solved.copy()), "", 4,16)
