import numpy as np
import math

def print_array(A):
  for i in range(0, len(A), 1):
    for j in range(0, len(A[i]), 1):
      if (A[i][j] != None):
        print("%.4f     " % A[i][j], end="")
      else:
        print("     ", "nul", end="")
    print("")

def print_vector(A):
  for j in range(0, len(A), 1):
    print("%.4f     " % A[j])
  print("")
  
def print_vector_18(A, p=18):
  for j in range(0, len(A), 1):
    print(f"%.{p}f     " % A[j])
  print("")


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

def matrix_jacobian_for_seidel(params):
    x, y, z = params
    return np.array([
        [-2*x, 2*z, 2*y],
        [-3*z, 2*y, -3*x],
        [-2*y, -2*x, -2*z]
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

      if get_norm_vector(dot, dot_new) < eps:
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
    arr = np.array([-zz1,-zz2,-zz3])
    aa= np.round(arr,decimals=16)
    return aa, k

# Достаточное условие сходимости
def dost_usl(dot):
  
    # dot1 = np.array([0, 0, 0])

    # norm_X = np.zeros((3, 1))
    # for i in range(3):
    #   norm_X[i] = (abs(dot1[i] - dot[i]))**2
    # max_sum = math.sqrt(sum(norm_X))

    max_sum = 0

    for fi in matrix_jacobian_for_seidel(dot):
        max_sum = max(max_sum, np.sum(abs(fi)))

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

# Точка Xo для начального приближения
# matrix_X0_dot_arr = [0.1, -0.2,0.3]
matrix_X0_dot_arr = [0.0, -0.2, 0.1]
matrix_X0_dot = np.array(matrix_X0_dot_arr, float)

arr, steps = method_seidel(matrix_X0_dot_arr[0], matrix_X0_dot_arr[1], matrix_X0_dot_arr[2])
print("Метод Зейделя")
print_vector_18(arr, 8)
print("Количество шагов\n",steps)
print("\nВектор невязки")
r = vector_nevyazki([0, 0, 0], arr.copy())
print_vector_18(r, 8)

#Проверка выполнения достаточного условия сходимости
dost_usl(matrix_X0_dot_arr)


print('Метод Ньютона')
matrix_dot_solved, steps = method_Newton(matrix_X0_dot.copy())
print_vector_18(matrix_dot_solved, 8)
print('Шагов метода:', steps)

# Вектор невязки
print("\nВектор невязки")
r = vector_nevyazki([0, 0, 0], matrix_dot_solved.copy())
print_vector_18(r,10)
