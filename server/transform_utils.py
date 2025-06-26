import numpy as np

# Gera uma matriz de pontos aleatórios 3D (x, y, z) com valores entre 0 e 1
def generate_random_points(n):
    return np.random.rand(n, 3)

# Cria uma matriz de transformação 4x4 com escala, rotação e translação
def create_transformation_matrix(scale, rotation, translation):
    from math import radians, cos, sin  # Importa funções matemáticas para conversão e trigonometria

    sx, sy, sz = scale  # Desempacota os valores de escala
    rx, ry, rz = map(radians, rotation)  # Converte os ângulos de rotação de graus para radianos
    tx, ty, tz = translation  # Desempacota os valores de translação

    # Matriz de rotação no eixo X
    Rx = np.array([[1, 0, 0, 0],
                   [0, cos(rx), -sin(rx), 0],
                   [0, sin(rx), cos(rx), 0],
                   [0, 0, 0, 1]])

    # Matriz de rotação no eixo Y
    Ry = np.array([[cos(ry), 0, sin(ry), 0],
                   [0, 1, 0, 0],
                   [-sin(ry), 0, cos(ry), 0],
                   [0, 0, 0, 1]])

    # Matriz de rotação no eixo Z
    Rz = np.array([[cos(rz), -sin(rz), 0, 0],
                   [sin(rz), cos(rz), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Matriz de escala (valores na diagonal)
    S = np.diag([sx, sy, sz, 1])

    # Matriz de translação (identidade com última coluna modificada)
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]

    # Multiplica as matrizes na ordem: T (translação) * Rz * Ry * Rx * S (escala)
    return T @ Rz @ Ry @ Rx @ S
