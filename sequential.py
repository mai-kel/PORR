from random import random

# i - wiersz
# j - kolumna

EPSILON = 0.00001
MAX_ITER = 100

A = [
    [-1,  -2,   3],
    [ 0,   2,  -1],
    [-1,   3,   0]
]


def multiply_matrices(a: list[list[float]],
                      b: list[list[float]]) -> list[list[float]]:
    output = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(a[0])):
                output[i][j] += a[i][k] * b[k][j]

    return output


def add_matrices(a: list[list[float]],
                 b: list[list[float]]) -> list[list[float]]:
    output = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for j in range(len(a[0])):
            output[i][j] = a[i][j] + b[i][j]

    return output


def subtract_matrices(a: list[list[float]],
                      b: list[list[float]]) -> list[list[float]]:
    output = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for j in range(len(a[0])):
            output[i][j] = a[i][j] - b[i][j]

    return output


def get_identity_matrix(n: int) -> list[list[int]]:
    output = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
    return output


def get_matrix_trace(matrix: list[list[float]]) -> float:
    total = 0
    for i in range(len(matrix)):
        total += matrix[i][i]

    return total


def get_transposition(matrix: list[list[float]]) -> list[list[float]]:
    new_matrix = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new_matrix[j][i] = matrix[i][j]

    return new_matrix


def divide_matrix_by_scalar(matrix: list[list[float]],
                            num: int) -> list[list[float]]:
    new_matrix = [[0 for j in range(len(matrix))] for i in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new_matrix[i][j] = matrix[i][j] / num

    return new_matrix


def multiply_matrix_by_scalar(matrix: list[list[float]],
                              num: int) -> list[list[float]]:
    new_matrix = [[0 for j in range(len(matrix))] for i in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new_matrix[i][j] = matrix[i][j] * num

    return new_matrix


def get_sum_of_quadrative_elements(matrix: list[list[float]]) -> float:
    total = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            total += matrix[i][j]**2

    return total


# ----------------------------------------------------------------


def get_frobenius_norm_diff(original: list[list[float]],
                            solution: list[list[float]]) -> float:
    diff_matrix = multiply_matrices(a=solution, b=original)
    diff_matrix = subtract_matrices(diff_matrix,
                                    get_identity_matrix(len(diff_matrix)))

    frobenius_norm = get_sum_of_quadrative_elements(diff_matrix)**0.5

    return frobenius_norm


def get_cauchy_diff(previous_matrix: list[list[float]],
                    new_matrix: list[list[float]]) -> float:
    diff_matrix = subtract_matrices(new_matrix, previous_matrix)
    frobenius_norm = get_sum_of_quadrative_elements(diff_matrix)**0.5

    return frobenius_norm


def get_first_element(matrix: list[list[float]]) -> list[list[float]]:
    trace = get_sum_of_quadrative_elements(matrix)

    return divide_matrix_by_scalar(matrix=get_transposition(matrix=matrix),
                                   num=trace)


def calculate_reverse_matrix(matrix: list[list[float]]) -> list[list[float]]:
    cur_iter = 0
    b_i = get_first_element(matrix=matrix)

    while cur_iter < MAX_ITER:
        temp_m = multiply_matrices(a=b_i, b=matrix)
        b_k_a_b_k = multiply_matrices(a=temp_m, b=b_i)
        double_bk = multiply_matrix_by_scalar(matrix=b_i, num=2)

        b_next = subtract_matrices(
            a=double_bk,
            b=b_k_a_b_k
        )

        if get_cauchy_diff(previous_matrix=b_i, new_matrix=b_next) < EPSILON:
            break

        b_i = b_next
        cur_iter += 1

    return b_i


def generate_invertible_matrix(n: int) -> list[list[float]]:
    matrix = [[random.uniform(-10.0, 10.0) for _ in range(n)] for _ in range(n)]

    for i in range(n):
        row_sum = 0
        for j in range(n):
            if i != j:
                row_sum += abs(matrix[i][j])

        sign = 1 if matrix[i][i] >= 0 else -1

        matrix[i][i] = sign * (row_sum + random.uniform(1.0, 5.0))

    return matrix


def print_matrix(matrix: list[list[float]]) -> None:
    for row in matrix:
        print(row)


print_matrix(calculate_reverse_matrix(A))
