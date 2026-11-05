import random

from common import EPSILON, MAX_ITER, A, matrix_type


def multiply_matrices(a: matrix_type,
                      b: matrix_type) -> matrix_type:
    output = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for k in range(len(a[0])):
            r = a[i][k]
            for j in range(len(b[0])):
                output[i][j] += r * b[k][j]

    return output


def add_matrices(a: matrix_type,
                 b: matrix_type) -> matrix_type:
    output = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for j in range(len(a[0])):
            output[i][j] = a[i][j] + b[i][j]

    return output


def subtract_matrices(a: matrix_type,
                      b: matrix_type) -> matrix_type:
    output = [[0 for _ in range(len(a[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for j in range(len(a[0])):
            output[i][j] = a[i][j] - b[i][j]

    return output


def get_identity_matrix(n: int) -> list[list[int]]:
    output = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
    return output


def get_matrix_trace(matrix: matrix_type) -> float:
    total = 0
    for i in range(len(matrix)):
        total += matrix[i][i]

    return total


def get_transposition(matrix: matrix_type) -> matrix_type:
    new_matrix = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new_matrix[j][i] = matrix[i][j]

    return new_matrix


def divide_matrix_by_scalar(matrix: matrix_type,
                            num: int) -> matrix_type:
    new_matrix = [[0 for j in range(len(matrix))] for i in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new_matrix[i][j] = matrix[i][j] / num

    return new_matrix


def multiply_matrix_by_scalar(matrix: matrix_type,
                              num: int) -> matrix_type:
    new_matrix = [[0 for j in range(len(matrix))] for i in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new_matrix[i][j] = matrix[i][j] * num

    return new_matrix


def get_sum_of_quadrative_elements(matrix: matrix_type) -> float:
    total = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            total += matrix[i][j]**2

    return total


# ----------------------------------------------------------------


def get_frobenius_norm_diff(original: matrix_type,
                            solution: matrix_type) -> float:
    diff_matrix = multiply_matrices(a=solution, b=original)
    diff_matrix = subtract_matrices(a=diff_matrix,
                                    b=get_identity_matrix(len(diff_matrix)))

    frobenius_norm = get_sum_of_quadrative_elements(matrix=diff_matrix) ** 0.5

    return frobenius_norm


def get_cauchy_diff(previous_matrix: matrix_type,
                    new_matrix: matrix_type) -> float:
    diff_matrix = subtract_matrices(a=new_matrix, b=previous_matrix)
    frobenius_norm = get_sum_of_quadrative_elements(matrix=diff_matrix) ** 0.5

    return frobenius_norm


def get_first_element(matrix: matrix_type) -> matrix_type:
    trace = get_sum_of_quadrative_elements(matrix=matrix)

    return divide_matrix_by_scalar(matrix=get_transposition(matrix=matrix),
                                   num=trace)


def calculate_reverse_matrix(matrix: matrix_type) -> matrix_type:
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

        diff = get_cauchy_diff(previous_matrix=b_i, new_matrix=b_next)
        b_i = b_next
        cur_iter += 1

        if diff < EPSILON:
            break

    return b_i


def generate_invertible_matrix(n: int) -> matrix_type:
    matrix = [[random.uniform(-10.0, 10.0) for _ in range(n)] for _ in range(n)]

    for i in range(n):
        row_sum = 0
        for j in range(n):
            if i != j:
                row_sum += abs(matrix[i][j])

        sign = 1 if matrix[i][i] >= 0 else -1

        matrix[i][i] = sign * (row_sum + random.uniform(1.0, 5.0))

    return matrix


def print_matrix(matrix: matrix_type) -> None:
    for row in matrix:
        print(row)


if __name__ == "__main__":
    print_matrix(calculate_reverse_matrix(A))
