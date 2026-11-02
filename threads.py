import random
import concurrent.futures

# i - wiersz
# j - kolumna

matrix_type = list[list[float]]

EPSILON = 0.00001
MAX_ITER = 100
THREADS = 4

A = [
    [-1,  -2,   3],
    [ 0,   2,  -1],
    [-1,   3,   0]
]


def split_range_for_workers(rows: int, workers: int) -> list[tuple[int, int]]:
    ranges = []
    full_range_len = rows // workers
    over_len_ranges = rows % workers

    left_interval = 0
    for i in range(min(workers, rows)):
        right_interval = left_interval + full_range_len
        if i < over_len_ranges:
            right_interval += 1
        ranges.append((left_interval, right_interval))
        left_interval = right_interval

    return ranges


def multiply_matrices(a: matrix_type,
                      b: matrix_type,
                      workers: int | None = THREADS) -> matrix_type:
    output = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        row_ranges = split_range_for_workers(rows=len(output), workers=workers)
        for row_range in row_ranges:
            executor.submit(multiply_matrices_rows_range, a=a, b=b, output=output,
                            start_row=row_range[0], end_row=row_range[1])

    return output


def multiply_matrices_rows_range(a: matrix_type,
                                 b: matrix_type,
                                 output: matrix_type,
                                 start_row: int,
                                 end_row: int) -> None:
    for i in range(start_row, end_row):
        for k in range(len(a[0])):
            r = a[i][k]
            for j in range(len(b[0])):
                output[i][j] += r * b[k][j]


def add_matrices(a: matrix_type,
                 b: matrix_type,
                 workers: int | None = THREADS) -> matrix_type:
    output = [[0.0 for _ in range(len(a[0]))] for _ in range(len(a))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        row_ranges = split_range_for_workers(rows=len(a), workers=workers)
        for row_range in row_ranges:
            executor.submit(add_matrices_rows_range, a=a, b=b, output=output,
                            start_row=row_range[0], end_row=row_range[1])

    return output


def add_matrices_rows_range(a: matrix_type,
                            b: matrix_type,
                            output: matrix_type,
                            start_row: int,
                            end_row: int) -> None:
    for i in range(start_row, end_row):
        for j in range(len(a[0])):
            output[i][j] = a[i][j] + b[i][j]


def subtract_matrices(a: matrix_type,
                      b: matrix_type,
                      workers: int | None = THREADS) -> matrix_type:
    output = [[0.0 for _ in range(len(a[0]))] for _ in range(len(a))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        row_ranges = split_range_for_workers(rows=len(a), workers=workers)
        for row_range in row_ranges:
            executor.submit(subtract_matrices_rows_range, a=a, b=b, output=output,
                            start_row=row_range[0], end_row=row_range[1])

    return output


def subtract_matrices_rows_range(a: matrix_type,
                                 b: matrix_type,
                                 output: matrix_type,
                                 start_row: int,
                                 end_row: int) -> None:
    for i in range(start_row, end_row):
        for j in range(len(a[0])):
            output[i][j] = a[i][j] - b[i][j]


def get_identity_matrix(n: int) -> list[list[int]]:
    output = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
    return output


def get_matrix_trace(matrix: matrix_type) -> float:
    total = 0
    for i in range(len(matrix)):
        total += matrix[i][i]

    return total


def multiply_matrix_by_scalar(matrix: matrix_type,
                              num: float,
                              workers: int | None = THREADS) -> matrix_type:
    output = [[0.0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        row_ranges = split_range_for_workers(rows=len(matrix), workers=workers)
        for row_range in row_ranges:
            executor.submit(multiply_matrix_by_scalar_rows_range, matrix=matrix, num=num, output=output,
                            start_row=row_range[0], end_row=row_range[1])

    return output


def multiply_matrix_by_scalar_rows_range(matrix: matrix_type,
                                         num: float,
                                         output: matrix_type,
                                         start_row: int,
                                         end_row: int) -> None:
    for i in range(start_row, end_row):
        for j in range(len(matrix[0])):
            output[i][j] = matrix[i][j] * num


def divide_matrix_by_scalar(matrix: matrix_type,
                            num: float,
                            workers: int | None = THREADS) -> matrix_type:
    output = [[0.0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        row_ranges = split_range_for_workers(rows=len(matrix), workers=workers)
        for row_range in row_ranges:
            executor.submit(divide_matrix_by_scalar_rows_range, matrix=matrix, num=num, output=output,
                            start_row=row_range[0], end_row=row_range[1])

    return output


def divide_matrix_by_scalar_rows_range(matrix: matrix_type,
                                       num: float,
                                       output: matrix_type,
                                       start_row: int,
                                       end_row: int) -> None:
    for i in range(start_row, end_row):
        for j in range(len(matrix[0])):
            output[i][j] = matrix[i][j] / num


def get_transposition(matrix: matrix_type,
                      workers: int | None = THREADS) -> matrix_type:
    new_matrix = [[0.0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        row_ranges = split_range_for_workers(rows=len(matrix), workers=workers)
        for row_range in row_ranges:
            executor.submit(get_transposition_rows_range, matrix=matrix, output=new_matrix,
                            start_row=row_range[0], end_row=row_range[1])

    return new_matrix


def get_transposition_rows_range(matrix: matrix_type,
                                 output: matrix_type,
                                 start_row: int,
                                 end_row: int) -> None:
    for i in range(start_row, end_row):
        for j in range(len(matrix[0])):
            output[j][i] = matrix[i][j]


def get_sum_of_quadrative_elements(matrix: matrix_type,
                                   workers: int | None = THREADS) -> float:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        row_ranges = split_range_for_workers(rows=len(matrix), workers=workers)
        futures = [
            executor.submit(get_sum_of_quadrative_elements_rows_range, matrix=matrix,
                            start_row=row_range[0], end_row=row_range[1])
            for row_range in row_ranges
        ]
        total = sum(f.result() for f in futures)

    return total


def get_sum_of_quadrative_elements_rows_range(matrix: matrix_type,
                                              start_row: int,
                                              end_row: int) -> float:
    local_sum = 0.0
    for i in range(start_row, end_row):
        for j in range(len(matrix[0])):
            local_sum += matrix[i][j] ** 2
    return local_sum


# ----------------------------------------------------------------


def get_frobenius_norm_diff(original: matrix_type,
                            solution: matrix_type) -> float:
    diff_matrix = multiply_matrices(a=solution, b=original)
    diff_matrix = subtract_matrices(diff_matrix,
                                    get_identity_matrix(len(diff_matrix)))

    frobenius_norm = get_sum_of_quadrative_elements(diff_matrix)**0.5

    return frobenius_norm


def get_cauchy_diff(previous_matrix: matrix_type,
                    new_matrix: matrix_type) -> float:
    diff_matrix = subtract_matrices(new_matrix, previous_matrix)
    frobenius_norm = get_sum_of_quadrative_elements(diff_matrix)**0.5

    return frobenius_norm


def get_first_element(matrix: matrix_type) -> matrix_type:
    trace = get_sum_of_quadrative_elements(matrix)

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
