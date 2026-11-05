import concurrent.futures
import random

from common import EPSILON, MAX_ITER, A, matrix_type


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


def multiply_matrices_worker(a: matrix_type,
                            b: matrix_type,
                            start_row: int,
                            end_row: int) -> matrix_type:
    chunk: matrix_type = []
    cols_b = len(b[0])
    cols_a = len(a[0])
    for i in range(start_row, end_row):
        row = [0.0] * cols_b
        for k in range(cols_a):
            r = a[i][k]
            for j in range(cols_b):
                row[j] += r * b[k][j]
        chunk.append(row)
    return chunk


def multiply_matrices(a: matrix_type,
                      b: matrix_type,
                      workers: int,
                      executor: concurrent.futures.ProcessPoolExecutor
                      ) -> matrix_type:
    row_ranges = split_range_for_workers(rows=len(a), workers=workers)
    futures = [
        executor.submit(multiply_matrices_worker, a, b, row_range[0], row_range[1])
        for row_range in row_ranges
    ]
    output: matrix_type = []
    for f in futures:
        output.extend(f.result())

    return output


def add_matrices_worker(a: matrix_type,
                       b: matrix_type,
                       start_row: int,
                       end_row: int) -> matrix_type:
    chunk: matrix_type = []
    cols = len(a[0])
    for i in range(start_row, end_row):
        row = [a[i][j] + b[i][j] for j in range(cols)]
        chunk.append(row)
    return chunk


def add_matrices(a: matrix_type,
                 b: matrix_type,
                 workers: int,
                 executor: concurrent.futures.ProcessPoolExecutor) -> matrix_type:
    row_ranges = split_range_for_workers(rows=len(a), workers=workers)
    futures = [
        executor.submit(add_matrices_worker, a, b, r[0], r[1])
        for r in row_ranges
    ]
    output: matrix_type = []
    for f in futures:
        output.extend(f.result())
    return output


def subtract_matrices_worker(a: matrix_type,
                            b: matrix_type,
                            start_row: int,
                            end_row: int) -> matrix_type:
    chunk: matrix_type = []
    cols = len(a[0])
    for i in range(start_row, end_row):
        row = [a[i][j] - b[i][j] for j in range(cols)]
        chunk.append(row)
    return chunk


def subtract_matrices(a: matrix_type,
                      b: matrix_type,
                      workers: int,
                      executor: concurrent.futures.ProcessPoolExecutor) -> matrix_type:
    row_ranges = split_range_for_workers(rows=len(a), workers=workers)
    futures = [
        executor.submit(subtract_matrices_worker, a, b, r[0], r[1])
        for r in row_ranges
    ]
    output: matrix_type = []
    for f in futures:
        output.extend(f.result())
    return output


def get_identity_matrix(n: int) -> list[list[int]]:
    output = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
    return output


def get_matrix_trace(matrix: matrix_type) -> float:
    total = 0.0
    for i in range(len(matrix)):
        total += matrix[i][i]
    return total


def multiply_matrix_by_scalar_worker(matrix: matrix_type,
                                    num: float,
                                    start_row: int,
                                    end_row: int) -> matrix_type:
    chunk: matrix_type = []
    cols = len(matrix[0])
    for i in range(start_row, end_row):
        row = [matrix[i][j] * num for j in range(cols)]
        chunk.append(row)
    return chunk


def multiply_matrix_by_scalar(matrix: matrix_type,
                              num: float,
                              workers: int,
                              executor: concurrent.futures.ProcessPoolExecutor) -> matrix_type:
    row_ranges = split_range_for_workers(rows=len(matrix), workers=workers)
    futures = [
        executor.submit(multiply_matrix_by_scalar_worker, matrix, num, r[0], r[1])
        for r in row_ranges
    ]
    output: matrix_type = []
    for f in futures:
        output.extend(f.result())
    return output


def divide_matrix_by_scalar_worker(matrix: matrix_type,
                                  num: float,
                                  start_row: int,
                                  end_row: int) -> matrix_type:
    chunk: matrix_type = []
    cols = len(matrix[0])
    for i in range(start_row, end_row):
        row = [matrix[i][j] / num for j in range(cols)]
        chunk.append(row)
    return chunk


def divide_matrix_by_scalar(matrix: matrix_type,
                            num: float,
                            workers: int,
                            executor: concurrent.futures.ProcessPoolExecutor) -> matrix_type:
    row_ranges = split_range_for_workers(rows=len(matrix), workers=workers)
    futures = [
        executor.submit(divide_matrix_by_scalar_worker, matrix, num, r[0], r[1])
        for r in row_ranges
    ]
    output: matrix_type = []
    for f in futures:
        output.extend(f.result())
    return output


def get_transposition_worker(matrix: matrix_type,
                             start_row: int,
                             end_row: int) -> matrix_type:
    chunk: matrix_type = []
    rows_orig = len(matrix)
    for i in range(start_row, end_row):
        row = [matrix[j][i] for j in range(rows_orig)]
        chunk.append(row)
    return chunk


def get_transposition(matrix: matrix_type,
                      workers: int,
                      executor: concurrent.futures.ProcessPoolExecutor) -> matrix_type:
    cols = len(matrix[0])
    row_ranges = split_range_for_workers(rows=cols, workers=workers)
    futures = [
        executor.submit(get_transposition_worker, matrix, r[0], r[1])
        for r in row_ranges
    ]
    output: matrix_type = []
    for f in futures:
        output.extend(f.result())
    return output


def get_sum_of_quadrative_elements_worker(matrix: matrix_type,
                                          start_row: int,
                                          end_row: int) -> float:
    local_sum = 0.0
    cols = len(matrix[0])
    for i in range(start_row, end_row):
        for j in range(cols):
            local_sum += matrix[i][j] ** 2
    return local_sum


def get_sum_of_quadrative_elements(matrix: matrix_type,
                                   workers: int,
                                   executor: concurrent.futures.ProcessPoolExecutor) -> float:
    row_ranges = split_range_for_workers(rows=len(matrix), workers=workers)
    futures = [
        executor.submit(get_sum_of_quadrative_elements_worker, matrix, r[0], r[1])
        for r in row_ranges
    ]
    return sum(f.result() for f in futures)





# ----------------------------------------------------------------


def get_frobenius_norm_diff(original: matrix_type,
                            solution: matrix_type,
                            workers: int) -> float:

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        diff_matrix = multiply_matrices(a=solution, b=original,
                                        workers=workers, executor=executor)
        diff_matrix = subtract_matrices(a=diff_matrix,
                                        b=get_identity_matrix(len(diff_matrix)),
                                        workers=workers,
                                        executor=executor)
        frobenius_norm = get_sum_of_quadrative_elements(matrix=diff_matrix,
                                                        workers=workers,
                                                        executor=executor) ** 0.5

    return frobenius_norm


def get_cauchy_diff(previous_matrix: matrix_type,
                    new_matrix: matrix_type,
                    workers: int,
                    executor: concurrent.futures.ProcessPoolExecutor) -> float:
    diff_matrix = subtract_matrices(a=new_matrix,
                                    b=previous_matrix,
                                    workers=workers,
                                    executor=executor)
    frobenius_norm = get_sum_of_quadrative_elements(matrix=diff_matrix,
                                                    workers=workers,
                                                    executor=executor) ** 0.5
    return frobenius_norm


def get_first_element(matrix: matrix_type,
                      workers: int,
                      executor: concurrent.futures.ProcessPoolExecutor
                      ) -> matrix_type:
    trace = get_sum_of_quadrative_elements(matrix=matrix,
                                           workers=workers,
                                           executor=executor)
    transposition = get_transposition(matrix=matrix,
                                      workers=workers,
                                      executor=executor)
    return divide_matrix_by_scalar(matrix=transposition,
                                   num=trace,
                                   workers=workers,
                                   executor=executor)


def calculate_reverse_matrix(matrix: matrix_type, workers: int) -> matrix_type:
    cur_iter = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        b_i = get_first_element(matrix, workers, executor)

        while cur_iter < MAX_ITER:
            temp_m = multiply_matrices(a=b_i,
                                       b=matrix,
                                       workers=workers,
                                       executor=executor)
            b_k_a_b_k = multiply_matrices(a=temp_m,
                                          b=b_i,
                                          workers=workers,
                                          executor=executor)
            double_bk = multiply_matrix_by_scalar(matrix=b_i,
                                                  num=2.0,
                                                  workers=workers,
                                                  executor=executor)
            b_next = subtract_matrices(a=double_bk,
                                       b=b_k_a_b_k,
                                       workers=workers,
                                       executor=executor)

            diff = get_cauchy_diff(previous_matrix=b_i,
                                   new_matrix=b_next,
                                   workers=workers,
                                   executor=executor)
            b_i = b_next
            cur_iter += 1

            if diff < EPSILON:
                break

        return b_i


def generate_invertible_matrix(n: int) -> matrix_type:
    matrix = [[random.uniform(-10.0, 10.0) for _ in range(n)] for _ in range(n)]

    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i != j:
                row_sum += abs(matrix[i][j])

        sign = 1.0 if matrix[i][i] >= 0 else -1.0
        matrix[i][i] = sign * (row_sum + random.uniform(1.0, 5.0))

    return matrix


def print_matrix(matrix: matrix_type) -> None:
    for row in matrix:
        print(row)


if __name__ == "__main__":
    print_matrix(calculate_reverse_matrix(A, workers=4))