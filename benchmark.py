import random
import time

import sequential as seq

def generate_matrix(n: int, seed: int = 42) -> list[list[float]]:
    random.seed(seed)
    return seq.generate_invertible_matrix(n)


def test_sequential(A: list[list[float]]):
    n = len(A)
    print(f"\n--- SEQUENTIAL (N={n}) ---")

    t_start = time.perf_counter()
    b_inv = seq.calculate_reverse_matrix(A)
    elapsed = time.perf_counter() - t_start

    err = seq.get_frobenius_norm_diff(A, b_inv)
    print(f"Time: {elapsed:.4f} s | Error: {err:.2e}")
    return elapsed

if __name__ == "__main__":
    N = 100
    SEED = 42

    print(f"Generating matrix {N}x{N}...")
    A = generate_matrix(N, seed=SEED)

    test_sequential(A)
