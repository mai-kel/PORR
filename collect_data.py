import gc
import os
import random
import time
from collections.abc import Callable

import matplotlib.pyplot as plt

import sequential as seq
import threads as thr
from common import matrix_type


class ExecutorStrategy:
    """Adapter pattern for switching execution backends (Threads/Processes)"""
    def __init__(self,
                 name: str,
                 func: Callable,
                 err_func: Callable) -> None:
        self.name = name
        self.func = func
        self.err_func = err_func


def wrap_seq(matrix: matrix_type, workers: int) -> matrix_type:
    return seq.calculate_reverse_matrix(matrix=matrix)


def wrap_seq_err(original: matrix_type,
                 result: matrix_type,
                 workers: int) -> float:
    return seq.get_frobenius_norm_diff(original=original,
                                       solution=result)


def wrap_thr(matrix: matrix_type, workers: int) -> matrix_type:
    return thr.calculate_reverse_matrix(matrix=matrix, workers=workers)


def wrap_thr_err(original: matrix_type,
                 result: matrix_type,
                 workers: int) -> float:
    return thr.get_frobenius_norm_diff(original=original,
                                       solution=result,
                                       workers=workers)


def generate_matrix(n: int, seed: int = 42) -> matrix_type:
    random.seed(a=seed)
    return seq.generate_invertible_matrix(n=n)


def benchmark_func(func: Callable,
                   matrix: matrix_type,
                   workers: int,
                   runs: int = 3) -> tuple[float, matrix_type]:
    times: list[float] = []
    result: matrix_type | None = None

    for i in range(runs):
        gc.collect()
        gc.disable()

        t_start: float = time.perf_counter()
        result = func(matrix=matrix, workers=workers)
        t_end: float = time.perf_counter()

        gc.enable()

        elapsed: float = t_end - t_start
        times.append(elapsed)
        print(f"    - Run {i + 1}/{runs}: {elapsed:.4f} s")

    if result is None:
        raise RuntimeError("Benchmark failed.")

    avg_time: float = sum(times) / runs
    return avg_time, result


def create_plot(n: int,
                workers_list: list[int],
                times: list[float],
                seq_time: float,
                output_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(workers_list,
             times,
             marker='o',
             linestyle='-',
             color='b',
             label='Parallel Time')
    plt.axhline(y=seq_time,
                color='r',
                linestyle='--',
                label='Sequential (Baseline)')

    plt.title(label=f"Execution time vs workers (N={n})")
    plt.xlabel(xlabel="Workers (p)")
    plt.ylabel(ylabel="Average Time [s]")
    plt.xticks(ticks=workers_list)
    plt.grid(visible=True, linestyle=':', alpha=0.7)
    plt.legend()

    filename: str = f"benchmark_N{n}.png"
    filepath: str = os.path.join(output_dir, filename)

    plt.savefig(fname=filepath)
    print(f"Plot saved as '{filepath}'.")

    # Non-blocking show, allows next iterations to proceed
    plt.show(block=False)


def run_suite(matrix_sizes: list[int],
              workers_list: list[int],
              runs: int = 3,
              output_dir: str = ".") -> None:
    seq_strat = ExecutorStrategy(name="Sequential",
                                 func=wrap_seq,
                                 err_func=wrap_seq_err)
    thr_strat = ExecutorStrategy(name="Threads",
                                 func=wrap_thr,
                                 err_func=wrap_thr_err)

    for n in matrix_sizes:
        print(f"\n=== BENCHMARK N={n} ===")
        matrix: matrix_type = generate_matrix(n=n)

        print(f"\nTesting {seq_strat.name}...")
        seq_t, seq_res = benchmark_func(func=seq_strat.func,
                                        matrix=matrix,
                                        workers=1,
                                        runs=runs)
        seq_err: float = seq_strat.err_func(original=matrix,
                                            result=seq_res,
                                            workers=1)
        print(f"  Avg Time: {seq_t:.4f} s | Err: {seq_err:.2e}")

        par_times: list[float] = []

        print(f"\nTesting {thr_strat.name}...")
        for w in workers_list:
            print(f"  Workers: {w}")
            thr_t, thr_res = benchmark_func(func=thr_strat.func,
                                            matrix=matrix,
                                            workers=w,
                                            runs=runs)
            speedup: float = seq_t / thr_t
            thr_err: float = thr_strat.err_func(original=matrix,
                                                result=thr_res,
                                                workers=w)
            print(f"  Time: {thr_t:.4f} s | "
                  f"Speedup: {speedup:.2f}x | Err: {thr_err:.2e}")

            par_times.append(thr_t)

        print("\n=== SUMMARY ===")
        print(f"{'Workers (p)':<12} | "
              f"{'Time [s]':<15} | "
              f"{'Speedup S(n,p)':<25}")
        print("-" * 55)
        print(f"{'Sequential':<12} | "
              f"{seq_t:<15.4f} | "
              f"{'1.000':<25}")

        for w, t in zip(workers_list, par_times):
            speedup_val: float = seq_t / t
            print(f"{w:<12} | "
                  f"{t:<15.4f} | "
                  f"{speedup_val:<25.4f}")

        create_plot(n=n,
                    workers_list=workers_list,
                    times=par_times,
                    seq_time=seq_t,
                    output_dir=output_dir)

    # Blocking show at the very end to keep all windows alive
    plt.show()


if __name__ == "__main__":
    MATRICES_TO_TEST: list[int] = [50, 100, 200, 300, 400]
    WORKERS_TO_TEST: list[int] = [1, 2, 4, 8, 12, 16, 20]
    MEASUREMENT_RUNS: int = 3
    PLOTS_DIR: str = "./plots/threads"

    os.makedirs(PLOTS_DIR, exist_ok=True)

    run_suite(matrix_sizes=MATRICES_TO_TEST,
              workers_list=WORKERS_TO_TEST,
              runs=MEASUREMENT_RUNS,
              output_dir=PLOTS_DIR)
