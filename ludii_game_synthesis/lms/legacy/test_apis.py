import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import os
import psutil
import signal
import time

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from fitness_helpers import _get_fast_evaluation, _close_fast_evaluation
from java_api import Autocomplete, FastTrace, StandardEvaluation

game = open("./ludii_data/games/expanded/board/space/line/Fettas.lud", "r").read()

autocomplete = Autocomplete()

random_evaluator = None
fast_trace_evaluator = None

def evaluate(idx_game_str):
    global random_evaluator, fast_trace_evaluator

    start = time.perf_counter()

    idx, game_str = idx_game_str

    if random_evaluator is None:
        random_evaluator = StandardEvaluation()
    random_evaluator.evaluate(game_str)

    if fast_trace_evaluator is None:    
        fast_trace_evaluator = FastTrace()
    fast_trace_evaluator.evaluate(game_str)
    
    time_taken = f"{(time.perf_counter() - start):.2f}s"
    memory = psutil.virtual_memory().percent
    num_java_processes = len([p for p in psutil.process_iter() if p.name() == 'java'])
    cpu_usages = psutil.cpu_percent(percpu=True)

    data = [idx, time_taken, memory, num_java_processes] + cpu_usages

    return data


def pass_in_evaluate(eval_game_strs):
    evaluator, game_strs = eval_game_strs
    
    for game_str in game_strs:
        evaluator.evaluate(game_str)

NUM_EVALS = 72
NUM_THREADS = 16
all_games = [game] * NUM_EVALS

data = []
headers = ["Idx", "Time", "Mem", "# Java"] + [f"C{i}" for i in range(mp.cpu_count())]

start = time.perf_counter()
with mp.Pool(processes=NUM_THREADS) as pool:
    with tqdm(total=NUM_EVALS, desc="Evaluating fitnesses", leave=True) as pbar:
        for out in pool.imap(evaluate, zip(range(NUM_EVALS), all_games)):
            data.append(out)
            pbar.update(1)

        # for out in pool.imap(_get_fast_evaluation, all_games):
        #     data.append(out)
        #     pbar.update(1)

    pool.map(terminate, zip(range(NUM_EVALS), all_games))
    # pool.map(_close_fast_evaluation, all_games)

# print(tabulate(
#     data,
#     headers=headers,
#     tablefmt="fancy_grid",
# ))


print(f"\nTIME TAKEN FOR {NUM_EVALS} EVALS ON {NUM_THREADS} THREADS: {time.perf_counter() - start:.2f}s")

exit()

random_evaluators = [StandardEvaluation() for _ in range(NUM_THREADS)]
splits = np.array_split(all_games, NUM_THREADS)
start = time.perf_counter()

for evaluator, game_strs in zip(random_evaluators, splits):
    pass_in_evaluate((evaluator, game_strs))


print(f"\nTIME TAKEN FOR {NUM_EVALS} PASS-IN EVALS ON {NUM_THREADS} THREADS: {time.perf_counter() - start:.2f}s")

input("Press Enter to continue...")