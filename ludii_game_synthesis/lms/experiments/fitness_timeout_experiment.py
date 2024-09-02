import glob
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import VALIDATION_GAMES
from java_api import StandardEvaluation
from fitness_helpers import _get_fast_evaluation, _evaluate_fitness, FITNESS_METRIC_KEYS

NUM_TRIALS = 5

# Load the .lud files for each game
game_strs = []
for name in VALIDATION_GAMES:
    filename = glob.glob(os.path.join("ludii_data/games/expanded", "**", f"{name}.lud"), recursive=True)[0]
    game_strs.append(open(filename, "r").read())

evaluator = StandardEvaluation()

for idx, game_str in tqdm(enumerate(game_strs), desc="Evaluating", total=len(game_strs)):
    evaluations, times = [], []

    # for _ in tqdm(range(NUM_TRIALS), desc=f"Evaluating: {VALIDATION_GAMES[idx]}", leave=False):
    start = time.perf_counter()
    evaluation = evaluator.evaluate(game_str, ai_name="UCT", num_games=NUM_TRIALS, thinking_time=0.5, max_turns=100)
    evaluations.append(evaluation)
    duration = time.perf_counter() - start
    times.append(duration)

    print(f"{VALIDATION_GAMES[idx]} ({(duration / NUM_TRIALS):.2f}s avg): {evaluation}")


    # average_evaluation = {
    #     key: sum(float(val[key]) for val in evaluations) / len(evaluations)
    #     for key in FITNESS_METRIC_KEYS if key in evaluations[0]
    # }
    # print(f"{VALIDATION_GAMES[idx]} ({np.mean(times):.2f}s avg): {average_evaluation}")