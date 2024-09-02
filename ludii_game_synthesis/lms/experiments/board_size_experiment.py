from itertools import product
import os
import sys

import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from fitness_helpers import _get_fast_evaluation, _evaluate_fitness

NUM_TRIALS = 50
NUM_EVALS_PER_GAME = 25

RANDOM_EVAL_ARGS = {"ai_name": "Random", "num_games": NUM_EVALS_PER_GAME, "thinking_time": 0.1, "max_turns": 250}
FAST_UCT_EVAL_ARGS = {"ai_name": "UCT", "num_games": NUM_EVALS_PER_GAME, "thinking_time": 0.1, "max_turns": 250}
MED_UCT_EVAL_ARGS = {"ai_name": "UCT", "num_games": NUM_EVALS_PER_GAME, "thinking_time": 0.5, "max_turns": 250}

REGIMES = {
    "random": RANDOM_EVAL_ARGS,
    "uct_fast": FAST_UCT_EVAL_ARGS,
    # "uct_med": MED_UCT_EVAL_ARGS
}

game_strs = [
    open("./experiments/notakto_variant.lud", "r").read(),
    open("./experiments/notakto_variant_square.lud", "r").read(),
    open("./experiments/notakto_variant_square_large.lud", "r").read()
]

results = []
total_iters = NUM_TRIALS * len(game_strs) * len(REGIMES)
for trial, game_str, regime in tqdm(product(range(NUM_TRIALS), game_strs, REGIMES.keys()),
                                    total=total_iters,
                                    desc="Comparing fitness evals"):

    eval_args = REGIMES[regime]
    evaluation = _get_fast_evaluation(game_str, **eval_args)
    fitness = _evaluate_fitness(evaluation, aggregation_fn=stats.hmean)

    result = {"trial": trial, "game_str": game_str, "regime": regime, "fitness": fitness,
              **evaluation}

    results.append(result)

    # Convert to dataframe and save
    results_df = pd.DataFrame(results)
    results_df.to_pickle(os.path.join("exp_outputs", "fitness_eval_comparison", "board_size_results_2.pkl"))