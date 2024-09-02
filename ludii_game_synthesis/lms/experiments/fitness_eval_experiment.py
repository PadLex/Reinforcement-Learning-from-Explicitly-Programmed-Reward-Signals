import glob
from itertools import product
import os

import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

from fitness_helpers import _get_fast_evaluation, _evaluate_fitness


TEST_GAMES = [
    "Alquerque",
    "Amazons",
    "ArdRi",
    "Ataxx",
    "Bao Ki Arabu (Zanzibar 1)",
    "Breakthrough",
    "Chess",
    "English Draughts",
    "Fanorona",
    "Gomoku",
    "Havannah",
    "Hex",
    "Knightthrough",
    "Konane",
    "Pretwa",
    "Reversi",
    "Shobu",
    "Tablut",
    "XII Scripta",
    "Yavalath"
]

NUM_TRIALS = 10
NUM_EVALS_PER_GAME = 10

RANDOM_EVAL_ARGS = {"ai_name": "Random", "num_games": NUM_EVALS_PER_GAME, "thinking_time": 0.1, "max_turns": 250}
FAST_UCT_EVAL_ARGS = {"ai_name": "UCT", "num_games": NUM_EVALS_PER_GAME, "thinking_time": 0.1, "max_turns": 250}
MED_UCT_EVAL_ARGS = {"ai_name": "UCT", "num_games": NUM_EVALS_PER_GAME, "thinking_time": 0.5, "max_turns": 250}
SLOW_UCT_EVAL_ARGS = {"ai_name": "UCT", "num_games": NUM_EVALS_PER_GAME, "thinking_time": 2, "max_turns": 250}

REGIMES = {
    "random": RANDOM_EVAL_ARGS,
    "uct_fast": FAST_UCT_EVAL_ARGS,
    "uct_med": MED_UCT_EVAL_ARGS,
    "uct_slow": SLOW_UCT_EVAL_ARGS
}

# Load the .lud files for each game
game_strs = []
for name in TEST_GAMES:
    filename = glob.glob(os.path.join("ludii_data/games/expanded", "**", f"{name}.lud"), recursive=True)[0]
    game_strs.append(open(filename, "r").read())


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
    results_df.to_pickle(os.path.join("exp_outputs", "fitness_eval_comparison", "eval_results.pkl"))