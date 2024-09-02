import json

from datasets import load_dataset

import os
import numpy as np
from tqdm import tqdm

import multiprocessing
from multiprocessing import Pool
from lms.java_api import StandardEvaluation

# import traceback


def scalar_reward(game: str):
    evaluation = StandardEvaluation().evaluate(game, ai_name="random", num_games=100, max_turns=500)

    if not evaluation["compilable"]:
        return 0

    if not evaluation["playable"]:
        return 0.05

    # Take Minimum-Biased Power Mean of the balance, completion and drawishness
    return (evaluation["balance"] ** (1 / 3) + evaluation["completion"] ** (1 / 3) + evaluation["drawishness"] ** (1 / 3)) / 3


def batched_evaluate(max_workers, timeout, inputs):
    # Function to handle each task
    def handle_task(input_string):
        try:
            result = pool.apply_async(scalar_reward, (input_string,)).get(timeout=timeout)
            return result
        except multiprocessing.TimeoutError:
            return 0.03
        except Exception as e:
            return 0.01

    # Create a pool of workers
    with Pool(max_workers) as pool:
        results = [handle_task(input_string) for input_string in inputs]

    results = [r if r is not None else 0 for r in results]

    return results


standard_eval = None

def evaluate_with_timeout(game, timeout):
    global standard_eval

    if standard_eval == None:
        standard_eval = StandardEvaluation()

    # print(game)

    # evaluation = standard_eval.evaluate(game, ai_name="random", num_games=100, max_turns=500, timeout=timeout)
    try:
        evaluation = standard_eval.evaluate(game, ai_name="random", num_games=100, max_turns=500, timeout=timeout)
    except Exception as e:
        print("Error - Restarting Evaluation Process:\n", e)
        # traceback.print_exc()
        standard_eval = StandardEvaluation()
        return 0.01

    if not evaluation["compilable"]:
        return 0

    if not evaluation["playable"]:
        return 0.05

    # Take Minimum-Biased Power Mean of the balance, completion and drawishness
    return (evaluation["balance"] ** (1 / 3) + evaluation["completion"] ** (1 / 3) + evaluation["drawishness"] ** (1 / 3)) / 3

def sequential_evaluate(timeout, games):
    return [evaluate_with_timeout(game, timeout) for game in games]


import re
def standardize(game: str):
    # print("standardize:", game)
    game = game.strip()
    game = re.sub(r'\s+', ' ', game)

    game = game.replace(": ", ":")

    game = game.replace("( (", "((")
    game = game.replace(") )", "))")
    game = game.replace("{ {", "{{")
    game = game.replace("} }", "}}")
    game = game.replace("{ (", "{(")
    game = game.replace("} )", "})")

    return game


if __name__ == "__main__":
    base_game_rewards = {}
    # if os.path.exists("base_game_rewards.json"):
    #     with open("base_game_rewards.json", "r") as f:
    #         base_game_rewards = json.load(f)

    descriptions = (list(load_dataset("LudiiLMs/code-llama-13b-fitm-mask-heldout-base-data")["train"]["base_game"]) +
                    list(load_dataset("LudiiLMs/code-llama-13b-fitm-mask-heldout-base-data")["val"]["base_game"]))

    # Filter out games that have already been processed
    descriptions = [game for game in descriptions if game not in base_game_rewards]

    # Parallel evaluation
    max_workers = 8
    timeout = 10
    batch_size = max_workers

    for i in (pbar := tqdm(range(0, len(descriptions), batch_size))):
        batch = descriptions[i:i + batch_size]
        # results = batched_evaluate(max_workers, timeout, batch)
        results = sequential_evaluate(timeout, batch)

        for game, reward in zip(batch, results):
            base_game_rewards[standardize(game)] = reward

        with open("base_game_rewards.json", "w") as f:
            json.dump(base_game_rewards, f)

        pbar.set_description(f"Batch Reward: {np.mean(results)}")


# Strip it
# if __name__ == "__main__":
#     base_game_rewards = {}
#     if os.path.exists("base_game_rewards.json"):
#         with open("base_game_rewards.json", "r") as f:
#             base_game_rewards = json.load(f)
#
#     for key in list(base_game_rewards.keys()):
#         base_game_rewards[standardize(key)] = base_game_rewards.pop(key)
#
#     with open("base_game_rewards.json", "w") as f:
#         json.dump(base_game_rewards, f)
