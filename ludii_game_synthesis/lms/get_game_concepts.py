import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

from ludii_datasets import _collect_games
from java_api import Concepts
from java_helpers import CONCEPT_DTYPES

concept_generator = Concepts()

BASE_PATH = "./ludii_data/games/official"
game_paths = _collect_games(BASE_PATH, categories='board')

data = []
for game_path in tqdm(game_paths, desc="Getting game concepts"):
    game_name = os.path.basename(game_path).split(".")[0]
    game_categories = game_path.split(BASE_PATH)[1].split("/")[1:-1]

    game_text = open(game_path, "r").read()
    all_game_concepts = concept_generator.compile(game_text)
    boolean_concepts = [concept for idx, concept in enumerate(all_game_concepts) if CONCEPT_DTYPES[idx] == bool]
    
    data_point = {"name": game_name, "categories": game_categories, "all_concepts": all_game_concepts, "boolean_concepts": boolean_concepts}
    data.append(data_point)

with open("./caches/board_game_concepts.json", "w") as f:
    json.dump(data, f, indent=4)