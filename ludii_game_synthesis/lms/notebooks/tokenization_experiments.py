import glob
import json
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.abspath('..'))
from java_api import ClassPaths

BASE_PATH = "../ludii_data/games/expanded"
game_files = glob.glob(os.path.join(BASE_PATH, "**", "*.lud"), recursive=True)

class_path_extractor = ClassPaths(bitecode_path="../ludii_fork/", verbose=False)

data = []
for path in tqdm(game_files, desc="Extracting class paths"):
    game_name = os.path.basename(path).split(".")[0]
    categories = [cat for cat in path.split(BASE_PATH)[1].split(game_name)[0].split("/") if cat != '']

    game = open(path, "r").read()
    class_paths = class_path_extractor.query(game).split(" ")

    data.append({"name": game_name, "categories": categories, "class_paths": class_paths})

# Save output to JSON file
with open("../caches/class_paths.json", "w") as f:
    json.dump(data, f, indent=2)