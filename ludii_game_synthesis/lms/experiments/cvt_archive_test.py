from collections import defaultdict
import glob
import json
import os
import pickle
import sys

import numpy as np
from ribs.archives import CVTArchive
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from archives import ConceptPCAArchive
from config import ArchiveGame, VALIDATION_GAMES
from java_api import Concepts
from java_helpers import CONCEPT_DTYPES

CACHE_DIR = "./caches"
PCA_TYPE = "boolean"
USE_VALIDATION = True

# Load the concepts for the validation games
if USE_VALIDATION:
    concept_generator = Concepts()
    val_game_strs = []
    for name in VALIDATION_GAMES:
        filename = glob.glob(os.path.join("ludii_data/games/expanded", "**", f"{name}.lud"), recursive=True)[0]
        val_game_strs.append(open(filename, "r").read())

    game_concepts = []
    for game_str in val_game_strs:
        concepts = concept_generator.compile(game_str)

        if PCA_TYPE == "boolean":
            concepts = [concept for idx, concept in enumerate(concepts) if CONCEPT_DTYPES[idx] == bool]

        game_concepts.append(concepts)

else:
    concept_data = json.load(open(os.path.join(CACHE_DIR, "board_game_concepts.json"), "r"))
    game_concepts = [np.array(data["boolean_concepts"], dtype=np.int16) for data in concept_data]

for pca_dims in [2, 3, 4, 5, 6, 7]:
    for num_cells in [100, 500, 1000]:
        model_dir = os.path.join(CACHE_DIR, f"{PCA_TYPE}_concepts_pca_{pca_dims}.pkl")
        pca_model = pickle.loads(open(model_dir, "rb").read())

        transformed_concepts = pca_model.transform(game_concepts)

        cvt_archive_sizes = []
        for seed in [0, 1, 2, 3, 4, 5]:
            cvt_archive = CVTArchive(
                solution_dim=pca_dims,
                cells=num_cells,
                ranges=[(-5, 5)] * pca_dims,
                seed=seed
            )

            archive_indices = set()
            for entry in tqdm(transformed_concepts, desc="Evaluating archive (CVT)", leave=False):
                archive_index = cvt_archive.index_of_single(entry)
                archive_indices.add(archive_index)

            cvt_archive_sizes.append(len(archive_indices))


        cells_per_dim = int(num_cells ** (1 / pca_dims))
        default_archive = ConceptPCAArchive(
            only_boolean=PCA_TYPE == "boolean",
            pca_dims=pca_dims,
            cells_per_dim=cells_per_dim,
            dim_extents=[-5, 5],
            entries_per_cell=1
        )

        archive_indices = set()
        for entry in tqdm(transformed_concepts, desc=f"Evaluating archive (default, {cells_per_dim} cells per dimension)", leave=False):
            archive_index = tuple([np.digitize(c, default_archive.cell_boundaries) for c in entry])
            archive_indices.add(archive_index)

    
        print(f"[d={pca_dims}, c={num_cells}] {len(game_concepts)} games --> {np.mean(cvt_archive_sizes):.2f} unique cells (CVT) vs {len(archive_indices)} unique cells (default)")

    print("\n")



# archive_index_to_leading_category = defaultdict(list)
            # for entry, categories in tqdm(zip(transformed_concepts, game_categories), desc="Evaluating archive", leave=False):
            #     archive_index = archive.index_of_single(entry)
            #     archive_index_to_leading_category[archive_index].append(categories[1])

            # archive_size = len(archive_index_to_leading_category)
            # average_distinct_categories_per_cell = np.mean([len(set(categories)) for categories in archive_index_to_leading_category.values() if len(categories) > 1])

            # archive_sizes.append(archive_size)
            # avg_distinct_categories.append(average_distinct_categories_per_cell)

        # print(f"[d={pca_dims}, c={num_cells}] {np.mean(archive_sizes):.2f} unique cells, {np.mean(avg_distinct_categories):.2f} average distinct categories per cell")