import argparse
from collections import defaultdict
from functools import partial
import glob
import itertools
import json
import multiprocessing as mp
import os
import pickle
import typing

from datasets import load_dataset
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from evolution import ArchiveGame, SelectedConceptArchive, ConceptPCAArchive
from fitness_helpers import _get_fast_evaluation, _close_fast_evaluation, _evaluate_fitness
from java_api import StandardEvaluation, FastTrace, Novelty
from java_helpers import SEMANTIC_BEHAVIORAL_CHARACTERISTICS, SYNTACTIC_BEHAVIORAL_CHARACTERISTICS
from utils import pretty_format_single_line_game

if __name__ == '__main__':
    DATA_DIR = "./exp_outputs"

    folders = list(sorted([folder for folder in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, folder))]))
    prompt = "\n".join([f"[{i}]: {folder}" for i, folder in enumerate(folders)])
    selection = int(input(f"Select a folder to evaluate:\n{prompt}\n"))

    selected_path = os.path.join(DATA_DIR, folders[selection])

    print(f"Loading run from {selected_path}...")

    run_args = argparse.Namespace() 
    run_args.__dict__ = json.load(open(os.path.join(selected_path, "run_args.json"), "r"))

    # Load run stats JSON into dataframe
    run_stats = pd.read_json(os.path.join(selected_path, 'run_stats.json'))

    # Load the final archive state
    last_epoch = -1
    for path in os.listdir(selected_path):
        if path.startswith('archive'):
            epoch = int(path.split('_')[1].split('.')[0])
            if epoch > last_epoch:
                last_epoch = epoch

    print(f"Loading archive from epoch {last_epoch}...")
    with open(os.path.join(selected_path, f'archive_{last_epoch}.pkl'), 'rb') as f:
        archive = pickle.load(f)

    print(f"Loading dataset used during training...")
    dataset_train = load_dataset(run_args.model_name + '-base-data', split='train', token=os.environ['HF_TOKEN'])
    dataset_val = load_dataset(run_args.model_name + '-base-data', split='val', token=os.environ['HF_TOKEN'])
    dataset_test = load_dataset(run_args.model_name + '-base-data', split='test', token=os.environ['HF_TOKEN'])

    print(f"Initializing dummy archive with correct behavioral characteristics...")
    # Create the archive
    if run_args.archive_type == "selected_concept":

        # Determine the behavioral characteristics to use
        if run_args.bc_type == "syntactic":
            bc_concepts = SYNTACTIC_BEHAVIORAL_CHARACTERISTICS
        elif run_args.bc_type == "semantic":
            bc_concepts = SEMANTIC_BEHAVIORAL_CHARACTERISTICS
        elif run_args.bc_type == "combined":
            bc_concepts = SYNTACTIC_BEHAVIORAL_CHARACTERISTICS + SEMANTIC_BEHAVIORAL_CHARACTERISTICS
        else:
            raise ValueError(f"Behavioral characteristic type {run_args.bc_type} not recognized")
        
        dummy_archive = SelectedConceptArchive(bc_concepts, run_args.entries_per_cell)

    elif run_args.archive_type == "pca":
        dummy_archive = ConceptPCAArchive(only_boolean=not run_args.use_all_concepts,
                                          pca_dims=run_args.pca_dims,
                                          cells_per_dim=run_args.cells_per_dim,
                                          dim_extents=run_args.dim_extents,
                                          entries_per_cell=run_args.entries_per_cell)

    train_cells = set([dummy_archive._get_cell(ArchiveGame(game['base_game'], 0, {}, 0, [], 0, game['name'])) for game in tqdm(dataset_train, desc="Determinings cells for training games")])
    val_cells = set([dummy_archive._get_cell(ArchiveGame(game['base_game'], 0, {}, 0, [], 0, game['name'])) for game in tqdm(dataset_val, desc="Determinings cells for validation games")])
    test_cells = set([dummy_archive._get_cell(ArchiveGame(game['base_game'], 0, {}, 0, [], 0, game['name'])) for game in tqdm(dataset_test, desc="Determinings cells for testing games")])

    occupied_cells = {
        'train': train_cells,
        'val': val_cells,
        'test': test_cells,
        'any': train_cells.union(val_cells).union(test_cells)
    }

    print("Initializion complete!\n\n" + "=" * 100 + "\n")

    # For each split / all splits combined, contains all games which aren't in cells covered by that split
    novel_cells = defaultdict(list)
    for cell, games in archive.items():
        if cell not in train_cells:
            novel_cells['train'].append(cell)
        if cell not in val_cells:
            novel_cells['val'].append(cell)
        if cell not in test_cells:
            novel_cells['test'].append(cell)

        if cell not in train_cells and cell not in val_cells and cell not in test_cells:
            novel_cells['any'].append(cell)

    total_games_in_archive = sum([len(games) for games in archive.values()]) 

    print(f"Archive contains {total_games_in_archive} games in {len(archive)} cells")
    for split in ['train', 'val', 'test', 'any']:
        unoccupied_cells = novel_cells[split]
        n_games = sum([len(archive[cell]) for cell in unoccupied_cells])
        print(f"-{n_games} games not in any of the {len(occupied_cells[split])} cells covered by {split} split")

    print("\n" + "=" * 100 + "\n")

    novel_game_evaluations = []
    novel_games = sum([archive[cell] for cell in novel_cells['any']], [])
    novel_game_strs = [game.game_str for game in novel_games]
    save_filename = os.path.join(selected_path, "post_run_evaluations.json")

    if os.path.exists(save_filename):
        print(f"Loading existing evaluations from {save_filename}...")
        with open(save_filename, 'r') as f:
            novel_game_evaluations = json.load(f)
        existing_game_strs = [game['game_str'] for game in novel_game_evaluations]
    else:
        existing_game_strs = []

    to_evaluate = [game for game in novel_games if game.game_str not in existing_game_strs]

    print("Loading novelty evaluator...")
    novelty = Novelty()
    novelty.load_game_library(dataset_train['base_game'] + dataset_val['base_game'] + dataset_test['base_game'])

    for game in tqdm(to_evaluate, desc="Evaluating fitnesses"):
        post_evaluation = _get_fast_evaluation(game.game_str, ai_name="UCT", thinking_time=0.5, timeout_duration=300)        
        post_evaluation["original_evaluation"] = game.evaluation
        post_evaluation["novelty"] = novelty.evaluate(game.game_str)

        _close_fast_evaluation(game.game_str)

        novel_game_evaluations.append(post_evaluation)
        with open(save_filename, 'w') as f:
            json.dump(novel_game_evaluations, f, indent=4)


    print("UCT evaluation complete, comparing fitness scores and novelty...")
    for evaluation in novel_game_evaluations:
        fitness = _evaluate_fitness(evaluation, stats.hmean)
        evaluation["fitness"] = fitness
        evaluation["old_fitness_avg"] = _evaluate_fitness(evaluation["original_evaluation"], stats.hmean)

    os.makedirs(os.path.join(selected_path, "best_games"), exist_ok=True)

    fitness_sorting = np.argsort([evaluation["fitness"] for evaluation in novel_game_evaluations])[::-1]
    games_by_fitness = [novel_games[i] for i in fitness_sorting]
    evals_by_fitness = [novel_game_evaluations[i] for i in fitness_sorting]
    cells_by_fitness = [dummy_archive._get_cell(game) for game in games_by_fitness]

    zipped = list(zip(games_by_fitness, evals_by_fitness, cells_by_fitness))

    for idx, (game, evaluation, cell) in enumerate(zipped):
        formatted = pretty_format_single_line_game(evaluation["game_str"])

        original_game_name = game.original_game_name
        filename = glob.glob(os.path.join("ludii_data/games/expanded", "**", f"{original_game_name}.lud"), recursive=True)[0]
        oringal_game_str = open(filename, "r").read()
        dummy_game = ArchiveGame(oringal_game_str, 0, {}, 0, [], 0, original_game_name)
        original_game_cell = dummy_archive._get_cell(dummy_game)
        
        info = "\n".join([
            f"Fitness: {evaluation['fitness']}",
            f"Old fitness: {evaluation['old_fitness_avg']}",
            f"Novelty: {evaluation['novelty']}",
            f"Cell: {cell}",
            f"Original game: {original_game_name}",
            f"Original game cell: {original_game_cell}",
        ])

        text = f"{formatted}\n\n{info}"
        with open(os.path.join(selected_path, "best_games", f"game_{idx}.txt"), 'w') as f:
            f.write(text)

        if idx == 19:
            break

    print("Done!")