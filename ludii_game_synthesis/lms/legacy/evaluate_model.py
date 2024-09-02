import argparse
import os
import typing

from datasets import Dataset
import networkx as nx
import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from thefuzz import fuzz
from tqdm import tqdm

from java_api import Compile, Autocomplete, StandardEvaluation
from ludii_datasets import  get_ludii_dataset

def convert(stdout_line: str):
    if stdout_line.strip().endswith("compiled successfully"):
        return True
    elif stdout_line.strip().endswith("failed to compile"):
        return False
    else:
        return None

def check_compilability(output_games: typing.List[str], autocomplete: typing.Optional[Autocomplete] = None):
    '''
    Determine whether each of the Ludii games in the specified directory
    can be compiled
    '''

    if autocomplete is None:
        autocomplete = Autocomplete()

    compilabilities = []
    for output in tqdm(outputs, desc="Checking compilability", leave=False):
        evaluation = None

        while evaluation is None:
            try:
                autocomplete_output = autocomplete.next_tokens(output)
                if autocomplete_output == ['COMPLETE!']:
                    evaluation = 1
                else:
                    evaluation = 0

            except (SyntaxError, BrokenPipeError) as e:
                if isinstance(e, SyntaxError):
                    evaluation = 0
                else:
                    print("\nBroken pipe, restarting process...")
                    autocomplete = Autocomplete()


        compilabilities.append(evaluation)

    return compilabilities

def check_clustering(output_games: typing.List[str], k_range: typing.List[int] = [5, 8, 10, 15, 20]):
    '''
    Given a list of generated games, computes the optimal K for
    K-medoids clustering based on edit distance
    '''
    distance_matrix = np.array([[fuzz.ratio(game_1, game_2) for game_2 in output_games] for game_1 in output_games])
    
    for k in k_range:
        kmedoids = KMedoids(n_clusters=k, random_state=0, metric="precomputed",
                            init="k-medoids++", max_iter=500).fit(distance_matrix)
        print(f"[k = {k}] intertia = {kmedoids.inertia_}")

def check_diversity(output_games: typing.List[str], distinctness_threshold: float = 0.5, 
                    clique_limit: int = 100000):
    '''
    Return the (approximate) diversity of a set of samples, which is the size of the 
    largest subset of samples such that each entry in the subset is at least a
    specified edit distance from each other samples
    '''
    
    graph = nx.Graph()
    graph.add_nodes_from(range(len(output_games)))

    edges = []
    for i in range(len(output_games)):
        for j in range(i+1, len(output_games)):
            distance = 1 - (fuzz.ratio(output_games[i], output_games[j]) / 100)
            if distance >= distinctness_threshold:
                edges.append((i, j))

    graph.add_edges_from(edges)

    biggest_clique = []
    num_cliques = 0

    for clique in nx.find_cliques(graph):
        if len(clique) > len(biggest_clique):
            biggest_clique = clique
        num_cliques += 1

        if num_cliques > clique_limit:
            break


    clique_membership = np.zeros(len(output_games), dtype=np.int32)
    clique_membership[biggest_clique] = 1

    return clique_membership

def check_novelty(output_games: typing.List[str],
                  dataset: Dataset,
                  novelty_threshold: float = 0.5):

    
    novelties = []
    for output in tqdm(output_games, desc="Checking novelty", leave=False):
        distances = []
        for game in dataset:
            distance = 1 - (fuzz.ratio(output, game) / 100)
            distances.append(distance)

        novelties.append(int(min(distances) >= novelty_threshold))

    return novelties

def full_eval(output_games: typing.List[str],
              evaluator: StandardEvaluation):
    
    compilabilities = []
    playabilities = []
    balances = []
    completion_rates = []
    drawishnesses = []
    avg_turn_lengths = []

    for game in tqdm(output_games, desc="Performing full evaluation", leave=False):
        eval_output = evaluator.raw_evaluate(game)

        if eval_output == ["-1"]:
            compilabilities.append(0)
            playabilities.append(0)
            continue
    
        elif eval_output == ["-2"]:
            compilabilities.append(1)
            playabilities.append(0)
            continue

        else:
            compilabilities.append(1)
            playabilities.append(1)
            balance, completion_rate, drawishness, avg_turn_length = [float(out) for out in eval_output]

            if not np.isnan(balance): balances.append(balance)
            if not np.isnan(completion_rate): completion_rates.append(completion_rate)
            if not np.isnan(drawishness): drawishnesses.append(drawishness)
            if not np.isnan(avg_turn_length): avg_turn_lengths.append(avg_turn_length)

    return compilabilities, playabilities, balances, completion_rates, drawishnesses, avg_turn_lengths
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--base_exp_dir", type=str, default="./exp_outputs")
    
    args = parser.parse_args()

    exp_dir = os.path.join(args.base_exp_dir, args.exp_name)
    input_df = pd.read_pickle(os.path.join(exp_dir, "model_samples.pkl"))

    autocomplete = Autocomplete()
    evaluator = StandardEvaluation()

    dataset = [entry['text'] for entry in get_ludii_dataset(expand_definitions=True,
                                                            realize_options=True,
                                                            mask_names=True,
                                                            enforce_bracket_spacing=False)]

    data = []
    for key, indices in tqdm(input_df.groupby(["temperature", "top_k", "top_p", "constraint"]).groups.items(), desc="Performing evaluations"):
        temp, top_k, top_p, constraint = key
        outputs = input_df.iloc[indices]["text"].tolist()

        compilabilities, playabilities, balances, completion_rates, drawishnesses, avg_turn_lengths = full_eval(outputs, evaluator)

        prop_compilable = np.array(compilabilities).sum() / len(compilabilities)
        prop_playable = np.array(playabilities).sum() / len(playabilities)

        avg_balance = np.array(balances).mean()
        avg_completion_rate = np.array(completion_rates).mean()
        avg_drawishness = np.array(drawishnesses).mean()


        novelties = check_novelty(outputs, dataset, 0.25)
        prop_novel = np.array(novelties).sum() / len(novelties)

        diverse_set = check_diversity(outputs, 0.25)
        prop_diverse = diverse_set.sum() / len(outputs)

        combined = np.array(compilabilities) & np.array(novelties)
        prop_combined = combined.sum() / len(combined)

        compilable_games = [outputs[idx] for idx in range(len(outputs)) if compilabilities[idx]]
        restricted_diversity = "NaN" if len(compilable_games) == 0 else check_diversity(compilable_games) / len(compilabilities)

        data.append({"Constraint": constraint, "Temperature": temp, "Top-k": top_k, "Top-p": top_p,  "Compilable": prop_compilable,
                     "Playable": prop_playable, "Balance": avg_balance, "Completion Rate": avg_completion_rate, "Drawishness": avg_drawishness,
                     "Novel": prop_novel, "Diverse": prop_diverse, "Combined": prop_combined})
        

    
    output_df = pd.DataFrame(data)
    output_df = output_df.round(2)
    print(output_df)

    output_df = output_df.drop(columns=["Top-p"])
    latex = output_df.to_latex(index=True, float_format="%.2f", multirow=True)

    print(latex)

        