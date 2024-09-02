import argparse
from collections import defaultdict
import json
import pickle
import random
import typing

from datasets import load_dataset
import inflect
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from java_api import StandardEvaluation, Autocomplete
from ludii_datasets import _extract_parentheticals, _mask_names, _format_fitm

MODEL_NAME = "LudiiLMs/code-llama-13b-fitm-mask"
TEMPERATURES = [0.5, 1.0, 1.5, 2.0]
SAMPLES_PER_TEMP = 500
AUTOCOMPLETE = Autocomplete()
USE_GRAMMAR = True


def _format_prompt(prefix: str, suffix: str):
    return f"<s><PRE> {prefix} <SUF>{suffix} <MID>"

def _format_output(output_token_ids: typing.List[int], tokenizer: AutoTokenizer):
    '''
    Convert token ids from FITM output into a string
    '''
    output = tokenizer.decode(output_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    if "<MID> " in output:
        output = output.split("<MID> ")[1]
    output = tokenizer.decode(tokenizer.encode(output), skip_special_tokens=True)

    # Newlines will break the evaluators
    output = output.replace("\n", "")

    return output

def _perform_mutation(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prefix: str,
                      middle: str, suffix: str, config: GenerationConfig, use_grammar: bool):
    '''
    Generate a single mutation from the model given the surrounding prefix and suffix
    by first sampling a continuation from the grammar, and then sampling the rest from
    the language model
    '''

    if use_grammar:
        next_tokens = AUTOCOMPLETE.next_tokens(prefix)

        # Only consider tokens that start ludemes and don't duplicate the middle
        filtered_tokens = []
        for token in next_tokens:
            if token.startswith("(") and not middle.strip().startswith(token):
                filtered_tokens.append(token)

        if len(filtered_tokens) == 0:
            return None, None
        
        next_token = random.choice(filtered_tokens)

    else:
        next_token = ""

    prompt = _format_prompt(prefix, suffix)
    prompt += f" {next_token}" if next_token else ""

    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        generation_outputs = model.generate(
            input_ids=inputs,
            generation_config=config,
            max_new_tokens=512,
        )

    outputs = [_format_output(output.cpu().tolist(), tokenizer) for output in generation_outputs]
    new_games = [f"{prefix}{output}{suffix}".strip() for output in outputs]

    del inputs
    del generation_outputs

    # Clear the CUDA cache to stop memory usage from steadily increasing
    torch.cuda.empty_cache()

    return new_games, outputs

def _select_mutation(game):
    '''
    Select a random mutation from a game, evenly distributed by depth
    '''
    parentheticals = _extract_parentheticals(game)
    parentheticals_by_depth = defaultdict(list)
    for parenthetical in parentheticals:
        depth = parenthetical[3]
        parentheticals_by_depth[depth].append(parenthetical)

    # Select a random mutation (evenly distributed by depth)
    depth = random.choice(list(parentheticals_by_depth.keys()))
    return random.choice(parentheticals_by_depth[depth])

def mutate(dataset, model, tokenizer, config, use_grammar=False):
    '''
    Mutate a game using the model
    '''

    done = False
    while not done:

        idx = random.randint(0, len(dataset['base_game']) - 1)
        game = dataset['base_game'][idx]
        name = dataset['name'][idx]

        prefix, middle, suffix, depth = _select_mutation(game)
        new_games, outputs = _perform_mutation(model, tokenizer, prefix, middle, suffix, config, use_grammar)
        if new_games is not None:
            done = True

    return new_games, outputs, name, (prefix, middle, suffix, depth)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    evaluator = StandardEvaluation()

    train_dataset = load_dataset(MODEL_NAME + "-base-data", split="train")
    val_dataset = load_dataset(MODEL_NAME + "-base-data", split="val")

    SAVE_FILENAME = "../caches/mutation_comparison_results.json" if not USE_GRAMMAR else "../caches/mutation_comparison_results_grammar.json"

    RESULTS = []

    for temp in TEMPERATURES:

        config = GenerationConfig(
            temperature=temp,
            top_k=50,
            do_sample=True,
            num_return_sequences=3,
            renormalize_logits=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        for _ in tqdm(range(SAMPLES_PER_TEMP), desc=f"Eval games at temperature {temp}"):

            new_games, outputs, name, (prefix, middle, suffix, depth) = mutate(train_dataset, model, tokenizer, config, use_grammar=USE_GRAMMAR)
            try:
                evaluations = [evaluator.evaluate(game, max_turns=5)['playable'] for game in new_games]
            except:
                evaluator = StandardEvaluation()
                evaluations = [None for game in new_games]
                print(f"Eval failed for {name}, attempting to restart evaluator and continue...")
            exact_matches = [output.strip() == middle for output in outputs]

            result = {"split": "train", "name": name, "temperature": temp, "depth": depth, "target_len": len(middle), "evaluations": evaluations, "exact_matches": exact_matches}
            RESULTS.append(result)

            new_games, outputs, name, (prefix, middle, suffix, depth) = mutate(val_dataset, model, tokenizer, config, use_grammar=USE_GRAMMAR)
            try:
                evaluations = [evaluator.evaluate(game, max_turns=5)['playable'] for game in new_games]
            except:
                evaluator = StandardEvaluation()
                evaluations = [None for game in new_games]
                print(f"Eval failed for {name}, attempting to restart evaluator and continue...")
            exact_matches = [output.strip() == middle for output in outputs]

            result = {"split": "val", "name": name, "temperature": temp, "depth": depth, "target_len": len(middle), "evaluations": evaluations, "exact_matches": exact_matches}
            RESULTS.append(result)

            # Save the results
            with open(SAVE_FILENAME, "w") as f:
                json.dump(RESULTS, f)

                