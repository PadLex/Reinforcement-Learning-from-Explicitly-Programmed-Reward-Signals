from collections import Counter
import json
import multiprocessing as mp
import os
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool
import psutil
import random
import time

from datasets import load_dataset
import inflect
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from ludii_datasets import _extract_parentheticals, _format_fitm, _mask_names, LudiiDataset
from java_api import FastTrace, StandardEvaluation, Autocomplete

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MUTATIONS_PER_GAME = 20
SAMPLES_PER_MUTATION = 5
EVALS_PER_SAMPLE = 5

def _format_output(output_token_ids: str, tokenizer: AutoTokenizer):
    '''
    Convert token ids from FITM output into a string
    '''
    output = tokenizer.decode(output_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    output = output.split("<MID> ")[1]
    output = tokenizer.decode(tokenizer.encode(output), skip_special_tokens=True)

    return output


AUTOCOMPLETE = Autocomplete()
FAST_TRACE = FastTrace()
RANDOM_EVAL = StandardEvaluation()

def eval_game(game_info, use_trace=True, use_random_eval=True):
    game_name, prefix, output, suffix, depth = game_info
    game = (prefix + output + suffix).strip()

    if use_trace:
        trace_score = FAST_TRACE.evaluate(game)
    else:
        trace_score = 0

    if use_random_eval:
        evaluation = RANDOM_EVAL.evaluate(game)
    else:
        evaluation = {}

    return game_name, prefix, output, suffix, depth, trace_score, evaluation

def simple_eval(game):
    trace_score = FAST_TRACE.evaluate(game)
    evaluation = RANDOM_EVAL.evaluate(game)

    return trace_score, evaluation

def on_spawn_eval(game):
    trace = FastTrace()
    random_eval = StandardEvaluation()
    trace_score = trace.evaluate(game)
    evaluation = random_eval.evaluate(game)

    trace.terminate()
    random_eval.terminate()

    return trace_score, evaluation

def reference_eval(game_and_evaluators):
    game, (fast_trace, random_eval) = game_and_evaluators
    trace_score = fast_trace.evaluate(game)
    evaluation = random_eval.evaluate(game)

    return trace_score, evaluation

engine = inflect.engine()

model_name = "code-llama-13b-fitm-mask"
model = AutoModelForCausalLM.from_pretrained("LudiiLMs/" + model_name)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-hf")

dataset = LudiiDataset._load_from_dir("./logs/" + model_name)
val_game_names = dataset.val_game_names
# val_game_names = load_dataset(model_name + "-dataset-names", split="val")['name']

config = GenerationConfig(
    temperature=1,
    top_k=100,
    top_p=1,
    typical_p=1,
    num_beams=1,
    num_beam_groups=1,
    repetition_penalty=1,
    diversity_penalty=0,

    do_sample=True,
    num_return_sequences=SAMPLES_PER_MUTATION,
    pad_token_id=tokenizer.eos_token_id,
    renormalize_logits=True,
)

game_strs = []
for game_name in val_game_names:
    path = Path("./ludii_data/games/expanded").rglob(game_name + ".lud").__next__().as_posix()
    game_strs.append(_mask_names(open(path, "r").read(), engine))

repeated_game_strs = game_strs

# evaluators = [(FastTrace(), RandomEvaluation()) for _ in tqdm(range(16), desc="Creating evaluators")]
# with Pool(16) as pool:
#     with tqdm(total=len(repeated_game_strs), desc="Evaluating fitnesses (pathos)", leave=True) as pbar:
#         for fitness in pool.imap(reference_eval, zip(repeated_game_strs, evaluators)):
#             pbar.update(1)

# for game_str in tqdm(repeated_game_strs, desc="Evaluating fitnesses (single core)", leave=True):
#     trace_score = FAST_TRACE.evaluate(game_str)
#     evaluation = RANDOM_EVAL.evaluate(game_str)

with mp.Pool(mp.cpu_count()) as pool:
    fitnesses = []
    with tqdm(total=len(repeated_game_strs), desc="Evaluating fitnesses (pool.imap, spawn new evaluators)", leave=True) as pbar:
        for fitness in pool.imap(on_spawn_eval, repeated_game_strs):
            pbar.update(1)
            fitnesses.append(fitness)

    breakpoint()

    with tqdm(total=len(repeated_game_strs), desc="Evaluating fitnesses (pool.imap)", leave=True) as pbar:
        for fitness in pool.imap(simple_eval, repeated_game_strs):
            pbar.update(1)

    with tqdm(total=len(repeated_game_strs), desc="Evaluating fitnesses (pool.imap_unordered)", leave=True) as pbar:
        for fitness in pool.imap_unordered(simple_eval, repeated_game_strs):
            pbar.update(1)


for fitness in map(simple_eval, tqdm(repeated_game_strs, desc="Evaluating fitnesses (map)", leave=True)):
    pass

exit()

data = []
for game_name in tqdm(val_game_names, desc="Evaluating mutations"):
    path = Path("./ludii_data/games/expanded").rglob(game_name + ".lud").__next__().as_posix()
    
    game = _mask_names(open(path, "r").read(), engine)

    initial_evals = {}
    trace_scores, evaluations = zip(*[simple_eval(game) for _ in tqdm(range(EVALS_PER_SAMPLE), desc=f"Initial evals for {game_name}", leave=True)])
    initial_evals["trace_score"] = np.mean(trace_scores)
    for key in evaluations[0].keys():
        initial_evals[key] = np.mean([eval[key] for eval in evaluations])

    parentheticals = _extract_parentheticals(game)

    selected_parentheticals = random.sample(parentheticals, MUTATIONS_PER_GAME)

    outputs = []
    for parenthetical in tqdm(selected_parentheticals, desc=f"Generating mutations for {game_name}", leave=True):
        prefix, middle, suffix, depth  = parenthetical
        prompt, _ = _format_fitm((prefix, middle, suffix, depth)).split("<MID>")

        prompt = prompt + "<MID>"

        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        generation_outputs = model.generate(
            input_ids=inputs,
            generation_config=config,
            max_new_tokens=1024
        )

        outputs += [(game_name, prefix, _format_output(output, tokenizer), suffix, depth) for output in generation_outputs]
        # outputs += [(game_name, prefix, middle, suffix, depth) for _ in range(SAMPLES_PER_MUTATION)]

    # Extend the outputs by the number of evals per sample
    outputs = outputs * EVALS_PER_SAMPLE

    with mp.Pool(16) as pool:
        with tqdm(total=len(outputs), desc=f"Evaluating mutations for {game_name}", leave=True) as pbar:
            for game_info in pool.imap_unordered(eval_game, outputs):
                game_name, prefix, output, suffix, depth, trace_score, evaluation = game_info
                id = hash((game_name, prefix, output, suffix, depth))

                data_point = {"game_name": game_name, "prefix": prefix, "suffix": suffix, "depth": depth, "output": output,
                              "trace_score": trace_score, "id": id}
                data_point.update(evaluation)

                for key in initial_evals.keys():
                    data_point["initial_" + key] = initial_evals[key]

                data.append(data_point)

                pbar.update(1)
                pbar.set_postfix({"cpu": psutil.cpu_percent(), "mem": psutil.virtual_memory().percent})

    # Save data to json
    with open(f"./exp_outputs/{model_name}_mutations.json", "w") as f:
        json.dump(data, f, indent=4)