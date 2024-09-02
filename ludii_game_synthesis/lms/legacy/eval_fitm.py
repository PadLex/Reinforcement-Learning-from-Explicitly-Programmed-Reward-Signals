import argparse
import json
import os

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, set_seed

from java_api import Eval
from ludii_datasets import LudiiDataset
from utils import load_model_and_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default='./logs/llm_test', help="Directory to load the model from")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_return_sequences', type=int, default=1)

    # Hyperparameters
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--no_sample', action='store_true')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--typical_p', type=float, default=1)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1)
    parser.add_argument('--penalty_alpha', type=float, default=1)

    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.load_dir)

    run_args = argparse.Namespace() 
    run_args.__dict__ = json.load(open(os.path.join(args.load_dir, "run_args.json"), "r"))

    run_name = args.load_dir.split("/")[-1]
    save_filename = os.path.join("./results", run_name + ".json")

    # Load the dataset that was used during training
    dataset = LudiiDataset(tokenizer=tokenizer,
        max_length=run_args.block_size,
        dataset_type=run_args.dataset_type,
        mask_names=run_args.mask_names,
        categories=run_args.data_categories,
        sub_categories=run_args.data_subcategories,
        train_prop=run_args.train_prop,
        val_prop=run_args.val_prop,
        seed=run_args.seed)
    
    config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        typical_p=args.typical_p,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        penalty_alpha=args.penalty_alpha,

        do_sample=not args.no_sample,
        num_return_sequences=args.num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        renormalize_logits=True,
    )

    evaluator = Eval()

    with tqdm(enumerate(dataset.val['text']), total=len(dataset.val['text']), desc="Evaluating FITM accuracy") as pbar:
        data = []
        for i, data_point in pbar:
            prompt, target = data_point.split("<MID>")

            inputs = tokenizer(prompt + "<MID>", return_tensors="pt").input_ids.to(model.device)
            generation_outputs = model.generate(
                input_ids=inputs,
                generation_config=config,
                max_new_tokens=run_args.block_size, # no longer than the sequence length used to train the model
                # logits_processor=logits_processor_list,
            )

            decoded = tokenizer.batch_decode(generation_outputs, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            decoded = [d.split("<MID>")[1] for d in decoded]

            # For now, assume only one return sequence
            output = tokenizer.decode(tokenizer.encode(decoded[0]), skip_special_tokens=True)
            target = tokenizer.decode(tokenizer.encode(target), skip_special_tokens=True)

            prefix = tokenizer.decode(tokenizer.encode(prompt.split("<SUF>")[0]), skip_special_tokens=True)
            suffix = tokenizer.decode(tokenizer.encode(prompt.split("<SUF>")[1]), skip_special_tokens=True)
            game = (prefix + output + suffix).strip()

            results = {"index": i, "prefix": prefix, "suffix": suffix, "target": target,
                       "output": output, "game": game}


            results["exact_match"] = (target == output)

            eval_output = evaluator.evaluate(game)

            results["compilable"] = (eval_output != ["-1"])
            results["playable"] = (eval_output != ["-1"] and eval_output != ["-2"])

            data.append(results)

            pbar.set_postfix({"Accuracy": sum([entry["exact_match"] for entry in data]) / (i + 1),
                              "Compilable": sum([entry["compilable"] for entry in data]) / (i + 1),
                              "Playable": sum([entry["playable"] for entry in data]) / (i + 1)})
            
            # Save as json
            json.dump(data, open(save_filename, "w"), indent=4)