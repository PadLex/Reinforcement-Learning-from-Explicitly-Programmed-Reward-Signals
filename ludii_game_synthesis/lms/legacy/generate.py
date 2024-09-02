import argparse
from itertools import product
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "true"
import typing

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GenerationConfig, LogitsProcessorList, pipeline, TextStreamer, TopKLogitsWarper, set_seed

from grammar_constraints import GrammarConstrainedLogitsProcessor, PICARDGrammarConstraint, MixedGrammarConstraint
from java_api import StandardEvaluation
from utils import *
from ludii_datasets import _format_instruct_without_output


def generate_samples(model,
                     tokenizer,
                     prompt: str,
                     max_length: int,
                     num_return_sequences: int,
                     temperature: float,
                     no_sample: bool,
                     top_k: int,
                     top_p: float,
                     typical_p: float,
                     num_beams: int,
                     repetition_penalty: float,
                     penalty_alpha: float,
                     remove_prompt: bool = True,
                     stop_token: typing.Optional[str] = None,
                     streamer: typing.Optional[TextStreamer] = None,
                     logits_processor_list: typing.Optional[LogitsProcessorList] = None):
    
    config = GenerationConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        typical_p=typical_p,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        penalty_alpha=penalty_alpha,

        do_sample=not no_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        renormalize_logits=True,
    )

    if not prompt.startswith(tokenizer.bos_token):
        prompt = tokenizer.bos_token + prompt
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        generation_outputs = model.generate(
            input_ids=inputs,
            generation_config=config,
            max_new_tokens=max_length,
            streamer=streamer,
            logits_processor=logits_processor_list,
        )

    output_texts = []
    for output in generation_outputs:
        text = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if stop_token: 
            text = text.replace(stop_token, "")
        # text = text[:text.find(stop_token) if stop_token else None]

        if remove_prompt:
            # Find the first occurrence of '(game' and start there
            start_idx = text.find("(game")
            if start_idx == -1:
                start_idx = 0
            
            text = text[start_idx:].strip()

        output_texts.append(text)

    return output_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Experiment override
    parser.add_argument('--exp_args', type=str, default="", 
                        help="Path to a json file which includes the arguments for a large-scale evaluation experiment")

    # Generation Arguments
    parser.add_argument('--load_dir', type=str, default='./logs/llm_test', help="Directory to load the model from")
    parser.add_argument('--prompt', type=str, default="", help="The prompt to use for generation")
    parser.add_argument('--stop_token', type=str, default="", help="The token at which generation is stopped")
    parser.add_argument('--stream', action='store_true', help="Whether to stream generation (instead of generating all at once)")
    parser.add_argument('--constrain', action='store_true', help="Whether to use grammar-constrained generation")
    parser.add_argument('--constraint_type', type=str, default="PICARD", choices=["PICARD", "autocomplete", "mixed"], help="The type of constraint to use")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--instructions', action='store_true', help="Whether to format the prompt as instructions")

    # Hyperparameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--enforce_spacing', action='store_true')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--no_sample', action='store_true')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--typical_p', type=float, default=1)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1)
    parser.add_argument('--penalty_alpha', type=float, default=1)

    args = parser.parse_args()

    # If specified, override the arguments with the ones for the current experiment
    if args.exp_args:
        for key, val in json.load(open(os.path.join(args.exp_args), "r")).items():
            setattr(args, key, val)

    # Set seed for reproducibility
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.load_dir)

    # Run the specified experiment
    if args.exp_args:

        # Create the output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Save the experiment args
        with open(os.path.join(args.output_dir, "exp_args.json"), "w") as file:
            json.dump(json.load(open(os.path.join(args.exp_args), "r")), file)

        if args.test_constrain:

            if args.constraint_type == "PICARD":
                logits_processor_list = LogitsProcessorList([PICARDGrammarConstraint(tokenizer, backoff_k=None, verbose=args.verbose)])
                
            elif args.constraint_type == "autocomplete":
                logits_processor_list = LogitsProcessorList([GrammarConstrainedLogitsProcessor(tokenizer, 
                                                                enforce_spacing=args.enforce_spacing,
                                                                verbose=args.verbose)])
                
            elif args.constraint_type == "mixed":
                logits_processor_list = LogitsProcessorList([TopKLogitsWarper(50),
                                                             MixedGrammarConstraint(tokenizer, args.verbose)])

            constraints = [None, logits_processor_list]
        
        else:
            constraints = [None]

        all_sample_data = []
        total_options = len(args.temperatures) * len(args.top_ks) * len(args.top_ps) * len(constraints)
        for temperature, top_k, top_p, constraint in tqdm(product(args.temperatures, args.top_ks, args.top_ps, constraints), desc="Running generation experiment", total=total_options):
            for batch in tqdm(range(0, args.samples_per_option, args.sequences_per_call), desc="Running batch", leave=False):
                generated_outputs = generate_samples(model, tokenizer, args.prompt, args.max_length, args.sequences_per_call, temperature, 
                                                     args.no_sample, top_k, top_p, args.typical_p, args.num_beams, args.repetition_penalty,
                                                     args.penalty_alpha, True, args.stop_token, None, constraint)

                for text in generated_outputs:
                    constraint_name = "None" if constraint is None else "Grammar"
                    sample_data = {"temperature": temperature, "top_k": top_k, "top_p": top_p, "constraint": constraint_name, "text": text}

                    all_sample_data.append(sample_data)

            df = pd.DataFrame(all_sample_data)
            df.to_pickle(os.path.join(args.output_dir, "model_samples.pkl"))

    else:
        
        if args.instructions:
            prompt = _format_instruct_without_output(args.prompt)
        else:
            prompt = args.prompt

        if args.constrain:
            if args.constraint_type == "PICARD":
                logits_processor_list = LogitsProcessorList([PICARDGrammarConstraint(tokenizer, backoff_k=25, verbose=args.verbose)])
                
            elif args.constraint_type == "autocomplete":
                logits_processor_list = LogitsProcessorList([GrammarConstrainedLogitsProcessor(tokenizer, 
                                                                enforce_spacing=args.enforce_spacing,
                                                                verbose=args.verbose)])
                
            elif args.constraint_type == "mixed":
                logits_processor_list = LogitsProcessorList([TopKLogitsWarper(50),
                                                             MixedGrammarConstraint(tokenizer, args.verbose)])
            
        else:
            logits_processor_list = None

        if args.stream:
            streamer = TextStreamer(tokenizer, skip_special_tokens=True)
        else:
            streamer = None

        generated_outputs = generate_samples(model, tokenizer, prompt, args.max_length, args.num_return_sequences, args.temperature, 
                                             args.no_sample, args.top_k, args.top_p, args.typical_p, args.num_beams, args.repetition_penalty,
                                             args.penalty_alpha, True, args.stop_token, streamer, logits_processor_list)
        
        # Save the autocomplete calls
        # np.save("./autocomplete_calls.npy", logits_processor_list[0].autocomplete_calls)
        
        evaluator = StandardEvaluation(verbose=False)
        for idx, text in enumerate(generated_outputs):
            print(f"\n[{idx}] '{text}'")

            eval_output = evaluator.raw_evaluate(text)
            compilable = False
            playable = False
            balance, completion_rate, drawishness, avg_turn_length = ["N/A"] * 4

            print(f"Eval output: {eval_output}")

            if eval_output == ["-1"]:
                pass
        
            elif eval_output == ["-2"]:
                compilable = True

            else:
                playable = True
                compilable = True
                balance, completion_rate, drawishness, avg_turn_length = [float(out) for out in eval_output]

            print(f"\tCompilable: {compilable}")
            print(f"\tPlayable: {playable}")
            print(f"\tBalance: {balance}")
            print(f"\tCompletion rate: {completion_rate}")
            print(f"\tDrawishness: {drawishness}")
            print(f"\tAvg turn length: {avg_turn_length}")