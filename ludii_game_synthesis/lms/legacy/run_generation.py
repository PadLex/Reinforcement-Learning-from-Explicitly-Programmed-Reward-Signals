#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import json
import logging
import os

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer

from generation_utils import apply_jit, set_seed, adjust_length_to_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="Path to training output path to use for generation")

    parser.add_argument("--prompt", type=str, default="[GAME]", help="Text prompt to add to the input")
    parser.add_argument("--stop_token", type=str, default="[/GAME]", help="Token at which text generation is stopped")
    parser.add_argument("--length", type=int, default=100, help="Number of tokens to generate")

    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (lower is greedier)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--typical_p", type=float, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--penalty_alpha', type=float, default=0.6)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--stream", action="store_true", help="Whether to stream generation (instead of generating all at once)")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--jit", action="store_true", help="Whether or not to use jit trace to accelerate inference")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    # Set seed for reproducibility
    set_seed(args)

    # Look for the latest checkpoint in the supplied folder
    checkpoint_dirs = [os.path.join(args.save_dir, d) for d in os.listdir(args.save_dir) if 
                           d.startswith("checkpoint") and os.path.isdir(os.path.join(args.save_dir, d))]
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    
    if len(checkpoint_dirs) == 0:
        raise FileNotFoundError(f"Could not find any checkpoints in {args.save_dir}")

    checkpoint_path = checkpoint_dirs[-1]
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Intialize the model, map it to the device, and apply fp16 if necessary
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model.to(args.device)
    if args.fp16:
        model.half()

    max_seq_length = getattr(model.config, "max_position_embeddings", 0)
    args.length = adjust_length_to_model(args.length, max_sequence_length=max_seq_length)
    logger.info(args)

    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(args.device)

    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        typical_p=args.typical_p,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        penalty_alpha=args.penalty_alpha,

        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        renormalize_logits=True,
    )

    if args.stream:
        streamer = TextStreamer(tokenizer, skip_special_tokens=True)
        print("\nGENERATED OUTPUT: ", end="")
    else:
        streamer = None

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    if args.jit:
        model, tokenizer = apply_jit(model, tokenizer)

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=args.length + len(encoded_prompt[0]),
        generation_config=gen_config,
        streamer=streamer,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)

        # Remove all text after the stop token
        text = text[:text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)

        if not args.stream:
            print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            print("Total sequence:", total_sequence)

    return generated_sequences


if __name__ == "__main__":
    main()