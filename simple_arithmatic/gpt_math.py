from typing import Optional, Tuple

from tqdm import tqdm
from transformers import GPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments, \
    TrainerCallback, PreTrainedTokenizer, TrainerState, TrainerControl

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

import wandb
import torch
import numpy as np
from datasets import Dataset
import json
import os

from simple_operations import generate_expression, solve_expression, reward_function

EXPRESSION_LENGTH = 10
CONTEXT_SIZE = 64
DATASET_SIZE = 2**8  # 2**8
COMPLETE_TOKENS = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


tokens_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '*', '=']
tokens_complete = list(range(0, 100)) + ['=']


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]", model_max_length=CONTEXT_SIZE):
        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}
        self.token_to_id = self.vocab

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            model_max_length=model_max_length
        )

    def _tokenize(self, text, **kwargs):
        # Split the text into characters, only if they are in the vocab
        return [ch for ch in text if ch in self.token_to_id]

    def _convert_token_to_id(self, token):
        return self.token_to_id[token]

    def _convert_id_to_token(self, index):
        return self.id_to_token[index]

    def get_vocab(self):
        return self.vocab

    def vocab_size(self) -> int:
        return len(self.vocab)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        filename_prefix = "" if filename_prefix is None else filename_prefix + "-"
        vocab_file = os.path.join(save_directory, f"{filename_prefix}vocab.json")
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)
        return (vocab_file,)

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return super().decode(token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False).replace(' ', '')  # TODO looks dodgy


def CompleteMathTokenizer():
    special_tokens = {"bos_token": "[BOS]", "eos_token": "[EOS]", "pad_token": "[PAD]"}

    # Combine tokens and special tokens
    all_tokens = tokens_complete + list(special_tokens.values())

    # Create the vocabulary dictionary
    vocab = {str(token): idx for idx, token in enumerate(all_tokens)}

    # Initialize the tokenizer with the WordLevel model
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

    tokenizer.pre_tokenizer = Whitespace()

    # Set special tokens
    tokenizer.add_special_tokens(
        [special_tokens["bos_token"], special_tokens["eos_token"], special_tokens["pad_token"]])

    # Post-processing to add special tokens to sequences
    tokenizer.post_processor = TemplateProcessing(
        single=f"{special_tokens['bos_token']} $A {special_tokens['eos_token']}",
        pair=f"{special_tokens['bos_token']} $A {special_tokens['eos_token']} $B:1 {special_tokens['eos_token']}:1",
        special_tokens=[
            (special_tokens['bos_token'], vocab[special_tokens['bos_token']]),
            (special_tokens['eos_token'], vocab[special_tokens['eos_token']])
        ],
    )

    # Save the tokenizer
    tokenizer.save("complete_math_tokenizer.json")

    # Load the tokenizer using PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="complete_math_tokenizer.json")

    # Define the special tokens for the fast tokenizer
    fast_tokenizer.bos_token = special_tokens['bos_token']
    fast_tokenizer.eos_token = special_tokens['eos_token']
    fast_tokenizer.pad_token = special_tokens['pad_token']

    return fast_tokenizer


def dataset_generator(size, spaces=False):
    texts = []
    input_ids = []
    attention_mask = []

    for _ in tqdm(range(size)):
        solved = solve_expression(generate_expression())

        if len(solved) > CONTEXT_SIZE - 2:  # -2 for the [BOS] and [EOS] tokens
            continue

        if spaces:
            solved = solved.replace('+', " + ").replace('=', ' = ')

        texts.append(solved)
        encoded_sample = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(solved))

        padding = CONTEXT_SIZE - 2 - len(encoded_sample)
        input_ids.append([tokenizer.pad_token_id] * padding + [tokenizer.bos_token_id] + encoded_sample + [tokenizer.eos_token_id])
        attention_mask.append([0] * padding + [1] * (len(encoded_sample) + 2))

    # Assert that the decoded text is the same as the original text
    for i in tqdm(range(min(size, 100))):
        decoded = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        assert decoded == texts[i], f"Decoded: {decoded}, Actual: {texts[i]}"

    if len(texts) < size:
        print(f"Generated {size - len(texts)} less than the requested size of {size}")

    return {"text": texts, "input_ids": input_ids, "attention_mask": attention_mask}


if __name__ == "__main__":

    tokenizer = CompleteMathTokenizer() if COMPLETE_TOKENS else CharacterTokenizer(vocab=tokens_char)

    print("vocab size:", tokenizer.vocab_size)

    # try tokenizing an expression
    example = solve_expression(generate_expression())
    print(example)
    print(tokenizer.tokenize(example))
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example)))
    print(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example)), skip_special_tokens=True))

    train_dataset = Dataset.from_dict(dataset_generator(DATASET_SIZE, COMPLETE_TOKENS))  # 2**15, 2**17
    test_dataset = Dataset.from_dict(dataset_generator(2**5, COMPLETE_TOKENS))

    print(train_dataset)

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer) + 3,
        n_ctx=CONTEXT_SIZE,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    print(config)

    model = GPT2LMHeadModel(config=config)

    print(f"Model Size: {sum(p.numel() for p in model.parameters()) / 1_000_000}M")

    x_rewards = []
    y_rewards = []

    class RewardLoggingCallback(TrainerCallback):
        def __init__(self):
            pass  # Add initialization code if needed

        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            """Called at the end of each training step."""
            # Check if the current step is a multiple of 10
            if state.global_step % 50 == 0:
                # Calculate your custom reward using some function
                reward = self.calculate_reward(kwargs['model'])

                # Log the reward using the logger available, e.g., print, wandb, etc.
                print(f"Step {state.global_step}: Reward = {reward}")
                x_rewards.append(state.global_step)
                y_rewards.append(reward)

        def rindex(self, lst, value):
            lst.reverse()
            i = lst.index(value)
            lst.reverse()
            return len(lst) - i - 1

        def calculate_reward(self, model):
            # Calculate the reward using the model over the test dataset
            rewards = []
            for sample in test_dataset:
                last_equals = self.rindex(sample["input_ids"], tokenizer.convert_tokens_to_ids("="))

                # print("Last Equals: ", last_equals)
                # print("Input IDs: ", sample["input_ids"])
                # print("Cut   IDs: ", sample["input_ids"][:last_equals + 1])

                input_ids = torch.tensor(sample["input_ids"][:last_equals + 1]).unsqueeze(0).to(model.device)
                # attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0)

                output = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_new_tokens=1) # , max_length=CONTEXT_SIZE

                predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)

                reward = reward_function(predicted_text)

                print(f"\nPredicted: {predicted_text}")
                print(f"Actual: {sample['text']}")
                print(f"Reward: {reward}")

                rewards.append(reward)

            return np.mean(rewards)


    training_args = TrainingArguments(
            num_train_epochs=1,
            output_dir='./math-results',
            auto_find_batch_size=True,
            gradient_accumulation_steps=8,
            weight_decay=0.1,
            lr_scheduler_type="cosine",
            # learning_rate=5e-5,
            learning_rate=1e-4,
            evaluation_strategy="no",
            do_eval=False,
            hub_model_id=f"gpt2-math_{CONTEXT_SIZE}_{DATASET_SIZE}_{'complete' if COMPLETE_TOKENS else 'char'}",
            push_to_hub=True,
            report_to=["wandb"],
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[RewardLoggingCallback()]
    )

    trainer.train()

    table = wandb.Table(data=[[x, y] for (x, y) in zip(x_rewards, y_rewards)], columns=["x", "y"])

    wandb.log(
        {
            "reward": wandb.plot.line(
                table, "x", "y", title="Custom Y vs X Line Plot"
            )
        }
    )