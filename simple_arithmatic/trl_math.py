import numpy as np
import torch

import wandb
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import time
from tqdm import tqdm
from datasets import Dataset
import os

from simple_operations import generate_expression, solve_expression, reward_function
from gpt_math import CharacterTokenizer, tokens_char, CompleteMathTokenizer


EXPRESSION_LENGTH = 10
CONTEXT_SIZE = 64
COMPLETE_TOKENS = True

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(torch.cuda.current_device())

def rindex(s, value):
    reversed_s = s[::-1]
    i = reversed_s.index(value)
    return len(s) - i - 1


def rl_dataset_generator(size):
    texts = []
    input_ids = []
    attention_mask = []

    for _ in tqdm(range(size)):
        solved = solve_expression(generate_expression())

        if COMPLETE_TOKENS:
            solved = solved.replace("+", " + ").replace("=", " = ")

        last_equals = rindex(solved, "=")
        solved = solved[:last_equals + 1]

        if len(solved) > CONTEXT_SIZE - 2:  # -2 for the [BOS] and [EOS] tokens
            continue

        texts.append(solved)
        encoded_sample = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(solved))

        padding = CONTEXT_SIZE - 2 - len(encoded_sample)
        input_ids.append(torch.tensor([tokenizer.pad_token_id] * padding + [tokenizer.bos_token_id] + encoded_sample, dtype=torch.long))
        attention_mask.append(torch.tensor([0] * padding + [1] * (len(encoded_sample) + 2), dtype=torch.long))

    # Assert that the decoded text is the same as the original text
    for i in tqdm(range(min(size, 100))):
        decoded = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        assert decoded == texts[i], f"Decoded: {decoded}, Actual: {texts[i]}"

    if len(texts) < size:
        print(f"Generated {size - len(texts)} less than the requested size of {size}")

    return {"query": texts, "input_ids": input_ids, "attention_mask": attention_mask}


model_name = f"Padlex/gpt2-math_{CONTEXT_SIZE}_131072_{'complete' if COMPLETE_TOKENS else 'char'}"

model_name = "Padlex/gpt2-math_64_256_complete"

config = PPOConfig(
    model_name=model_name,
    # learning_rate=5e-6,
    log_with="wandb",
    remove_unused_columns=False,
    # ppo_epochs=1, # If can't do kl,
    use_score_norm=False,
    use_score_scaling=False,
    score_clip=None,
    batch_size=128,
    mini_batch_size=128,
    # kl_penalty="full",  # Did this stabilize training?
    entropy_coef=0.3,
    # batch_entropy_coef=0.3
)


tokenizer = CompleteMathTokenizer() if COMPLETE_TOKENS else CharacterTokenizer(vocab=tokens_char)

rl_dataset = Dataset.from_dict(rl_dataset_generator(2 ** 17)) # 17

print("Mean result:", np.mean([eval(text.split('=')[0]) for text in rl_dataset["query"]]))


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name, device_map="auto")
ref_mod = None


ppo_trainer = PPOTrainer(config, model, model, tokenizer, dataset=rl_dataset, data_collator=collator)


generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True, # True
    "pad_token_id": tokenizer.pad_token_id,
    # "max_length": CONTEXT_SIZE,
    # "temperature": 2.0,
    "bad_words_ids": [[tokenizer.pad_token_id], [tokenizer.bos_token_id], [tokenizer.eos_token_id]],
}


def process_rewards(texts):
    return [torch.tensor(reward_function(text), dtype=torch.float) for text in texts]


def process_no_symbols_rewards(texts):
    def no_symbols_reward(text):
        try:
            int(text.split("=")[-1])
            return 1
        except ValueError:
            return 0

    return [torch.tensor(no_symbols_reward(text), dtype=torch.float) for text in texts]



#%%

total_generation_time = 0
total_evaluation_time = 0
total_update_time = 0
total_batches = 0

wandb.init()

for batch in tqdm(ppo_trainer.dataloader):
    total_batches += 1

    start_time = time.time()
    query_tensors = [torch.tensor(ids, dtype=torch.long).to('cuda') for ids in batch["input_ids"]]  # .to('cuda')

    #### Get response from gpt2
    response_tensors = []
    for query in query_tensors:
        generation_kwargs["max_new_tokens"] = 1
        generation_kwargs["min_new_tokens"] = 1

        query = query
        # print(type(query), query)
        response = ppo_trainer.generate(query, return_prompt=False, **generation_kwargs)
        response = response.squeeze()

        # Check if the response is zero-dimensional and adjust if necessary
        if response.dim() == 0:
            # print(f"Zero-dimensional response detected: {response}")
            response = response.unsqueeze(0)  # Add a dimension if tensor is zero-dimensional
            # print(tokenizer.decode(response))

        # print("query + response:   ", tokenizer.decode(query), "@", tokenizer.decode(response.squeeze()))
        response_tensors.append(response)

    batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    total_generation_time += time.time() - start_time
    start_time = time.time()

    #### Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    rewards = process_rewards(texts)

    tqdm.write("Rewards: " + str(rewards))
    mean_reward = torch.mean(torch.stack(rewards)).item()
    tqdm.write(f"Mean reward: {mean_reward}")

    total_evaluation_time += time.time() - start_time
    start_time = time.time()

    # print("query_tensors: ", query_tensors)
    print("response_tensors: ", response_tensors)

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    total_update_time += time.time() - start_time

    tqdm.write(
        f"Average Times - Generation: {total_generation_time / total_batches:.3f}s, Evaluation: {total_evaluation_time / total_batches:.3f}s, Update: {total_update_time / total_batches:.3f}s")


model.save_pretrained(f"trl-gpt2-math_{CONTEXT_SIZE}")
# ppo_trainer.push_to_hub("Padlex/latest_and_greatest")
wandb.finish()