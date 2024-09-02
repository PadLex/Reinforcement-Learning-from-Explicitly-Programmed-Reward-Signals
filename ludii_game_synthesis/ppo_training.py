from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, StoppingCriteria
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from base_rewards import batched_evaluate, standardize, sequential_evaluate
import numpy as np
import os
from tqdm import tqdm
import time
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import concatenate_datasets



os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "Padlex/pythia-410m-ludii-sft"
CONTEXT_SIZE = 1024

NUM_EVAL_BATCHES = 2


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "left"

print(tokenizer.pad_token_id)


masked_dataset = load_dataset("Padlex/ludii_trl")

def split_game(example):
    prompt, completion = example['text'].split("### MID:")
    prompt = prompt + "### MID:"
    return {"query": prompt, "original": completion}

masked_dataset = masked_dataset.map(split_game)

print("before:", masked_dataset)
masked_dataset = masked_dataset.filter(lambda example: len(example["original"]) >= len(example["query"])/5)
print("after:", masked_dataset)

def tokenize_function(examples):
    return tokenizer(examples["query"], padding="longest", ) # , padding="max_length", truncation=True, max_length=CONTEXT_SIZE

masked_dataset = masked_dataset.map(tokenize_function, batched=True)
masked_dataset = masked_dataset.filter(lambda example: len(example['input_ids']) <= CONTEXT_SIZE - 2)



masked_dataset.set_format(type="torch")

ds = concatenate_datasets([masked_dataset["train"], masked_dataset["val"]])
# data_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) # , collate_fn=data_collator
print(ds)

def recompose_game(text: str):
    # print("\nbefore:", text)
    text = text.replace("[PAD]", "").replace("<s>", "").replace("</s>", "").strip()

    # print("\ntext:")
    # print(text)

    if text.count("### PRE:") != 1 or text.count("### SUF: ") != 1 or text.count("### MID:") != 1:
        return None

    _, text = text.split("### PRE:")

    prefix, text = text.split("### SUF: ")

    suffix, mid = text.split("### MID:")

    # print("game 1: ", prefix + mid + suffix)

    game = prefix + mid + suffix

    # print("game 2: ", game)

    return game


# model = model.to('cuda')  # Move model to GPU

def bracket_trim(sample):
    round = 0
    curly = 0
    for i, c in enumerate(sample):
        if c == "(":
            round += 1
        elif c == ")":
            round -= 1
        elif c == "{":
            curly += 1
        elif c == "}":
            curly -= 1
        
        if (c == ")" or c == "}") and round == 0 and curly == 0:
            return sample[:i+1]
    
    return sample

def split_and_trim(generated):
        prompt, generated_slice = generated.split("### MID:")
        return prompt + "### MID:" + bracket_trim(generated_slice)
    
    
def get_ludii_rewards(texts):    
    games = [recompose_game(split_and_trim(text)) for text in texts]
    new_rewards = sequential_evaluate(5, games)

    # Initial filtering for None values
    rewards = [r if r is not None else 0 for r in new_rewards]
    print("After filtering None:", rewards)

    # Filtering out NaN values
    rewards = [r if not np.isnan(r) else 0 for r in rewards]
    print("After filtering NaNs:", rewards)

    # Filtering out infinity values
    rewards = [r if not np.isinf(r) else 1 for r in rewards]
    print("After filtering infinities:", rewards)

    # Convert to PyTorch tensors
    return [torch.tensor(r, dtype=torch.float) for r in rewards]
    

start_batch = 0

config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    log_with="wandb",
    remove_unused_columns=False,
    # ppo_epochs=1, # If can't do kl
    use_score_norm=False,
    use_score_scaling=False,
    score_clip=None,
    batch_size=12,  # 8 or 2
    mini_batch_size=6 # 4 or 1
)


generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_length": 1024,
}

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name, device_map="auto")

ref_model = model

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=ds, data_collator=collator)

for e in range(10):
    print(f"Epoch {e}")
    for i, batch in enumerate(ppo_trainer.dataloader):
        print(f"Batch {i} of {len(ds) // config.batch_size}")
        # print(f"Batch {i} of {len(ds) // BATCH_SIZE}: {len(batch)}x{batch['input_ids'].size}")
    
        # print(batch.shape)
    
        if i < start_batch:
            continue
    
        start_time = time.time()
        
        # inputs = batch['input_ids'].to('cuda')
        # attention_mask = batch['attention_mask'].to('cuda')
    
        response_tensors = ppo_trainer.generate(
                batch['input_ids'],
                return_prompt=False,
                **generation_kwargs,
            )
        
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
        generated_texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    
        print(f"Generation time {time.time() - start_time}")
    
        rewards = get_ludii_rewards(generated_texts)
        # gen_rewards += batched_evaluate(4, 5, generated_texts)
        print("rewards:", rewards)
        print("Generation mean:", np.mean(np.nan_to_num(rewards)))
        
        print(f"Reward time {time.time() - start_time}")
    
    
        #### Run PPO step
        stats = ppo_trainer.step(batch['input_ids'], response_tensors, rewards)
        
        ppo_trainer.log_stats(stats, batch, rewards)


print(gen_rewards)
print(base_rewards)

print(np.mean(np.nan_to_num(np.array(gen_rewards)) - np.nan_to_num(np.array(base_rewards))))
        

trainer.train()
trainer.save_model(model_name + "-ppo")
trainer.push_to_hub(model_name + "-ppo")

wandb.finish()