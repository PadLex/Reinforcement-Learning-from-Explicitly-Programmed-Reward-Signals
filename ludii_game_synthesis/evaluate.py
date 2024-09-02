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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "Padlex/pythia-410m-ludii-sft"
CONTEXT_SIZE = 1024

NUM_EVAL_BATCHES = 2
BATCH_SIZE = 42
# BATCH_SIZE = 4


model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "left"

model.generation_config.pad_token_id = tokenizer.pad_token_id

print(tokenizer.pad_token_id)


masked_dataset = load_dataset("Padlex/ludii_trl")

def split_game(example):
    prompt, completion = example['text'].split("### MID:")
    prompt = prompt + "### MID:"
    return {"text": prompt, "original": completion}

masked_dataset = masked_dataset.map(split_game)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="longest", ) # , padding="max_length", truncation=True, max_length=CONTEXT_SIZE

masked_dataset = masked_dataset.map(tokenize_function, batched=True)
# masked_dataset = masked_dataset.filter(lambda example: len(example['input_ids']) <= CONTEXT_SIZE - 2)
masked_dataset.set_format(type="torch")

# def data_collator(batch):
#     # Extract input_ids and attention_mask from the batch
#     input_ids = [item['input_ids'] for item in batch]
#     attention_masks = [item['attention_mask'] for item in batch]

#     # Pad the input_ids and attention_masks to the longest sequence in the batch
#     padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left')
#     padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0, padding_side='left')  # 0 since it's an attention mask

#     # Return a dictionary of tensors
#     return {
#         'input_ids': padded_input_ids,
#         'attention_mask': padded_attention_masks,
#         'original': [item['original'] for item in batch]
#     }
ds = masked_dataset["val"]
data_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) # , collate_fn=data_collator


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


model = model.to('cuda')  # Move model to GPU

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

start_batch = 0

# 64 workers: Reward time 233.60089468955994
# 16 workers: Reward time 230.73461508750916
# 1 worker: Reward time 525.1574912071228

gen_rewards = []
base_rewards = []
for i, batch in enumerate(data_loader):
    print(f"Batch {i} of {len(ds) // BATCH_SIZE}: {len(batch)}x{batch['input_ids'].size}")

    if i < start_batch:
        continue

    start_time = time.time()
    
    inputs = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')

    model.eval()
    with torch.no_grad():
        outputs = model.generate(inputs, attention_mask=attention_mask, max_length=CONTEXT_SIZE)
    
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    def split_and_trim(generated):
        prompt, generated_slice = generated.split("### MID:")
        return prompt + "### MID:" + bracket_trim(generated_slice)
    
    generated_texts = [split_and_trim(g) for g in generated_texts]

    print(f"Generation time {time.time() - start_time}")
    
    # for original, generated, outputs in zip(batch["original"], generated_texts, outputs):
    #     prompt, generated_slice = generated.split("### MID")
    #     print("\nPrompt:", prompt)
    #     print("Original:", original)
    #     print("Generated:", generated_slice)
    #     print("IDs:", outputs)

    gen_rewards += sequential_evaluate(5, [recompose_game(g) for g in generated_texts])
    # gen_rewards += batched_evaluate(4, 5, generated_texts)
    print("Generation mean:", np.mean(np.nan_to_num(gen_rewards)))
    
    base_texts = []
    for generated, original in zip(generated_texts, batch["original"]):
        prompt, _ = generated.split("### MID:")
        base_texts.append(recompose_game(prompt + "### MID:" + original))
        # print(base_texts[-1])
    
    base_rewards += sequential_evaluate(5, base_texts)
    # base_rewards += batched_evaluate(4, 5, base_texts)
    print("Base mean:", np.mean(np.nan_to_num(base_rewards)))

    print(f"Reward time {time.time() - start_time}")

    # if i >= NUM_EVAL_BATCHES:
    #     break

print(gen_rewards)
print(base_rewards)

print(np.mean(np.nan_to_num(np.array(gen_rewards)) - np.nan_to_num(np.array(base_rewards))))
        
