
CONTEXT_SIZE = 1024
MASK_STRINGS = False
FORCE_LUDEMES = True

from tokenizers import models, trainers, decoders, pre_tokenizers, Tokenizer, processors
import os.path
from games import load_games, apply_masks
from transformers import PreTrainedTokenizerFast

mask_tokens = []
games = load_games()

if MASK_STRINGS:
    mask_tokens = ["~game~", "~piece~", "~map~", "~note~", "~region~", "~phase~", "~site~", "~tile~", "~variable~", "~track~", "~remember~", "~trigger~", "~map_entry~", "~propose~"]
    games = apply_masks(games)


print(f"Loaded {len(games)} games")
print(games.keys())
print(f"Hex: {games['Hex']}")

# from tokenizers.pre_tokenizers import Whitespace

initial_alphabet = list({char for doc in games.values() for char in doc})
# print(f"Initial alphabet {len(initial_alphabet)}:", initial_alphabet)
new_tokenizer = Tokenizer(models.BPE())
new_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# new_tokenizer.pre_tokenizer = Whitespace()
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[SEP]", "[MASK]", "[END]"] + mask_tokens, initial_alphabet=initial_alphabet)

new_tokenizer.enable_padding(length=CONTEXT_SIZE, pad_id=0)
new_tokenizer.train_from_iterator(games, trainer)
new_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
new_tokenizer.decoder = decoders.ByteLevel()

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=new_tokenizer,
    bos_token="[END]",
    eos_token="[END]",
    pad_token="[PAD]",
    mask_token="[MASK]",
    sep_token="[SEP]",
)

file_name = f"ludii_tokenizer-{CONTEXT_SIZE}-{'masked' if MASK_STRINGS else 'unmasked'}"
hf_tokenizer.save_pretrained(file_name)

hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(file_name)  # Load the tokenizer from the saved file before performing tests

print("Vocab size:", new_tokenizer.get_vocab_size(), hf_tokenizer.vocab_size)

print("Vocabulary:")
print(new_tokenizer.get_vocab())
print(hf_tokenizer.get_vocab())

games_tk = {name: new_tokenizer.encode(game).ids for name, game in games.items()}
games_tk_hf = {name: hf_tokenizer.encode(game, truncation=False, padding="max_length", max_length=CONTEXT_SIZE) for name, game in games.items()}

print(f"Mean game length: {sum(len(game) for game in games_tk.values()) / len(games_tk)}, hf: {sum(len(game) for game in games_tk_hf.values()) / len(games_tk_hf)}")
print(f"Number of games within {CONTEXT_SIZE} tokens: {sum(len(game) <= CONTEXT_SIZE for game in games_tk.values())}, hf: {sum(len(game) <= CONTEXT_SIZE for game in games_tk_hf.values())}")

games_decoded = {name: new_tokenizer.decode(game, False) for name, game in games_tk.items()}
games_decoded_hf = {name: hf_tokenizer.decode(game, skip_special_tokens=False) for name, game in games_tk_hf.items()}
print(f"Hex tokenizer    : {games_decoded['Hex']}")
print(f"Hex autotokenizer: {games_decoded_hf['Hex']}")
print(f"Hex original     : {games['Hex']}")


for original, decoded in zip(games.values(), games_decoded.values()):
    assert original == decoded.replace('[PAD]', ''), f"Decoding failed: \n{decoded} \n{original}"

print("Decoding successful")
