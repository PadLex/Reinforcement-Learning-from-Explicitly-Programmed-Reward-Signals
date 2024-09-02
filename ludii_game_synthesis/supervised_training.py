from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

masked_dataset = load_dataset("ludii_trl")

model_name = "EleutherAI/pythia-410m"  # "facebook/opt-350m"  

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

collator = DataCollatorForCompletionOnlyLM("### MID:", tokenizer=tokenizer)

sft_config = SFTConfig(output_dir="./temp", max_seq_length=1024)
trainer = SFTTrainer(
    model,  
    train_dataset=masked_dataset["train"],
    args=sft_config,
    formatting_func=lambda x: x["text"],
    data_collator=collator,
)
trainer.train()
