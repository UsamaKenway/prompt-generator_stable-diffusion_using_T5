import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import os

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the dataset
dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")

# Define the training arguments
batch_size = 2
num_epochs = 3
learning_rate = 5e-5
print_every = 100  # to print after how many batches
save_every = 2000  # save model every 2000 batches

model_dir = "saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Get the maximum sequence length in the dataset
max_length = max([len(tokenizer.encode(prompt)) for prompt in dataset['Prompt']])
print(f"Max Length: {max_length}")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    logging_steps=print_every,
    save_steps=save_every,
    save_total_limit=2,
)

# Define a function to tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["Prompt"], truncation=True, padding="max_length")

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)

# Define the data collator
data_collator = lambda examples: {
    "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in examples]),
    "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in examples]),
    "labels": torch.stack([torch.tensor(x["input_ids"]) for x in examples]),
}

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    compute_metrics=None,
)

# Start training
print("Training started")
trainer.train()

# Save the model
trainer.save_model(model_dir)
