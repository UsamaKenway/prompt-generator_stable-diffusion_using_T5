import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
import os
import pandas as pd

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("training on: {}".format(device))
model.to(device)


# Define a custom dataset class
class PromptDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = self.data[index]['Prompt']
        # Tokenize the prompt and return as tensors
        input_ids = tokenizer.encode(prompt, return_tensors='pt').squeeze()
        return {'input_ids': input_ids.to(device)}

# Load the dataset
from datasets import load_dataset
dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts")
train_dataset = PromptDataset(dataset['train'].select(range(100)))

num_rows = len(train_dataset)
print(f"Number of rows in train_dataset: {num_rows}")

# Define the training parameters
batch_size = 2
num_epochs = 100
learning_rate = 5e-5
print_every = 100  # to print after how many batches
save_every = 10  # save model every 2000 batches

# Define a collate function to pad the input tensors to a fixed length
def collate_fn(batch, padding_value=0.0):
    input_ids = [example['input_ids'].cpu() for example in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
    return {'input_ids': input_ids.pin_memory()}

# Create the data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Define Trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    save_total_limit=1,
    save_steps=5000,
    logging_dir='./logs',
    logging_steps=100,
    learning_rate=learning_rate,
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    optimizers=(optimizer, None),
    train_dataloader=train_loader,
    callbacks=None
)

# Start training
trainer.train()
