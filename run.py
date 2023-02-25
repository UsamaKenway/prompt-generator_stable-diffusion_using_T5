from datasets import load_dataset
import pandas as pd
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
#from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts")

train_dataset = dataset['train']
valid_dataset = dataset['test']

model_name = 't5-small'

tokenizer = T5Tokenizer.from_pretrained(model_name)

# Tokenize the training set
train_encodings = tokenizer(dataset['train']['Prompt'], truncation=True, padding=True)
device = 'cuda'
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).train()

# Define the training arguments
num_train_epochs = 3
per_device_train_batch_size = 4
warmup_steps = 500
gradient_accumulation_steps = 4
learning_rate = 1e-4


# Define the data loader
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_encodings['attention_mask']))
train_loader = DataLoader(train_dataset, batch_size=per_device_train_batch_size, shuffle=True)

# Define the optimizer and scheduler
total_steps = len(train_loader) * num_train_epochs // gradient_accumulation_steps
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.05)


# Start the training loop
for epoch in range(num_train_epochs):
    model.train()
    train_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        # Prepare the inputs and targets
        input_ids, attention_mask = batch
        target_ids = input_ids.clone()

        # Move the inputs and targets to the GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_ids = target_ids.to(device)

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)

        # Compute the loss
        loss = outputs.loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()
        scheduler.step()

        # Accumulate the training loss
        train_loss += loss.item()

        # Log the training loss
        if (i+1) % gradient_accumulation_steps == 0:
            avg_train_loss = train_loss / gradient_accumulation_steps
            batch_num = i+1
            total_batch = len(train_loader)
            #print(f"Epoch {epoch+1}, Batch {batch_num}/{total_batch}, Train Loss: {avg_train_loss:.4f}")
            train_loss = 0

            if batch_num % 100 == 0:
              print(f"Epoch {epoch+1}, Batch {batch_num}/{total_batch}, Train Loss: {avg_train_loss:.4f}")
            # Print progress after every 100 batches
            if batch_num % 1000 == 0:
                model.save_pretrained('./t5-base-finetuned')

# Save the trained model
model.save_pretrained('./t5-base-finetuned_complete')
