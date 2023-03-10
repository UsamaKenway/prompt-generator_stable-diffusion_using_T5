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
print("trainong on: {}".format(device))
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
val_dataset = PromptDataset(dataset['test'].select(range(100)))

# df = pd.DataFrame(train_dataset.data, columns=["prompt"])
#
# # Save the DataFrame to a CSV file
# df.to_csv("train_dataset.csv", index=False)

num_rows = len(train_dataset)
print(f"Number of rows in train_dataset: {num_rows}")

# Define the training parameters
batch_size = 2
num_epochs = 100
learning_rate = 5e-5
print_every = 100  # to print after how many batches
save_every = 10  # save model every 2000 batches

# model_dir = "saved_models"
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

# # Get the maximum sequence length in the dataset
# max_length = max([len(tokenizer.encode(prompt)) for prompt in dataset['train']['Prompt']])
# print(f"Max Length: {max_length}")



# Define a collate function to pad the input tensors to a fixed length
def collate_fn(batch, padding_value=0.0):
    input_ids = [example['input_ids'] for example in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
    return {'input_ids': input_ids}

# Create the data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Start training
total_batches = len(train_loader)
print("training started")
for epoch in range(num_epochs):
    total_loss = 0.0
    #for batch in tqdm(train_loader):
    for i, batch in enumerate(tqdm(train_loader)):
        # Zero the gradients
        optimizer.zero_grad()

        # Get the inputs
        input_ids = batch['input_ids']

        # Generate the output logits
        outputs = model(input_ids=input_ids, labels=input_ids)

        # Calculate the loss
        loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), input_ids.view(-1))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update the total loss
        total_loss += loss.item()

        if (i + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}, Batch {i + 1}/{total_batches}, loss: {loss.item():.4f}")
        # if (i + 1) % save_every == 0:
        #    model.save_pretrained('model')
            #model_save_path = os.path.join(model_dir, f"model_epoch{epoch + 1}_batch{i + 1}.pt")
            #torch.save(model.state_dict(), model_save_path)

    # Print the average loss for the epoch
    avg_loss = total_loss / len(train_loader)

    with torch.no_grad():
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            input_ids = batch['input_ids']
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), input_ids.view(-1))
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

    if (epoch + 1) % save_every == 0:
        print("saving model")
        model.save_pretrained('model-small')


    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
