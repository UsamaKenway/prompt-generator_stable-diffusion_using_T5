import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
import os

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

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
train_dataset = PromptDataset(dataset['train'])

# Define the training parameters
batch_size = 2
num_epochs = 3
learning_rate = 5e-5
print_every = 100  # to print after how many batches
save_every = 2000  # save model every 2000 batches

model_dir = "saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Get the maximum sequence length in the dataset
max_length = max([len(tokenizer.encode(prompt)) for prompt in dataset['train']['Prompt']])
print(f"Max Length: {max_length}")

# Define a collate function to pad the input tensors to a fixed length
def collate_fn(batch, padding_value=0.0):
    input_ids = [example['input_ids'] for example in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
    return {'input_ids': input_ids}

# Create the data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

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
        if (i + 1) % save_every == 0:
            model.save_pretrained('model')
            #model_save_path = os.path.join(model_dir, f"model_epoch{epoch + 1}_batch{i + 1}.pt")
            #torch.save(model.state_dict(), model_save_path)

    # Print the average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
