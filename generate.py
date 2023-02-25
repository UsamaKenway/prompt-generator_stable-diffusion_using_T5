from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("model-small").to('cuda')

# # Load the saved state dictionary
# state_dict = torch.load("saved_models/model_epoch1_batch2000.pt")
#
# # Load the state dictionary into the model
# model.load_state_dict(state_dict)
#
# # Set the model to evaluation mode
# model.eval()

# Generate some text
prompt = "my name is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
output = model.generate(input_ids=input_ids, max_length=50, do_sample=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
