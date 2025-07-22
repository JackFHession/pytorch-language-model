import torch
from ai.model import CharLSTM
import random

data = torch.load("dataset/char_model.pth")
char2idx = data["char2idx"]
idx2char = data["idx2char"]

model = CharLSTM(len(char2idx), data["hidden_size"], data["num_layers"])
model.load_state_dict(data["model_state"])
model.eval()

def generate_text(start_str="hello", length=100):
    input_seq = torch.tensor([char2idx[c] for c in start_str], dtype=torch.long).unsqueeze(0)
    hidden = model.init_hidden(1)

    generated = list(start_str)
    for _ in range(length):
        out, hidden = model(input_seq[:, -1:].clone(), hidden)
        probs = torch.softmax(out[-1], dim=0)
        char_idx = torch.multinomial(probs, 1).item()
        generated.append(idx2char[char_idx])
        input_seq = torch.cat([input_seq, torch.tensor([[char_idx]])], dim=1)

    return ''.join(generated)

user_input = input("You: ")
#user_input = f"The user says {user_input}. You say "
print("LLM says:", generate_text(user_input.lower(), 200))
