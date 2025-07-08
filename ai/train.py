import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ai.model import CharLSTM
import os

with open("dataset/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}

encoded = [char2idx[c] for c in text]

class CharDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_length])
        y = torch.tensor(self.data[idx+1:idx+self.seq_length+1])
        return x, y

# Hyperparams
seq_length = 30
hidden_size = 128
num_layers = 2
batch_size = 32
num_epochs = 20
learning_rate = 0.003

dataset = CharDataset(encoded, seq_length)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CharLSTM(len(chars), hidden_size, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    hidden = model.init_hidden(batch_size)
    for x, y in loader:
        hidden = model.init_hidden(x.size(0))  # ✅ dynamically match batch size
        out, hidden = model(x, hidden)
        loss = criterion(out, y.reshape(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save({
    "model_state": model.state_dict(),
    "char2idx": char2idx,
    "idx2char": idx2char,
    "hidden_size": hidden_size,
    "num_layers": num_layers
}, "dataset/char_model.pth")

print("Model saved.")
