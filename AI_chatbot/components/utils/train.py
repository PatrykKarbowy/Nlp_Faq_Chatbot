import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from components.utils.model import ChatNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
categories = []
xy = []

for intent in intents['intents']:
    category = intent['category']
    categories.append(category)
    for question in intent['Question']:
        w = tokenize(question)
        all_words.extend(w)
        xy.append((w, category))
        
ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
categories = sorted((set(categories)))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)    
    
    label = categories.index(tag)
    y_train.append(label) # CrossEntropyLoss
    
X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        
    # dataset[index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return  self.n_samples

# Hyperparameters
batch_size = 8
hidden_size = 64
output_size = len(categories)
input_size = len(X_train[0])
learning_rate = 0.001
epochs = 500
    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChatNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device).long()
        
        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        #backward and optimizer step
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 50 == 0:
        print(f'epoch {epoch+1}/{epochs}, loss = {loss.item():.4f}')    
        
print(f'final loss, loss = {loss.item():.4f}')

data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'categories': categories
}

FILE = 'data.pth'
torch.save(data, FILE)

print(f'training complete, files saved to {FILE}')        
