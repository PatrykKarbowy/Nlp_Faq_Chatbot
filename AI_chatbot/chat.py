import random
import json
import torch
from model import ChatNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
categories = data['categories']
model_state = data['model_state']
    
model = ChatNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'KlaraAI'
print('Lets chat! type QUIT to exit')

while True:
    sentence = input('You: ')
    if sentence == 'QUIT':
        break
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    category = categories[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if category == intent['category']:
                print(f'{bot_name}: {random.choice(intent["Answer"])}')
    else:
        print(f'{bot_name}: I do not understand... Try to extend Your question')