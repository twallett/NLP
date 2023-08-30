#%%

# "Let's build GPT: from scratch, in code, spelled out." by Andrej Karpathy
# Youtube URL: https://www.youtube.com/watch?v=kCc8FmEb1nY
# From this research paper: https://arxiv.org/pdf/1706.03762.pdf

#%%
# Loading the .txt file

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

#%%
# Reading the data from .txt file

with open("input.txt", 'r', encoding= 'utf-8') as f:
    text = f.read()
    
#%%
# Amount of characters 

print(f"Length of the characters: {len(text)}")

#%%
# Sorted list of charcters 

characters = sorted(list(set(text)))
vocab_size = len(characters)
print("".join(characters))
print(vocab_size)

#%%
# Encoder/Decoder structure 

characters_to_numbers = { ch:enum for enum, ch in enumerate(characters)}
numbers_to_characters = { enum:ch for enum, ch in enumerate(characters)}

# Functions 
encoder = lambda param: [characters_to_numbers[x] for x in param]
decoder = lambda param: "".join([numbers_to_characters[x] for x in param])

# Example
print(encoder("pretty cool, huh?"))
asentence = encoder("pretty cool, huh?")
print(decoder(asentence))

#%%
# Hello pytorch! & encoding text

import torch

encoded_text = torch.tensor(encoder(text), dtype= torch.long)

#%%
# Train & test split

train_test_split = int(len(encoded_text) * 0.9)
train = encoded_text[:train_test_split]
test = encoded_text[train_test_split:]

#%%
# batch_size and block_size & batch function
# batch_size: Amount of independent sequence of characters that will be processed in parallel
# block_size: Amount of characters looking back... in order to make predictions

torch.manual_seed(123)

batch_size = 4
block_size = 8

def get_batch(split):
    # Determine which tensor to select: train or test 
    data = train if split == "train" else test
    
    # Select batch_size amount of random indexes from data 
    index = torch.randint(len(data) - block_size, (batch_size, ))
    
    # Horizontally stacking x and y block_sizes from previous index 
    x = torch.stack([data[i: i + block_size] for i in index])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in index])
    return x, y

samplex, sampley = get_batch("train")

#%%
# Building the bigram model

import torch.nn as nn
from torch.nn import functional as F 

torch.manual_seed(123)

class Bigram(nn.Module):
    
    def __init__(self, vocab_size):
        
        # super().__init__() allows you to reference the methods inside of the parent class which in this case is nn.module
        super().__init__()
        
        # Creating an embedding table 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, inputs, targets = None):
        
        # Calculating the individual scores from embedding table row vectors
        logits = self.token_embedding_table(inputs) # (B, T, C)
        
        if targets is None:
            
            loss = None
            
        else:
                
            # Reshaping to calculate loss
            B, T, C  = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            
            # Calculating loss
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, inputs, max_new_tokens):
        
        for _ in range(max_new_tokens):
            
            logits, loss = self(inputs)
            
            logits = logits[:,-1,:]
            
            probs = F.softmax(logits, dim = 1)
            
            next_input = torch.multinomial(probs, num_samples=1)
            
            inputs = torch.cat((inputs, next_input), dim=1)
        
        return inputs
    
model = Bigram(vocab_size)

output, loss = model(samplex, sampley)

print(output)
print(loss)
print(decoder(model.generate(torch.zeros((1,1), dtype= torch.long), max_new_tokens = 500)[0].tolist()))

#%%
# Training the bigram model

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-03)

#%%
# Evaluating the model

batch_size = 32

for steps in range(10000):
    
    samplex, sampley = get_batch("train")
    
    logits, loss = model(samplex, sampley)
    
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    
    optimizer.step()
    
print(loss.item())
print(decoder(model.generate(torch.zeros((1,1), dtype= torch.long), max_new_tokens = 500)[0].tolist()))

#%%
# Mathematical trick for self-attention

B, T, C = 4, 8, 2
x = torch.randn((B,T,C))
print(x)
# %%

# version 3: using softmax 

tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
xbow = wei @ x

# %%

# version 4: self-attention

B, T, C = 4, 8, 2
x = torch.randn((B,T,C))

# Single head of self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)

wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T)

tril = torch.tril(torch.ones(T,T))
# wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
# xbow = wei @ x
v = value(x)
out = wei @ v

# %%
