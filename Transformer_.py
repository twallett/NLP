#%%
# "Let's build GPT: from scratch, in code, spelled out." by Andrej Karpathy
# Youtube URL: https://www.youtube.com/watch?v=kCc8FmEb1nY
# From this research paper: https://arxiv.org/pdf/1706.03762.pdf

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters 

batch_size = 32
block_size = 128
max_iters = 5000
eval_interval = 500
learning_rate = 3e-03
eval_iters = 200
n_embed = 128
n_layer = 4
n_head = 4
dropout = 0.2
# ------------------

with open("input.txt", 'r', encoding= 'utf-8') as f:
    text = f.read()

# Sorted list of charcters 
characters = sorted(list(set(text)))
vocab_size = len(characters)

# Encoder/Decoder structure 
characters_to_numbers = { ch:enum for enum, ch in enumerate(characters)}
numbers_to_characters = { enum:ch for enum, ch in enumerate(characters)}

# Functions 
encoder = lambda param: [characters_to_numbers[x] for x in param]
decoder = lambda param: "".join([numbers_to_characters[x] for x in param])

# Encoding text
encoded_text = torch.tensor(encoder(text), dtype= torch.long)

# Train & test split
train_test_split = int(len(encoded_text) * 0.9)
train = encoded_text[:train_test_split]
test = encoded_text[train_test_split:]

def get_batch(split):
    # Determine which tensor to select: train or test 
    data = train if split == "train" else test
    
    # Select batch_size amount of random indexes from data 
    index = torch.randint(len(data) - block_size, (batch_size, ))
    
    # Horizontally stacking x and y block_sizes from previous index 
    x = torch.stack([data[i: i + block_size] for i in index])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in index])
    return x, y

@torch.no_grad()
def eval_losses():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias= False)
        self.query = nn.Linear(n_embed, head_size, bias= False)
        self.value = nn.Linear(n_embed, head_size, bias= False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultipleAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultipleAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 
        

class Bigram(nn.Module):
    
    def __init__(self):
        
        # super().__init__() allows you to reference the methods inside of the parent class which in this case is nn.module
        super().__init__()
        
        # Creating an embedding table (vocab_size, number of embeding dimensions)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # Each character will also get an embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # Blocks of multiplt self attention and MLP
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_layer = nn.LayerNorm(n_embed)
        # To go from embedding to logits we are going to need a linear layer
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        
    def forward(self, inputs, targets = None):
        B, T = inputs.shape
        
        # Calculating the individual scores from embedding table row vectors
        token_embedings = self.token_embedding_table(inputs) # (B, T, C)
        positional_embeddings = self.position_embedding_table(torch.arange(T))
        x = token_embedings + positional_embeddings
        x = self.blocks(x)
        x = self.ln_layer(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
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
            
            inputs_cond = inputs[:, -block_size:]
            
            logits, loss = self(inputs_cond)
            
            logits = logits[:,-1,:]
            
            probs = F.softmax(logits, dim = 1)
            
            next_input = torch.multinomial(probs, num_samples=1)
            
            inputs = torch.cat((inputs, next_input), dim=1)
        
        return inputs

model = Bigram()

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = eval_losses()
        print(f"Iteration #{iter}: training loss {losses['train']}, testing loss {losses['test']}")
    
    samplex, sampley = get_batch("train")
    
    logits, loss = model(samplex, sampley)
    
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    
    optimizer.step()
    
print(decoder(model.generate(torch.zeros((1,1), dtype= torch.long), max_new_tokens = 500)[0].tolist()))

# %%
