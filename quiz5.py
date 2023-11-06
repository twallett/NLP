#%%
# =================================================================
# Class_Ex7:
#
# The objective of this exercise to show the inner workings of Word2Vec in python using numpy.
# Do not be using any other libraries for that.
# We are not looking at efficient implementation, the purpose here is to understand the mechanism
# behind it. You can find the official paper here. https://arxiv.org/pdf/1301.3781.pdf
# The main component of your code should be the followings:
# Set your hyper-parameters
# Data Preparation (Read text file)
# Generate training data (indexing to an integer and the onehot encoding )
# Forward and backward steps of the autoencoder network
# Calculate the error
# look at error at by varying hidden dimensions and window size
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

with open("word-test.v1.txt", mode = 'r') as f:
    atext = f.read().split()

atext = atext[8:]

text = []
for token in atext:
    if token not in text or token == ':':
        text.append(token)

indx_cat = [enum for enum, token in enumerate(text) if token == ':']

np.random.seed(123)
N = 25 # window size
EPOCHS = 1000
BATCH = 32
encoder_shape = [30, 2]
decoder_shape = [30]
alpha = 1e-03

# Patches
# ------------------------------------------------------------

first_patches = [text[:N+i] for i in range(N)]
middle_patches = [text[i-N:N+i-1] for i in range(N+1, len(text) - N)]
last_patches = [text[i-N:] for i in range(len(text) - N, len(text)+1)]

patches = first_patches + middle_patches + last_patches

vocab = list(set(text))

inputs = np.zeros((len(vocab),len(text)))
    
for enum, patch in enumerate(patches):
    for word in patch:
        inputs[:,enum][vocab.index(word)] = 1

# Autoencoder
# ------------------------------------------------------------

input_shape = [inputs.shape[0]]

layers = input_shape + encoder_shape + decoder_shape + input_shape

def weight_initialization(layers):
    weights = []
    bias = []
    for i in range(0, len(layers) - 1):
        weights.append(np.random.randn(layers[i+1],layers[i]) * 0.1)
        bias.append(np.random.randn(layers[i+1],1) * 0.1)
    return weights, bias

def logistic(n):
    return 1/(1+np.exp(-n))

def forwardpropagation(input_batch, weights, bias, _):
    activations = [input_batch]
    for i in range(0, len(encoder_shape)):
        n = np.matmul(weights[i], activations[i]) + bias[i]
        a = logistic(n)
        activations.append(a)
        if _ == EPOCHS - 1 and i == len(encoder_shape) - 1:
            words.append(a)
    for i in range(len(encoder_shape), len(encoder_shape) + len(decoder_shape)+1):
        a = np.matmul(weights[i], activations[i]) + bias[i]
        activations.append(a)
    return activations

def backpropagation(input_batch, activations, weights):
    s = activations[-1] - input_batch
    gradients = [s]
    for i in range(len(encoder_shape) + len(decoder_shape), len(encoder_shape), -1):
        s = np.matmul(weights[i].T, gradients[-1])
        gradients.append(s)
    for i in range(len(encoder_shape), 0, -1):
        s = np.multiply((activations[i] * (1-activations[i])), np.matmul(weights[i].T, gradients[-1]))
        gradients.append(s)
    return gradients

weights, bias = weight_initialization(layers)

alist = []
words = []
for _ in range(EPOCHS):
    for i in range(0, inputs.shape[1], BATCH):

        input_batch = inputs[:,i:i+BATCH]

        activations = forwardpropagation(input_batch, weights, bias, _)

        loss = ((activations[-1] - input_batch)**2).mean(axis=0).mean().__round__(4)

        if i == 0 and _ % 100 == 0 and _ >= 1:
            print(f"epoch {_} | loss {loss}")
            alist.append(loss)

        gradients = backpropagation(input_batch, activations, weights)

        gradients.reverse()

        for i in range(len(layers)-1):
            weights[i] -= alpha * np.matmul(gradients[i], activations[i].T) / BATCH
            bias[i] -= alpha * np.sum(gradients[i], axis=1).reshape(-1,1) / BATCH

# Visualizations
# ------------------------------------------------------------
plt.plot(alist)
plt.show()

stack = np.hstack(x for x in words)

xs = stack[0]
ys = stack[1]

fig = go.Figure()

for enum, i in enumerate(indx_cat):
    if enum == len(indx_cat) - 1:
        fig.add_scatter(x = xs[indx_cat[enum]:],
                        y = ys[indx_cat[enum]:],
                        mode='markers',
                        text=text[indx_cat[enum]:])
        break
    fig.add_scatter(x = xs[indx_cat[enum]:indx_cat[enum+1]],
                    y = ys[indx_cat[enum]:indx_cat[enum+1]],
                    mode='markers',
                    text=text[indx_cat[enum]:indx_cat[enum+1]])

fig.show()

print(20 * '-' + 'End Q7' + 20 * '-')
# %%
