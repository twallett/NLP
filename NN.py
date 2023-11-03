#%%
# =================================================================
# Class_Ex1:
# Lets consider the 2 following sentences
# Sentence 1: I am excited about the perceptron network.
# Sentence 2: we will not test the classifier with real data.
# Design your bag of words set and create your input set.
# Choose your BOW words that suits perceptron network.
# Design your classes that Sent 1 has positive sentiment and sent 2 has a negative sentiment.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

corpus =  ["I am excited about the perceptron network.", "we will not test the classifier with real data."]

targets = [1,-1]

# 2 vectors where each position has the frequency of the word in the vocabulary 

vocab = list(set([word for sent in corpus for word in sent.split()]))

pos = [1 if x in corpus[0].split() else -1 for x in vocab]
neg = [-1 if x in corpus[1].split() else 1 for x in vocab]


print(20 * '-' + 'End Q1' + 20 * '-')


# =================================================================
# Class_Ex2_1:

# For preprocessing, the text data is vectorized into feature vectors using a bag-of-words approach.
# Each sentence is converted into a vector where each element represents the frequency of a word from the vocabulary.
# This allows the textual data to be fed into the perceptron model.

# The training data consists of sample text sentences and corresponding sentiment labels (positive or negative).
# The text is vectorized and used to train the Perceptron model to associate words with positive/negative sentiment.

# For making predictions, new text input is vectorized using the same vocabulary. Then the Perceptron model makes a
# binary prediction on whether the new text has positive or negative sentiment.
# The output is based on whether the dot product of the input vector with the trained weight vectors is positive
# or negative.

# This provides a simple perceptron model for binary sentiment classification on textual data. The vectorization
# allows text to be converted into numerical features that the perceptron model can process. Overall,
# it demonstrates how a perceptron can be used for an NLP text classification task.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2_1' + 20 * '-')

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.weights = np.zeros(shape=(1, X.shape[1]))
        self.bias = 0
        error_list = []
        for _ in range(self.n_iters):
            for i in range(len(X)):
                a = -1 if (np.matmul(self.weights,X[i]) + self.bias) < 0  else 1
                e = y[i] - a 
                error_list.append(e)
                self.weights += (e * X[i])
                self.bias += e
        return error_list

    def predict(self, X):
        return -1 if (np.matmul(self.weights,X[i]) + self.bias) < 0 else 1

# Sample training data
X_train = np.array([
    "I loved this movie, it was so much fun!",
    "The food at this restaurant is not good. Don't go there!",
    "The new iPhone looks amazing, can't wait to get my hands on it."
])

y_train = np.array([1, -1, 1])

# --- 

vocab = list(set([word for sent in X_train for word in sent.split()]))

pos = [1 if x in X_train[0].split() else -1 for x in vocab]
neg = [-1 if x in X_train[1].split() else 1 for x in vocab]
pos2 = [1 if x in X_train[2].split() else -1 for x in vocab]

X_train_vec = np.array([pos, neg, pos2])

model = Perceptron()

error = model.fit(X_train_vec,y_train)

plt.plot(error)
plt.show()

print(20 * '-' + 'End Q2_1' + 20 * '-')


# =================================================================
# Class_Ex6:

# Follow the below instruction for writing the auto encoder code.

#The code implements a basic autoencoder model to learn word vector representations (word2vec style embeddings).
# It takes sentences of words as input and maps each word to an index in a vocabulary dictionary.

#The model has an encoder portion which converts word indexes into a low dimensional embedding via a learned weight
# matrix W1. This embedding is fed through another weight matrix W2 to a hidden layer.

#The decoder portion maps the hidden representation back to the original word index space via weight matrix W3.

#The model is trained to reconstruct the original word indexes from the hidden embedding by minimizing the
# reconstruction loss using backpropagation.

#After training, the weight matrix W1 contains the word embeddings that map words in the vocabulary to dense
# vector representations. These learned embeddings encode semantic meaning and can be used as features for
# downstream NLP tasks.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')





print(20 * '-' + 'End Q6' + 20 * '-')
# %%
