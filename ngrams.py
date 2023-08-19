#%%
from collections import Counter

text = open('BERP.txt', 'r').read()

def hm_whitespace_tokenizer(text_):
    
    letters = []
    words_ = []
    len_word = 0
    len_list = 0
    for letter in text_:
        if letter == " ":
            words_.append("".join(letters[len_list - len_word:len_list]))
            len_word = 0
        if letter != " ":
            letters.append(letter)
            len_word += 1
            len_list += 1
            
    return words_

words = hm_whitespace_tokenizer(text)

counts = Counter(words).most_common()

#%%

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def unigram(words_, input_, smoothing = None):

    sentence = input_.split()

    n = len(sentence)

    A = np.zeros((n,1))

    if smoothing == None:
        
        for column in range(n):
            numerator = sentence[column]
            
            num_count = words_.count(numerator)
            
            A[column] = num_count/len(words_)

        sns.heatmap(A.T, annot=True, xticklabels=[i for i in sentence])
        plt.title(f"Uni-gram of '{input}''")
        plt.show()
    
    if smoothing == 'laplace':
        
        vocabulary = []
        for i in words:
            if i != vocabulary:
                vocabulary.append(i)
     
        for column in range(n):
            numerator = sentence[column]
            
            num_count = words_.count(numerator)
            
            A[column] = (1 + num_count)/(len(words_) + len(vocabulary))

        sns.heatmap(A.T, annot=True, xticklabels=[i for i in sentence])
        plt.title(f"Uni-gram of '{input}''")
        plt.show()
        
    return A

A_unigram = unigram(words, 
                    "i'd like to eat [uh] good food", 
                    smoothing='laplace')

#%%

def bigram(words_, input_, smoothing = None):

    sentence = input_.split()

    n = len(sentence)

    A = np.zeros((n,n))

    if smoothing == None:
        for row in range(n):
            for column in range(n):

                denominator = sentence[row]
                numerator = " ".join([sentence[row], sentence[column]])

                den_count = words_.count(denominator)

                num_count_list = []
                for enum, i in enumerate(words_):
                    try:
                        num_count_list.append(words_[enum] + " " + words_[enum+1])
                    except IndexError:
                        continue

                num_count = num_count_list.count(numerator)

                A[row, column] = num_count / den_count

        sns.heatmap(A, annot=True, xticklabels=[i for i in sentence], yticklabels=[i for i in sentence])
        plt.title(f"Bi-gram of '{input}''")
        plt.show()
        
    if smoothing == 'laplace':
        vocabulary = []
        for i in words:
            if i != vocabulary:
                vocabulary.append(i)

        for row in range(n):
            for column in range(n):

                denominator = sentence[row]
                numerator = " ".join([sentence[row], sentence[column]])

                den_count = words_.count(denominator)

                num_count_list = []
                for enum, i in enumerate(words_):
                    try:
                        num_count_list.append(words_[enum] + " " + words_[enum+1])
                    except IndexError:
                        continue

                num_count = num_count_list.count(numerator)

                A[row, column] = ((1 + num_count) * den_count) / (den_count + len(vocabulary))

        sns.heatmap(A, annot=True, xticklabels=[i for i in sentence], yticklabels=[i for i in sentence])
        plt.title(f"Bi-gram of '{input}''")
        plt.show()
        
    return A

A_bigram = bigram(words, 
       "i'd like to eat [uh] good food", 
       smoothing='laplace')

#%%

# Random unigram sampling 

def random_unigram(words_, length):
    
    sentence = []
    for i in range(length):
        sentence.append(Counter(words_).most_common()[i + np.random.randint(123):][0][0])
    
    sentence = " ".join(sentence)
    
    return print(sentence)

random_unigram(words,15)

#%%

# Random bigram sampling 

def random_bigram(words_, length):
    num_count_list = []
    for enum, i in enumerate(words_):
        try:
            num_count_list.append(words_[enum] + " " + words_[enum+1])
        except IndexError:
            continue

    sentence = []
    for i in range(length):
        sentence.append(Counter(num_count_list).most_common()[i+np.random.randint(123):][0][0])

    sentence = " ".join(sentence)

    print(sentence)

random_bigram(words, 15)
#%%

# perplexity unigram

def perplexity_unigram(A):
    perplexity = np.prod(A) ** (-1/len(A))
    return print(perplexity.round(2))

perplexity_unigram(A_unigram)

#%%

# perplexity bigram

def perplexity_bigram(A):
    probabilities = []
    for i in range(len(A_bigram) - 1):
        probabilities.append(A_bigram[i, i+1])

    perplexity = np.prod(probabilities) ** (-1/len(probabilities))
    return print(perplexity.round(2))

perplexity_bigram(A_bigram)
# %%
