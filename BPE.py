#%%
# Creating the file

import itertools

f = open("bpe_attempt.txt", "w+")
for i in range(5):
    f.write("low ")
for i in range(2):
    f.write("lowest ")
for i in range(6):
    f.write("newer ")
for i in range(3):
    f.write("wider ")
for i in range(2):
    f.write("new ")
f.close()

#%%
# Reading file

text = open("bpe_attempt.txt", "r").read()

#%%
# Home-made white space tokenizer

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

#%%

# BPE algorithm

from collections import Counter
import re

# letters 
l = []

for letters in words:
    for letter in letters:
        l.append(letter)

# corpus
corpus = []
for i in words:
    corpus.append(" ".join(i + "_").split())


def bpe(corpus_, k_merges):
    
    # vocabulary
    vocabulary = []
    vocabulary.append("_")
    for i in l:
        if i not in vocabulary:
            vocabulary.append(i)

    for i in range(k_merges):
            
        # token left and right
        tokens = []
        for i in corpus_:
            for j in range(len(i)-1):
                tokens.append((i[:-1][j], i[1:][j]))

        counts = Counter(tokens).most_common()

        t_new = counts[0][0]
        token_left = t_new[0]
        token_right = t_new[1]
        t_new = "".join(token_left+token_right)

        vocabulary.append(t_new)

        # update corpus_ 

        corpus_new = []
        for i in corpus_:
            if (token_left not in i) or (token_right not in i) == True:
                corpus_new.append(i)
            if (token_left in i) and (token_right in i) == True:
                indices = [index for index, item in enumerate(i) if i[index] == token_left and i[index+1] == token_right]
                if indices == []:
                    corpus_new.append(i)
                else:
                    i[indices[0]:indices[0]+2] = [t_new]
                    corpus_new.append(i)

        corpus_ = corpus_new
    return vocabulary

vocabulary = bpe(corpus, k_merges=8)

print(vocabulary)

# %%
