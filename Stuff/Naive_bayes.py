#%%

import pandas as pd
import numpy as np

# train naive bayes

df = pd.read_csv('twitter_train.csv', names=["id", "entity", "classes", "documents"])

df.documents = df.documents.astype(str)

classes_ = df.classes.unique()

def train_naive_bayes(documents, classes_):
    n_doc = len(documents)

    vocabulary = []
    for i in range(len(documents)):
        sentence = documents[i].split()
        for j in sentence:
            if j not in vocabulary:
                vocabulary.append(j)

    likelihood_all = []
    prior_class_all = []
    for i in range(len(classes_)):
        current_class = classes_[i]
        n_doc_class = (df.classes == classes_[i]).sum()

        prior_class = n_doc_class/n_doc

        big_doc_c = documents[df.classes == classes_[i]]
        corpus = []
        for i in range(len(big_doc_c)):
            sentence = big_doc_c.iloc[i].split()
            for j in sentence:
                corpus.append(j)

        corpus_unique = []
        for i in corpus:
            if i not in corpus_unique:
                corpus_unique.append(i)
        
        likelihood = [[current_class, word, (corpus.count(word) + 1)/(len(corpus) + len(vocabulary))] for word in corpus_unique]
        likelihood_all.append(likelihood)
        prior_class_all.append([current_class, prior_class])

    likelihood_all = [item for sublist in likelihood_all for item in sublist]
    likelihood_all = pd.DataFrame(likelihood_all, columns = ["classes", "word", "prob"])
    prior_class_all = pd.DataFrame(prior_class_all, columns= ["classes", "prior"])
    return likelihood_all, prior_class_all, corpus, vocabulary

likelihood_all, prior_class_all, corpus, vocabulary = train_naive_bayes(df.documents, classes_)

#%%

def test_naive_bayes(input, prior_class_all, likelihood_all, corpus, vocabulary):
    sentence = input.split()

    final_prob = []
    for i in classes_:
        prior_ = prior_class_all["prior"][prior_class_all["classes"] == i].values[0]
        prob_sentence = []
        for w in sentence:
            if w in vocabulary:
                try:
                    prob_word = prior_ * likelihood_all["prob"][(likelihood_all["word"] == w) & (likelihood_all["classes"] == i)].values[0]
                    prob_sentence.append(prob_word)
                except IndexError:
                    prob_sentence.append(0)
                    continue
            if w not in vocabulary:
                continue
        prob_sentence = np.prod(prob_sentence)
        final_prob.append([i, prob_sentence])

    final_prob = pd.DataFrame(final_prob, columns= ["classes", "prob"])

    class_prediction = final_prob["classes"][final_prob["prob"] == max(final_prob.prob)].values[0]

    print(class_prediction)
    return final_prob

test_naive_bayes("Make america great again", 
                 prior_class_all, 
                 likelihood_all, 
                 corpus, 
                 vocabulary)

# %%
