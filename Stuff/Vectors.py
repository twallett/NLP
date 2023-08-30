#%%

import pandas as pd

df = pd.read_csv('twitter_train.csv', names = ['id', 'entity', 'classes', 'document'])

#%%

# Term-document 

entities = list(df.entity.unique())
documents = df.document.astype(str)

#%%

vocabulary = []
for tweet in documents:
    tweet = tweet.split()
    for word in tweet:
        if word not in vocabulary:
            vocabulary.append(word)

#%%

term_document = []
for entity in entities:
    docuemnt = df.document[df.entity == entity].astype(str)
    for tweet in docuemnt:
        tweet = tweet.split()
        for word in tweet:
            if word in vocabulary:
                term_document.append([entity, word])
                
#%%
term_document = pd.DataFrame(term_document, columns=["entity_", "words_"])

#%%

term_document_table = term_document.pivot_table(index='words_', columns='entity_', aggfunc='size', fill_value=0)

#%%

# Amazon - Verizon - CallOfDuty - LeagueOfLegends

example_term_document = term_document_table[(term_document_table.index == 'PS5') | (term_document_table.index == 'buy') | (term_document_table.index == 'game')| (term_document_table.index == 'kill')]

#%%

import matplotlib.pyplot as plt 

plt.scatter(example_term_document.loc["PS5",:], example_term_document.loc["buy",:])

#%%

# Term-term

for i in docuemnt:
    i = i.split()
    print(i)
    for enum, j in enumerate(i):
        print(i[enum])
    
#%%
