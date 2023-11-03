#%%
# Packages
from nltk import word_tokenize, WordNetLemmatizer, FreqDist, sent_tokenize, pos_tag
from urllib import request
from bs4 import BeautifulSoup
import pandas as pd
import re
# ---

# (a)

url = "https://en.wikipedia.org/wiki/Natural_language_processing"
html = request.urlopen(url).read().decode('utf8')

raw = BeautifulSoup(html, 'html.parser').get_text()

#%%

# (b)

tokens = word_tokenize(raw)


# Removing special characters using regular expressions
tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]

#normalizing lower case
tokens = [x.lower() for x in tokens]

#lemmatizer 

wnl = WordNetLemmatizer()
lemma_tokens = [wnl.lemmatize(t) for t in tokens]

#%%

# (c)

freq = FreqDist(lemma_tokens)

most_frequent = FreqDist(lemma_tokens).most_common(30)


#%%

# (d)

from nltk import pos_tag

sent_token = sent_tokenize(raw)

word_counts = [len(word_tokenize(sentence)) for sentence in sent_token]

verb_counts_per_sentence = [
    sum(1 for word, pos in pos_tag(word_tokenize(sentence)) if pos.startswith('V'))
    for sentence in sent_token
]

noun_counts_per_sentence = [
    sum(1 for word, pos in pos_tag(word_tokenize(sentence)) if pos.startswith('N'))
    for sentence in sent_token
]


#%%

#(e)

df = pd.DataFrame()

df["sentences"] = sent_token

df["word_counts"] = word_counts

df["verb_counts"] = verb_counts_per_sentence

df["noun_counts"] = noun_counts_per_sentence


# %%

