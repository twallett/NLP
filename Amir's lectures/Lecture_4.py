#%%

import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'search a specific  Char' + 20 * '-' )
pattern1 = re.compile(r'abc')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
print(text[0:3])
# ----------------------------
print(20 * '-' + 'dot menas any chars except new line' + 20 * '-' )
pattern2 = re.compile(r'.')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Back slash escapes special characters' + 20 * '-' )
pattern3 = re.compile(r'\.')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Search an string' + 20 * '-' )
pattern4 = re.compile(r'amir\.com')
matches = pattern4.finditer(text)
for match in matches:
    print(match)


#%%

import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Digits' + 20 * '-' )
pattern1 = re.compile(r'\d')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Digits' + 20 * '-' )
pattern2 = re.compile(r'\D')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Digits' + 20 * '-' )
pattern3 = re.compile(r'\d\d')
matches = pattern3.finditer(text)
for match in matches:
    print(match)

#%%

import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Word' + 20 * '-' )
pattern1 = re.compile(r'\w')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Word' + 20 * '-' )
pattern2 = re.compile(r'\W')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Word' + 20 * '-' )
pattern3 = re.compile(r'\w\.\w')
matches = pattern3.finditer(text)
for match in matches:
    print(match)

#%%

import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Space' + 20 * '-' )
pattern1 = re.compile(r'\s')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Space' + 20 * '-' )
pattern2 = re.compile(r'\S')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Space' + 20 * '-' )
pattern3 = re.compile(r'\s\.\s')
matches = pattern3.finditer(text)
for match in matches:
    print(match)

#%%

import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Word Boundry' + 20 * '-' )
pattern1 = re.compile(r'\bHa')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Space' + 20 * '-' )
pattern2 = re.compile(r'\BHa')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # ----------------------------
print(20 * '-' + 'Space' + 20 * '-' )
pattern3 = re.compile(r'\b\s\(N')
matches = pattern3.finditer(text)
for match in matches:
    print(match)

#%%

import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Anchor' + 20 * '-' )
pattern1 = re.compile(r'^Start')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Anchor' + 20 * '-' )
pattern2 = re.compile(r'end$')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # ----------------------------
print(20 * '-' + 'Anchor' + 20 * '-' )
pattern3 = re.compile(r'^a')
matches = pattern3.finditer(text)
for match in matches:
    print(match)

#%%

import re
with open('re_text.txt','r') as f:
    text = f.read()
with open('re_fake_names.txt','r') as f:
    text1 = f.read()
# ----------------------------
print(20 * '-' + 'Example - Find Phone number 1' + 20 * '-' )
pattern1 = re.compile(r'\d\d\d')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - Find Phone number 2' + 20 * '-' )
pattern2 = re.compile(r'\d\d\d.\d\d\d.\d\d\d\d')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - Find Phone number 3' + 20 * '-' )
pattern3 = re.compile(r'\d\d\d.\d\d\d.\d\d\d\d')
matches = pattern3.finditer(text1)
for match in matches:
    print(match)

#%%

import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Example - Find Phone number' + 20 * '-' )
pattern1 = re.compile(r'\d\d\d[-.]\d\d\d[-.]\d\d\d\d')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - Find Phone number' + 20 * '-' )
pattern2 = re.compile(r'[89]00[-.]\d\d\d[-.]\d\d\d\d')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # # ----------------------------
print(20 * '-' + 'Example - Find Phone number' + 20 * '-' )
pattern3 = re.compile(r'[1-5]')
matches = pattern3.finditer(text)
for match in matches:
    print(match)

#%%

import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Example - lower' + 20 * '-' )
pattern1 = re.compile(r'[a-z]')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - lower upper' + 20 * '-' )
pattern2 = re.compile(r'[a-zA-Z]')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # # ----------------------------
print(20 * '-' + 'Example - negate the lower and upper' + 20 * '-' )
pattern3 = re.compile(r'[^a-zA-Z]')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
# # # ----------------------------
print(20 * '-' + 'Example - Find words end with at' + 20 * '-' )
pattern1 = re.compile(r'[^b]at')
matches = pattern1.finditer(text)
for match in matches:
    print(match)

#%%

import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Example - Find Phone number' + 20 * '-' )
pattern1 = re.compile(r'\d{3}[-.]\d{3}[-.]\d{4}')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# # ----------------------------
print(20 * '-' + 'Example - Find name' + 20 * '-' )
pattern2 = re.compile(r'Mr\.?\s[A-Z]\w*')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # # # ----------------------------
print(20 * '-' + 'Example - Find name with groups' + 20 * '-' )
pattern3 = re.compile(r'M(r|s|rs)\.?\s[A-Z]\w*')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
# # # # ----------------------------
print(20 * '-' + 'Example - Find name with groups' + 20 * '-')
pattern4 = re.compile(r'(Mr|Ms|Mrs)\.?\s[A-Z]\w*')
matches = pattern4.finditer(text)
for match in matches:
    print(match)

#%%

import re
with open('re_email.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Example - email' + 20 * '-' )
pattern1 = re.compile(r'[a-zA-z]+@[a-zA-z]+\.com')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# # # ----------------------------
print(20 * '-' + 'Example - email' + 20 * '-' )
pattern2 = re.compile(r'[a-zA-z.]+@[a-zA-z]+\.(com|edu)')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # # # # ----------------------------
print(20 * '-' + 'Example - email' + 20 * '-' )
pattern3 = re.compile(r'[a-zA-z0-9.-]+@[a-zA-z-]+\.(com|edu|net)')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
# # # # # ----------------------------
print(20 * '-' + 'Example - email' + 20 * '-')
pattern4 = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
matches = pattern4.finditer(text)
for match in matches:
    print(match)
    
#%%

import re
urls = '''https://www.google.com
          http://amir.com
          https://youtube.com
          https://www.epa.gov
       '''
pattern1 = re.compile(r'https?://(www\.)?\w+\.\w+')
matches = pattern1.finditer(urls)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - url' + 20 * '-' )
pattern2 = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')
matches = pattern2.finditer(urls)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - url' + 20 * '-')
pattern2 = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')
matches = pattern2.finditer(urls)
for match in matches:
    print(match.group(2))

#%%

sentence ='Thomas Jefferson began building Monticello at theage of 26.'
print(sentence.split())
print(str.split(sentence))

#%%

import numpy as np
import pandas as pd
sentence ='Thomas Jefferson began building Monticello at the age of 26.'
token_sequence = str.split(sentence)
vocab = sorted(set(token_sequence))
print(', '.join(vocab))
num_tokens = len(token_sequence)
vocab_size = len(vocab)
onehot_vectors = np.zeros((num_tokens,vocab_size), int)
for i, word in enumerate(token_sequence):
    onehot_vectors[i, vocab.index(word)] = 1
' '.join(vocab)
print(onehot_vectors)
df = pd.DataFrame(onehot_vectors, columns=vocab)
print(df)

#%%

num_rows = 3000 * 3500 * 15
print('{} Number of Rows'.format(num_rows))
num_bytes = num_rows * 1000000
print('{} Bytes'.format(num_bytes))
Size = num_bytes / 1e9
print('{} Terabytes'.format(Size/1000))

#%%

import re
sentence ='Thomas Jefferson began building Monticello at theage of 26.'
tokens = re.split(r'[-\s.,;!?]+', sentence)
print(tokens)

#%%

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
sentence ='Thomas Jefferson began building Monticello at theage of 26.'
print(tokenizer.tokenize(sentence))

from nltk.tokenize import TreebankWordTokenizer
sentence = "Monticello wasn't designated as UNESCO World Heritage Site until 1987."
tokenizer = TreebankWordTokenizer()
print(tokenizer.tokenize(sentence))

#%%

from nltk.tokenize.casual import casual_tokenize
message = "RT @TJMonticello Best day everrrrrrr at Monticello." \
          "Awesommmmmmeeeeeeee day :*)"
print(casual_tokenize(message))
print(casual_tokenize(message, reduce_len=True, strip_handles=True))

#%%

import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)
print(len(stop_words))

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
print(len(sklearn_stop_words))
print(len(set(stop_words).union(sklearn_stop_words)))
print(len(set(stop_words).intersection(sklearn_stop_words)))

#%%

import re
def stem(phrase):
    return ' '.join([re.findall('^(.*ss|.*?)(s)?$', word)
                     [0][0].strip("'") for word in phrase.lower().split()])
print(stem("Doctor House's calls"))
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
print(' '.join([stemmer.stem(w).strip("'")
                for w in "dish washer's washed dishes".split()]))

#%%

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("better", pos="n"))

#%%

from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
sentence = "The faster Harry got to the store, the faster Harry " \
           "the faster, would get home."
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
print(tokens)

bag_of_words = Counter(tokens)
print(bag_of_words)

#%%

from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
sentence = "The faster Harry got to the store, the faster Harry " \
           "the faster, would get home."
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
bag_of_words = Counter(tokens)
print(bag_of_words.most_common(4))

times_harry_appears = bag_of_words['harry']
num_unique_words = len(bag_of_words)
tf = times_harry_appears / num_unique_words; print(tf)

#%%

import nltk
from collections import Counter
from nltk.tokenize import TreebankWordTokenizer

with open('kite.txt','r') as f:
    kite_text = f.read()

tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(kite_text.lower())
token_counts = Counter(tokens)
print(token_counts)
print(20 *'-')
nltk.download('stopwords', quiet=True)
stopwords = nltk.corpus.stopwords.words('english')
tokens = [x for x in tokens if x not in stopwords]
kite_counts = Counter(tokens)
print(kite_counts)

#%%

from nltk.tokenize import TreebankWordTokenizer
from collections import OrderedDict, Counter
import copy
import pandas as pd
tokenizer = TreebankWordTokenizer()
docs = ["The faster Harry got to the store, the faster and faster "
        "Harry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")
doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
print(len(doc_tokens[0]))
all_doc_tokens = sum(doc_tokens, []) ; print(len(all_doc_tokens))
lexicon = sorted(set(all_doc_tokens)); print(len(lexicon))
zero_vector = OrderedDict((token, 0) for token in lexicon); print(zero_vector)
doc_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value / len(lexicon)
    doc_vectors.append(vec)
df = pd.DataFrame(doc_vectors);print(df)

#%%

import numpy as np
from numpy.linalg import norm
def cosine_sim(a,b):
    return  np.dot(a, b)/(norm(a)*norm(b))
print(cosine_sim([1,1], [1,1]))
print(cosine_sim([1,1], [-1,1]))
print(cosine_sim([1,1], [-1,-1]))

#%%

from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
tokenizer = TreebankWordTokenizer()
with open('kite.txt' , 'r') as f:
    kite_text = f.read()
with open('kite_history.txt' , 'r', encoding='utf-8') as f:
    kite_history = f.read()

kite_intro = kite_text.lower()
intro_tokens = tokenizer.tokenize(kite_intro)

kite_history = kite_history.lower()
history_tokens = tokenizer.tokenize(kite_history)

intro_total = len(intro_tokens) ; print(intro_total)
history_total = len(history_tokens) ; print(history_total)

intro_tf = {};history_tf = {}

intro_counts = Counter(intro_tokens)
history_counts = Counter(history_tokens)
intro_tf['kite'] = intro_counts['kite'] / intro_total
print('Term Frequency of "kite" in intro is: {:.4f}'.format(intro_tf['kite']))

history_tf['kite'] = history_counts['kite'] / history_total
print('Term Frequency of "kite" in history is: {:.4f}'.format(history_tf['kite']))

intro_tf['and'] = intro_counts['and'] / intro_total
history_tf['and'] = history_counts['and'] / history_total
print('Term Frequency of "and" in history is: {:.4f}'.format(history_tf['and']))

#%%

from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

tokenizer = TreebankWordTokenizer()

with open('kite.txt' , 'r') as f:
    kite_text = f.read()
with open('kite_history.txt' , 'r', encoding='utf-8') as f:
    kite_history = f.read()

kite_intro = kite_text.lower();intro_tokens = tokenizer.tokenize(kite_intro)
kite_history = kite_history.lower();history_tokens = tokenizer.tokenize(kite_history)
num_docs_containing_and ,num_docs_containing_kite = 0, 0
for doc in [intro_tokens, history_tokens]:
    if 'and' in doc:
        num_docs_containing_and += 1
    if 'kite' in doc:
        num_docs_containing_kite += 1
intro_total = len(intro_tokens) ;history_total = len(history_tokens)
intro_counts = Counter(intro_tokens);history_counts = Counter(history_tokens)
intro_tf = {};history_tf = {};intro_tfidf = {};history_tfidf = {}
intro_tf['and'] = intro_counts['and'] / intro_total
history_tf['and'] = history_counts['and'] / history_total
intro_tf['kite'] = intro_counts['kite'] / intro_total
history_tf['kite'] = history_counts['kite'] / history_total
num_docs = 2;intro_idf = {};history_idf = {};num_docs = 2
intro_idf['and'] = num_docs / num_docs_containing_and
history_idf['and'] = num_docs / num_docs_containing_and
intro_idf['kite'] = num_docs / num_docs_containing_kite
history_idf['kite'] = num_docs / num_docs_containing_kite
intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite'] * intro_idf['kite']
history_tfidf['and'] = history_tf['and'] * history_idf['and']
history_tfidf['kite'] = history_tf['kite'] * history_idf['kite']

#%%

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.shape)

#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

rng = np.random.RandomState(0)
n_samples = 500
cov = [[3, 3],
       [3, 4]]
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
pca = PCA(n_components=2).fit(X)


plt.scatter(X[:, 0], X[:, 1], alpha=.3, label='samples')
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    plt.plot([0, comp[0]], [0, comp[1]], label=f"Component {i}", linewidth=5,
             color=f"C{i + 2}")
plt.gca().set(aspect='equal',
              title="2-dimensional dataset with principal components",
              xlabel='first feature', ylabel='second feature')
plt.legend()
plt.show()

#%%

doc1 = "Data Science Machine Learning"
doc2 = "Money fun Family Kids home"
doc3 = "Programming Java Data Structures"
doc4 = "Love food health games energy fun"
doc5 = "Algorithms Data Computers"

doc_complete = [doc1, doc2, doc3, doc4, doc5]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X =vectorizer.fit_transform(doc_complete)

from sklearn.decomposition import TruncatedSVD
lsa = TruncatedSVD(n_components=2,n_iter=100)
lsa.fit(X)
terms = vectorizer.get_feature_names()

for i,comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:10]
    print("Concept %d:" % i)
    for term in sortedterms:
        print(term[0])
    print(" ")

#%%

