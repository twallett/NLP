#%%

# my hommie moby

# i. Analyzing Moby Dick text. Load the moby.txt file into python environment. (Load the raw data or Use the NLTK Text object)

with open("moby.txt", mode = "r") as f:
    moby = f.read()

print(f"moby.txt file: {moby}")

#%%

# ii. Tokenize the text into words. How many tokens (words and punctuation symbols) are in it?

from nltk import word_tokenize

moby_tokens = word_tokenize(moby)

print(f"How many tokens (words and punctuation symbols) are in it: {len(moby_tokens)}")

#%%
# iii. How many unique tokens (unique words and punctuation) does the text have?

moby_unique = set([x.lower() for x in moby_tokens])

print(f"How many unique tokens (unique words and punctuation) does the text have?: {len(moby_unique)}")

#%%
# iv. After lemmatizing the verbs, how many unique tokens does it have?

from nltk import pos_tag, WordNetLemmatizer

moby_pos = pos_tag(moby_tokens)

moby_verbs = [moby_pos[i][0].lower() for i in range(len(moby_pos)) if moby_pos[i][1].startswith("VB")]

lem = WordNetLemmatizer()
moby_unique_lemma = set([lem.lemmatize(x) for x in moby_verbs])

print(f"After lemmatizing the verbs, how many unique tokens does it have?: {len(moby_unique_lemma)}")

#%%
# v. What is the lexical diversity of the given text input?

print(f" What is the lexical diversity of the given text input?: {(len(set(moby_tokens))/len(moby_tokens) * 100).__round__(2)}%")

#%%
# vi. What percentage of tokens is ’whale’or ’Whale’?

from nltk import FreqDist

moby_freq = FreqDist(moby_tokens)

whale = moby_freq["whale"]
Whale = moby_freq["Whale"]

print(f" What percentage of tokens is ’whale’or ’Whale’? {whale} whale {Whale} Whale")

#%%
# vii. What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?

moby_most_common = FreqDist([x.lower() for x in moby_tokens])
moby_most_common_20 = moby_most_common.most_common(20)

print(f" What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?: {moby_most_common_20}")

#%%
# viii. What tokens have a length of greater than 6 and frequency of more than 160?

moby_words = [key for key, value in moby_most_common.items() if len(key) == 6 and value > 160]

#%%
# ix. Find the longest word in the text and that word’s length.

moby_longest_word = {len(x):x for x in moby_tokens}
moby_longest_index = sorted(moby_longest_word)[-1]

moby_longest_word = moby_longest_word.get(moby_longest_index)

#%%
# x. What unique words have a frequency of more than 2000? What is their frequency?

moby_ge_2000 = {key:value for key, value in moby_most_common.items() if value > 2000}

#%%
# xi. What is the average number of tokens per sentence?

import nltk

moby_sentences = nltk.corpus.gutenberg.sents('melville-moby_dick.txt')

moby_sentences_dict = {str(moby_sentences[i]):len(moby_sentences[i]) for i in range(len(moby_sentences))}

moby_sentences_length = [val for key, val in moby_sentences_dict.items()]

length = sum(moby_sentences_length)/len(moby_sentences_length)

#%%
# xii. What are the 5 most frequent parts of speech in this text? What is their frequency?

pos = [moby_pos[i][1] for i in range(len(moby_pos))]

pos = FreqDist(pos).most_common(5)

#%%

# my hommie ben 

# i. Write a function that scrape the web page and return the raw text file.

# ii. Use BeautifulSoup to get text file and clean the html file.

from urllib import request
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Benjamin_Franklin"
html = request.urlopen(url).read().decode("utf8")
ben = BeautifulSoup(html, "html.parser").get_text()

#%%
# iii. Write a function called unknown, which removes any items from this set that occur in the Words Corpus (nltk.corpus.words).

# iv. Find a list of novel words.

import nltk

def unknown(text):
    text = set([x.lower() for x in text])
    words = set([x.lower() for x in nltk.corpus.words.words()])
    return text - words

novel = unknown(ben)

#%%
# v. Use the porter stemmer to stem all the items in novel words the go through the unknown function, saving the result as novel-stems.

from nltk import PorterStemmer

stem = PorterStemmer()
novel_stem = [stem.stem(x) for x in novel]

#%%
# vi. Find as many proper names from novel-stems as possible, saving the result as propernames.













#%% 

# my hommie Dostoevsky

import nltk
from nltk import FreqDist

with open("crime_and_punishment.txt", "r") as f:
    dostoevsky = f.read()
    
dostoevsky_tokens = nltk.word_tokenize(dostoevsky)

stop_words = nltk.corpus.stopwords.words("english")

dostoevsky_clean = [x for x in dostoevsky_tokens if x.isalpha() and x not in stop_words]

frequency = FreqDist(dostoevsky_clean)

dostoevsky_clean = nltk.text.Text(dostoevsky_clean)

print(dostoevsky_clean.concordance("axe"))

#%%
print(dostoevsky_clean.similar("Rodion"))

dostoevsky_clean.plot(30)

# %%
dostoevsky_clean.collocations()
# %%

dostoevsky_clean.dispersion_plot(["Raskolnikov", "Razumihin", "Dounia", "Sonia", "axe"])
# %%


#%%

# =================================================================
# Class_Ex1:
# Use NLTK Book fnd which the related Sense and Sensibility.
# Produce a dispersion plot of the four main protagonists in Sense and Sensibility:
# Elinor, Marianne, Edward, and Willoughby. What can you observe about the different
# roles played by the males and females in this novel? Can you identify the couples?
# Explain the result of plot in a couple of sentences.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

from nltk.book import text2

text2.dispersion_plot(["Elinor", "Marianne", "Edward", "Willoughby"])


print(20 * '-' + 'End Q1' + 20 * '-')

#%%
# =================================================================
# Class_Ex2:
# What is the difference between the following two lines of code? Explain in details why?
# Make up and example base don your explanation.
# Which one will give a larger value? Will this be the case for other texts?
# 1- sorted(set(w.lower() for w in text1))
# 2- sorted(w.lower() for w in set(text1))
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

from nltk.book import text1

n1 = sorted(set(w.lower() for w in text1)) # normalized first then unique
n2 = sorted(w.lower() for w in set(text1)) # unique first then normalized  

print(f"len n1 {len(n1)}, len n2 {len(n2)}")

print(20 * '-' + 'End Q2' + 20 * '-')

#%%
# =================================================================
# Class_Ex3:
# Find all the four-letter words in the Chat Corpus (text5).
# With the help of a frequency distribution (FreqDist), show these words in decreasing order of frequency.
#
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

from nltk.book import text5
from nltk import FreqDist, word_tokenize

x  = FreqDist(text5).most_common()

for i in x:
    print(i)


print(20 * '-' + 'End Q3' + 20 * '-')

#%%
# =================================================================
# Class_Ex4:
# Write expressions for finding all words in text6 that meet the conditions listed below.
# The result should be in the form of a list of words: ['word1', 'word2', ...].
# a. Ending in ise
# b. Containing the letter z
# c. Containing the sequence of letters pt
# d. Having all lowercase letters except for an initial capital (i.e., titlecase)
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

from nltk.book import text6

end = [x for x in text6 if x.endswith("ise")]

# print(end)

contain_z = [x for x in text6 if "z" in x]

# print(contain_z)

contain_pt = [x for x in text6 if "pt" in x]

# print(contain_pt)

having = [x for x in text6 if x.isalpha() and (x.capitalize() == x or x.lower() == x)]

print(20 * '-' + 'End Q4' + 20 * '-')

#%%
# =================================================================
# Class_Ex5:
#  Read in the texts of the State of the Union addresses, using the state_union corpus reader.
#  Count occurrences of men, women, and people in each document.
#  What has happened to the usage of these words over time?
# Since there would be a lot of document use every couple of years.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

from nltk.corpus import state_union
from nltk import FreqDist, ConditionalFreqDist

# for fieldid in state_union.fileids():
#     freq = FreqDist(state_union.words(fieldid))
#     men = freq["men"]
#     women = freq["women"]
#     people = freq["people"]
#     print(men, women, people)
    
cfd = ConditionalFreqDist(
    (target, fileid) 
    for fileid in state_union.fileids()
    for word in state_union.words(fileid)
    for target in ["men", "women", "people"]
    if word.lower().startswith(target)
)
cfd.plot()


print(20 * '-' + 'End Q5' + 20 * '-')
#%%
# =================================================================
# Class_Ex6:
# The CMU Pronouncing Dictionary contains multiple pronunciations for certain words.
# How many distinct words does it contain? What fraction of words in this dictionary have more than one possible pronunciation?
#
#
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

from nltk.corpus import cmudict

pronouncing_dict = cmudict.dict()

# Count the total number of distinct words in the CMU Pronouncing Dictionary
distinct_words = len(pronouncing_dict)

# Count the number of words with more than one pronunciation
words_with_multiple_pronunciations = sum(len(pronunciations) > 1 for pronunciations in pronouncing_dict.values())

# Calculate the fraction of words with multiple pronunciations
fraction_with_multiple_pronunciations = words_with_multiple_pronunciations / distinct_words

print(20 * '-' + 'End Q6' + 20 * '-')
#%%
# =================================================================
# Class_Ex7:
# What percentage of noun synsets have no hyponyms?
# You can get all noun synsets using wn.all_synsets('n')
#
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

from nltk.corpus import wordnet as wn

# Get all noun synsets
noun_synsets = wn.all_synsets('n')

# Initialize counters
total_noun_synsets = 0
noun_synsets_with_no_hyponyms = 0

# Iterate through noun synsets and count those with no hyponyms
for synset in noun_synsets:
    total_noun_synsets += 1
    if not synset.hyponyms():
        noun_synsets_with_no_hyponyms += 1

# Calculate the percentage
percentage_no_hyponyms = (noun_synsets_with_no_hyponyms / total_noun_synsets) * 100

print(f"Percentage of noun synsets with no hyponyms: {percentage_no_hyponyms:.2f}%")

print(20 * '-' + 'End Q7' + 20 * '-')

#%%
# =================================================================
# Class_Ex8:
# Write a program to find all words that occur at least three times in the Brown Corpus.
# USe at least 2 different method.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')




print(20 * '-' + 'End Q8' + 20 * '-')
#%%
# =================================================================
# Class_Ex9:
# Write a function that finds the 50 most frequently occurring words of a text that are not stopwords.
# Test it on Brown corpus (humor), Gutenberg (whitman-leaves.txt).
# Did you find any strange word in the list? If yes investigate the cause?
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')



print(20 * '-' + 'End Q9' + 20 * '-')
#%%
# =================================================================
# Class_Ex10:
# Write a program to create a table of word frequencies by genre, like the one given in 1 for modals.
# Choose your own words and try to find words whose presence (or absence) is typical of a genre. Discuss your findings.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

from nltk import ConditionalFreqDist
from nltk.corpus import brown

cfd = ConditionalFreqDist((cat, words) 
                           for cat in brown.categories()
                           for words in brown.words(categories = cat))

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
print(cfd.tabulate(conditions=genres, samples=modals))

print(20 * '-' + 'End Q10' + 20 * '-')
#%%
# =================================================================
# Class_Ex11:
#  Write a utility function that takes a URL as its argument, and returns the contents of the URL,
#  with all HTML markup removed. Use from urllib import request and
#  then request.urlopen('http://nltk.org/').read().decode('utf8') to access the contents of the URL.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')





print(20 * '-' + 'End Q11' + 20 * '-')
#%%
# =================================================================
# Class_Ex12:
# Read in some text from a corpus, tokenize it, and print the list of all
# wh-word types that occur. (wh-words in English are used in questions,
# relative clauses and exclamations: who, which, what, and so on.)
# Print them in order. Are any words duplicated in this list,
# because of the presence of case distinctions or punctuation?
# Note Use: Gutenberg('bryant-stories.txt')
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')




print(20 * '-' + 'End Q12' + 20 * '-')
# =================================================================
# Class_Ex13:
# Write code to access a  webpage and extract some text from it.
# For example, access a weather site and extract  a feels like temprature..
# Note use the following site https://darksky.net/forecast/40.7127,-74.0059/us12/en
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q13' + 20 * '-')





print(20 * '-' + 'End Q13' + 20 * '-')
#%%
# =================================================================
# Class_Ex14:
# Use the brown tagged sentences corpus news.
# make a test and train sentences and then  use bi-gram tagger to train it.
# Then evaluate the trained model.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q14' + 20 * '-')

import nltk
from nltk.corpus import brown
from nltk.util import bigrams
from nltk import DefaultTagger, UnigramTagger, BigramTagger
from nltk.metrics import accuracy

# Load the "news" category from the Brown corpus
tagged_sents = brown.tagged_sents(categories='news')

# Split the data into train and test sets (80% train, 20% test)
train_size = int(0.8 * len(tagged_sents))
train_sents = tagged_sents[:train_size]
test_sents = tagged_sents[train_size:]

# Create a default tagger (assigns the most frequent tag)
default_tagger = DefaultTagger('NN')

# Create a unigram tagger and train it on the training data
unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)

# Create a bigram tagger and train it on the training data
bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)

# Evaluate the bigram tagger on the test data
accuracy_score = bigram_tagger.evaluate(test_sents)

print(f"Bigram Tagger Accuracy: {accuracy_score:.4f}")


print(20 * '-' + 'End Q14' + 20 * '-')
#%%
# =================================================================
# Class_Ex15:
# Use sorted() and set() to get a sorted list of tags used in the Brown corpus, removing duplicates.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q15' + 20 * '-')

print(20 * '-' + 'End Q15' + 20 * '-')

# =================================================================
# Class_Ex16:
# Write programs to process the Brown Corpus and find answers to the following questions:
# 1- Which nouns are more common in their plural form, rather than their singular form? (Only consider regular plurals, formed with the -s suffix.)
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q16' + 20 * '-')

print(20 * '-' + 'End Q16' + 20 * '-')
# %%
