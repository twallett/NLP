#%%

import spacy

# In part of this exercise, you will use Spacy software to explore the unrealistic news dataset called data.txt.

# i. Use pandas to load the data.csv data.

import pandas as pd 

df = pd.read_csv("data.csv")

#%%

# ii. Use Spacy to find the word level attributions ( Tokenized word, StartIndex, Lemma, punctuation,white space ,WordShape, PartOfSpeech, POSTag). Use one of the titles in the dataframe and create a dataframe which rows are the word and the columns are attributions.

nlp = spacy.load("en_core_web_md")

title = df.title[0]

nlp = nlp(title)

adict = {"Names": [x for x in nlp],
         "Tokenized words": [x.text for x in nlp], # tokenized words
         "Index": [x.i for x in nlp], # start index
         "Lemma": [x.lemma_ for x in nlp], # lemma
         "Punctuations": [x.is_punct for x in nlp], # punctuations 
         "White_spaces": [x.is_space for x in nlp], # white spaces
         "Word_shape": [x.shape_ for x in nlp], # word shape
         "Pos": [x.pos_ for x in nlp], # pos
         "Pos_tag": [x.tag_ for x in nlp] # pos tag
         }

ii_df = pd.DataFrame(adict)

# %%

# iii. Use spacy and find entities on the text in part ii.

ents = [(x.text, x.label_) for x in nlp.ents]

# %%

# iv. Grab a different title and use spacy to chunk the noun phrases, label them and finally find the roots of each chunk.

nlp = spacy.load("en_core_web_md")

different = df.title[1]

nlp = nlp(different)

for x in nlp.noun_chunks:
    print(x, x.label_, x.root.text)

# %%

# v. Use SPacy to analyzes the grammatical structure of a sentence, establishing relationships between ”head” words and words which modify those heads. Hint: Insatiate the nlp doc and then look for text, dependency, head text, head pos and children of it.

nlp = spacy.load("en_core_web_md")

doc = nlp(title)

for word in doc:
    print(word.text, word.dep_, word.head.text, word.head.pos_)
    children = [child for child in word.children]
    if children:
        print("Children (Modifiers):")
        for child in children:
            print(f"  {child.text} ({child.dep_})")

# %%

# vi. Use spacy to find word similarly measure. Spacy has word vector model as well. So we can use the same to find similar words. Use spacy large model to get a decent results.

nlp = spacy.load("en_core_web_lg")

doc = nlp(title)

doc2 = nlp(different)

doc.similarity(doc2)
# %%


# In part of this exercise, you will use Spacy software to explore the tweets dataset called data1.txt.

# i. Use pandas to load the data1.csv data.

df = pd.read_csv("data1.csv")

# %%

# ii. Let’s look at some examples of real world sentences. Grab a tweet and explain the text entities.

import spacy

text = df.text[0]

nlp = spacy.load("en_core_web_lg")

doc = nlp(text)

for word in doc.ents:
    print(word.text, word.label_)
    print(spacy.explain(word.label_))

# %%

# iii. One simple use case for NER is redact names. This is important and quite useful. Find a tweet which has a name in it and then redact it by word [REDACTED].

text = df.text[1]

doc = nlp(text)

redacted_text = []
for token in doc:
    if token.ent_type_ == "PERSON":
        redacted_text.append("[REDACTED]")
    else:
        redacted_text.append(token.text)

# Reconstruct the redacted text
redacted_tweet = " ".join(redacted_text)

# Print the redacted tweet
print(redacted_tweet)

# %%


# Use spacy to answer all the following questions.

# i. Apply part of speech Tags methods in spacy on a sentence.


#%%

# =================================================================
# Class_Ex1:
# Import spacy abd from the language class import english.
# Create a doc object
# Process a text : This is a simple example to initiate spacy
# Print out the document text from the doc object.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

from spacy.lang.en import English

nlp = English()

doc = nlp("This is a simple example to initiate spacy")

print(doc.text)


print(20 * '-' + 'End Q1' + 20 * '-')
#%%
# =================================================================
# Class_Ex2:
# Solve Ex1 but this time use German Language.
# Grab a sentence from german text from any website.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

from spacy.lang.de import German

nlp = German()

doc = nlp("Das ist ein schöner Tag.")

print(doc.text)

print(20 * '-' + 'End Q2' + 20 * '-')
#%%
# =================================================================
# Class_Ex3:
# Tokenize a sentence using sapaCy.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

import spacy

sentence = "This is a simple example to initiate spacy"

nlp = spacy.load("en_core_web_lg")

doc = nlp(sentence)

for word in doc:
    print(word.text)


print(20 * '-' + 'End Q3' + 20 * '-')
#%%
# =================================================================
# Class_Ex4:
# Use the following sentence as a sample text. and Answer the following questions.
# "In 2020, more than 15% of people in World got sick from a pandemic ( www.google.com ). Now it is less than 1% are. Reference ( www.yahoo.com )"
# 1- Check if there is a token resemble a number.
# 2- Find a percentage in the text.
# 3- How many url is in the text.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

import spacy

sentence = "In 2020, more than 15% of people in World got sick from a pandemic ( www.google.com ). Now it is less than 1% are. Reference ( www.yahoo.com )"

nlp = spacy.load("en_core_web_lg")

# 1- Check if there is a token resemble a number.

doc = nlp(sentence)

# for word in doc:
#     print(word.is_digit)
    
# 2- Find a percentage in the text.

from spacy.matcher import Matcher

pattern = [{"TEXT":"%"}]

matcher = Matcher(nlp.vocab)
matcher.add("Percentage Pattern", [pattern])

matches = matcher(doc)

# for match_id, start, end in matches:
#     matched_span = doc[start:end]
#     print(matched_span.text)

# 3- How many url is in the text.

pattern2 = [{"TEXT": {"REGEX": r"(https?|www)\S+"}}]

matcher2 = Matcher(nlp.vocab)
matcher2.add("Percentage Pattern", [pattern2])

matches2 = matcher2(doc)
print(len(matches2))


print(20 * '-' + 'End Q4' + 20 * '-')
#%%
# =================================================================
# Class_Ex5:
# Load small web english model into spaCy.
# USe the following text as a sample text. Answer the following questions
# "It is shown that: Google was not the first search engine in U.S. tec company. The value of google is 100 billion dollar"
# 1- Get the token text, part-of-speech tag and dependency label.
# 2- Print them in a tabular format.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')




print(20 * '-' + 'End Q5' + 20 * '-')
#%%
# =================================================================
# Class_Ex6:
# Use Ex 5 sample text and find all the entities in the text.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

print(20 * '-' + 'End Q6' + 20 * '-')
# =================================================================
# Class_Ex7:
# Use SpaCy and find adjectives plus one or 2 nouns.
# Use the following Sample text.
# Features of the iphone applications include a beautiful design, smart search, automatic labels and optional voice responses.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

import spacy

sentence = "Features of the iphone applications include a beautiful design, smart search, automatic labels and optional voice responses."

nlp = spacy.load("en_core_web_lg")

doc = nlp(sentence)

for word in doc:
    if  word.tag_ == "JJ" or word.tag_ == "NN":
        print(word.text, word.tag_)

print(20 * '-' + 'End Q7' + 20 * '-')
# =================================================================
# Class_Ex8:
# Use spacy lookup table and find the hash id for a cat
# Text : I have a cat.
# Next use the id and find the strings.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')

sentence = "I have a cat."



print(20 * '-' + 'End Q8' + 20 * '-')
# =================================================================
# Class_Ex9:
# Create a Doc object for the following sentence
# Spacy is a nice toolkit.
# Use the methods like text, token,... on the Doc and check the functionality.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')

print(20 * '-' + 'End Q9' + 20 * '-')
# =================================================================
# Class_Ex10:
# Use spacy and process the following text.
# Newyork looks like a nice city.
# Find which token is proper noun and which one is a verb.
#

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

print(20 * '-' + 'End Q10' + 20 * '-')
# =================================================================
# Class_Ex11:
# Read the list of countries in a json format.
# Use the following text as  sample text.
# Czech Republic may help Slovakia protect its airspace
# Use statistical method and rule based method to find the countries.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')

print(20 * '-' + 'End Q11' + 20 * '-')
# =================================================================
# Class_Ex12:
# Use spacy attributions and answer the following questions.
# Define the getter function that takes a token and returns its reversed text.
# Add the Token property extension "reversed" with the getter function
# Process the text and print the results.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')

print(20 * '-' + 'End Q12' + 20 * '-')
# =================================================================
# Class_Ex13:
# Read the tweets json file.
# Process the texts and print the entities
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q13' + 20 * '-')

print(20 * '-' + 'End Q13' + 20 * '-')
# =================================================================
# Class_Ex14:
# Use just spacy tokenization. for the following text
# "Burger King is an American fast food restaurant chain"
# make sure other pipes are disabled and not used.
# Disable parser and tagger and process the text. Print the tokens
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q14' + 20 * '-')

import spacy

# Load the SpaCy model with only the tokenizer (disable parser and tagger)
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])

# Text to tokenize
text = "Burger King is an American fast food restaurant chain"

# Process the text using only the tokenizer
doc = nlp(text)

# Print the tokens
for token in doc:
    print(token.text)


print(20 * '-' + 'End Q14' + 20 * '-')

# =================================================================
# %%
