#%%

# i. Load Email.txt dataset

with open("Email.txt", "r") as f:
    text = f.read()

# %%

# ii. Find all email addresses in the text file.

import re 

pattern = r'[A-Za-z0-9_.+-]+@[A-Za-z0-9_.+-]+\.[A-Za-z0-9_.+-]+'

pattern1 = re.compile(pattern)
matches = pattern1.finditer(text)

alist = []
for match in matches:
    print(match)
    alist.append(match)

# %%

# iii. Verify the results.

import pandas as pd

df = pd.read_csv("Email.txt", sep="\t")

print(len(df))
print(len(alist))

# %%

# i. Load war and peace by By Leo Tolstoy.

with open("war_and_peace.txt", "r") as f:
    text = f.read()
    
#%%

# ii. Check line by line and find any proper name ending with ”..ski” then print them all.

import re 

pattern = r'[A-Za-z]+ski'

pattern1 = re.compile(pattern)

matches = pattern1.finditer(text)

alist = []
for match in matches:
    print(match)
    alist.append(match[0])

# %%

# iii. Put all the names into a dictionary and sort them.

from collections import Counter, OrderedDict

adict = Counter(alist)

adict = dict(sorted(adict.items(), key=lambda item: item[1], reverse=True))

print(adict)

# %%

# i. Write a program with regular expression that joins numbers if there is a space between
# them (e.g., ”12 0 mph is a very high speed in the 6 6 interstate.” to ”120 mph is a very
# high speed in the 66 interstate.” )

import re 

astr = "12 0 mph is a very high speed in the 6 6 interstate."

pattern = r'[0-9]+\s[0-9]+'

pattern1 = re.compile(pattern)

matches = pattern1.finditer(astr)

alist = []
for match in matches:
    alist.append(match[0])

newlist = []
for item in alist:
    newlist.append("".join(item.split()))
    
print(newlist)
    
# %%

# ii. Write a program with regular expression that find the content in the parenthesise and
# replace it with ”(xxxxx)”

import re

astr = "Something (inside) a (parenthesis)"

# Regular expression pattern to find content inside parentheses
pattern = r'\((.*?)\)'

# Function to replace content in parentheses with "(xxxxx)"
def replace_with_x(match):
    return f"({'x' * len(match.group(1))})"

# Use re.sub() to replace the content
result_string = re.sub(pattern, replace_with_x, astr)

print(result_string)

# %%
# =================================================================
# Class_Ex1:
# Write a function that checks a string contains only a certain set of characters
# (all chars lower and upper case with all digits).
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Write a function that matches a string in which a followed by zero or more b's.
# Sample String 'ac', 'abc', abbc'
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Write Python script to find numbers between 1 and 3 in a given string.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

print(20 * '-' + 'End Q3' + 20 * '-')
# =================================================================
# Class_Ex4:
# Write a Python script to find the a position of the substrings within a string.
# text = 'Python exercises, JAVA exercises, C exercises'
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Write a Python script to find if two strings from a list starting with letter 'C'.
# words = ["Cython CHP", "Java JavaScript", "PERL S+"]
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

print(20 * '-' + 'End Q5' + 20 * '-')

# =================================================================
# Class_Ex6:
# Write a Python script to remove everything except chars and digits from a string.
# USe sub method
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

print(20 * '-' + 'End Q6' + 20 * '-')
# =================================================================
# Class_Ex7:
# Scrape the following website
# https://en.wikipedia.org/wiki/Natural_language_processing
# Find the tag which related to the text. Extract all the textual data.
# Tokenize the cleaned text file.
# print the len of the corpus and pint couple of the sentences.
# Calculate the words frequencies.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

print(20 * '-' + 'End Q7' + 20 * '-')
# =================================================================
# Class_Ex8:
# Grab any text from Wikipedia and create a string of 3 sentences.
# Use that string and calculate the ngram of 1 from nltk package.
# Use BOW method and compare the most 3 common words.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')

print(20 * '-' + 'End Q8' + 20 * '-')
# =================================================================
# Class_Ex9:
# Write a python script that accepts any string and do the following.
# 1- Tokenize the text
# 2- Doe word extraction and clean a text. Use regular expression to clean a text.
# 3- Generate BOW
# 4- Vectorized all the tokens.
# 5- The only package you can use is numpy and re.
# all sentences = ["sentence1", "sentence2", "sentence3",...]
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')

print(20 * '-' + 'End Q9' + 20 * '-')
# =================================================================
# Class_Ex10:
# Grab any text (almost a paragraph) from Wikipedia and call it text
# Preprocessing the text data (Normalize, remove special char, ...)
# Find total number of unique words
# Create an index for each word.
# Count number of the words.
# Define a function to calculate Term Frequency
# Define a function calculate Inverse Document Frequency
# Combining the TF-IDF functions
# Apply the TF-IDF Model to our text
# you are allowed to use just numpy and nltk tokenizer
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

print(20 * '-' + 'End Q10' + 20 * '-')
# =================================================================
# Class_Ex11:
# Grab arbitrary paragraph from any website.
# Creat  a list of stopwords manually.  Example :  stopwords = ['and', 'for', 'in', 'little', 'of', 'the', 'to']
# Create a list of ignore char Example: ' :,",! '
# Write a LSA class with the following functions.
# Parse function which tokenize the words lower cases them and count them. Use dictionary; keys are the tokens and value is count.
# Clac function that calculate SVD.
# TFIDF function
# Print function which print out the TFIDF matrix, first 3 columns of the U matrix and first 3 rows of the Vt matrix
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')

print(20 * '-' + 'End Q11' + 20 * '-')
# =================================================================
# Class_Ex12:
# Use the following doc
#  = ["An intern at OpenAI", "Developer at OpenAI", "A ML intern", "A ML engineer" ]
# Calculate the binary BOW.
# Use LSA method and distinguish two different topic from the document. Sent 1,2 is about OpenAI and sent3, 4 is about ML.
# Use pandas to show the values of dataframe and lsa components. Show there is two distinct topic.
# Use numpy take the absolute value of the lsa matrix sort them and use some threshold and see what words are the most important.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

# Define the list of documents
documents = ["An intern at OpenAI", "Developer at OpenAI", "A ML intern", "A ML engineer"]

# Step 1: Calculate Binary Bag-of-Words (BOW)
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(documents)

# Step 2: Perform LSA
num_topics = 2
lsa = TruncatedSVD(n_components=num_topics)
lsa_result = lsa.fit_transform(X)

# Step 3: Create a Pandas DataFrame to display results
topic_labels = [f"Topic {i+1}" for i in range(num_topics)]
df = pd.DataFrame(lsa_result, columns=topic_labels, index=documents)

# Display the DataFrame
print("LSA Components:")
print(df)

# Step 4: Identify the most important words using a threshold
lsa_components = lsa.components_
threshold = 0.3  # Adjust this threshold as needed
important_words = []

for i, topic in enumerate(topic_labels):
    component = lsa_components[i]
    important_indices = np.where(np.abs(component) > threshold)[0]
    important_words_topic = [vectorizer.get_feature_names_out()[index] for index in important_indices]
    important_words.append((topic, important_words_topic))

# Print the most important words for each topic
for topic, words in important_words:
    print(f"Important words for {topic}: {', '.join(words)}")

print(20 * '-' + 'End Q12' + 20 * '-')
# =================================================================
# %%
