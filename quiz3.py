#%%
import spacy
nlp = spacy.load("en_core_web_sm")
# -----------------------------------------------------------------------------------------
text = """SpaCy is a free, open-source library for advanced Natural Language Processing 
(NLP) in Python. If you're working with a lot of text, you'll eventually want to know 
more about it. For example, what's it about? What do the words mean in context? 
Who is doing what to whom? What companies and products are mentioned?
 Which texts are similar to each other?
SpaCy helps you answer these questions with a powerful, streamlined API that's easy 
to use and integrates seamlessly with other Python libraries. SpaCy also comes with 
pre-trained statistical models and word vectors, and supports deep learning workflows 
that allow you to train and update neural network models on your own data."""
# -----------------------------------------------------------------------------------------
# 1:
# Your code here

doc = nlp(text)

# Tokenize the sentences
sentences = list(doc.sents)
num_sentences = len(sentences)

# Print the number of sentences
print(f"Number of sentences in the text: {num_sentences}")

#%%

# -----------------------------------------------------------------------------------------
# 2
# Your code here

# Tokenize the first sentence
first_sentence = list(doc.sents)[0]

# Count the number of tokens in the first sentence
num_tokens_in_first_sentence = len(first_sentence)

print(f"Number of tokens in the first sentence: {num_tokens_in_first_sentence}")
#%%
# -----------------------------------------------------------------------------------------
#3
# Your code here

import pandas as pd

# Create a table with two columns: 'Token' and 'POS'
data = {"Token": [token.text for token in first_sentence],
        "POS": [token.pos_ for token in first_sentence]}

df = pd.DataFrame(data)

print(df)

#%%
# -----------------------------------------------------------------------------------------
#4
# Your code here

import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

def get_positive_adjectives(text):
    # Process the text using spaCy
    doc = nlp(text)
    
    # Initialize a list to store positive adjectives
    positive_adjectives = []
    
    # Iterate through each token in the processed text
    for token in doc:
        # Check if the token is an adjective
        if token.pos_ == 'ADJ':
            # Get the sentiment scores for the adjective using VADER
            sentiment = sid.polarity_scores(token.text)
            
            # Check if the compound sentiment score is positive
            if sentiment['compound'] > 0:
                positive_adjectives.append(token.text)
    
    return positive_adjectives

# Get the list of positive adjectives in the text
positive_adjectives = get_positive_adjectives(text)

# Print the list of positive adjectives
if positive_adjectives:
    print("Positive Adjectives:")
    for adjective in positive_adjectives:
        print(adjective)
else:
    print("No positive adjectives found in the text.")

print(f"Amount of positive adj:{len(positive_adjectives)}")

#%%
# -----------------------------------------------------------------------------------------
#5
# Your code here

import spacy

# Load the spaCy English tokenizer model
nlp = spacy.load("en_core_web_sm")

def calculate_average_word_length(text):
    # Process the text using spaCy
    doc = nlp(text)
    
    # Initialize variables to keep track of total characters and total words
    total_characters = 0
    total_words = 0
    
    # Iterate through each token in the processed text
    for token in doc:
        # Check if the token is not a punctuation or whitespace
        if not token.is_punct and not token.is_space:
            # Update the total characters and total words
            total_characters += len(token.text)
            total_words += 1
    
    # Calculate the average word length
    if total_words > 0:
        average_word_length = total_characters / total_words
    else:
        average_word_length = 0.0
    
    return average_word_length


# Calculate the average word length
average_length = calculate_average_word_length(text)
print(f"Average Word Length: {average_length:.2f}")

# %%
