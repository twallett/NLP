#%%
#**********************************
import nltk
import string
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize

#**********************************
#==================================================================================================================================================================
# Q1:
"""
For this question you need to download a text data from the NLTK movie_reviews.
Use you knowledge that you learned in the class and clean the text appropriately.
After Cleaning is done, please find the numerical representation of text by any methods that you learned.
You need to find a creative way to label the sentiment of the sentences.
This dataset already has positive and negative labels.
Labeling sentences as 'positive' or 'negative' based on sentiment scores and named then predicted sentiments.
Create a Pandas dataframe with sentences, true sentiment labels and predicted sentiment labels.
Calculate the accuracy of your predicted sentiment and true sentiments.
"""
#==================================================================================================================================================================

print("helo")

# Load the movie_reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Initialize a lemmatizer
lemmatizer = WordNetLemmatizer()

# Remove punctuation and convert to lowercase
def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    return text

# Tokenize and lemmatize the words, removing stopwords
def preprocess_text(words):
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Apply cleaning and preprocessing to the documents
cleaned_documents = [preprocess_text(clean_text(' '.join(document))) for document, category in documents]

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Fit and transform the cleaned text data
X = tfidf_vectorizer.fit_transform(cleaned_documents)

# Labels for sentiment (0 for negative, 1 for positive)
y = [1 if category == 'pos' else 0 for _, category in documents]


# %%
