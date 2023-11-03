# %%
import nltk
from nltk.corpus import nps_chat
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess a single post
def preprocess_post(post):
    # Tokenize the post
    tokens = word_tokenize(post.lower())
    
    # Lemmatize and remove stopwords
    preprocessed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join the preprocessed tokens to form the preprocessed text
    preprocessed_text = ' '.join(preprocessed_tokens)
    
    return preprocessed_text

# Access chat posts
posts = nps_chat.xml_posts()

# Print the preprocessed texts (for demonstration purposes)
for i, post in enumerate(posts[:5]):
    print(f"Raw Text {i+1}: {post.text}")

# Preprocess and collect preprocessed text for each post
preprocessed_texts = [preprocess_post(post.text) for post in posts]

# Print the preprocessed texts (for demonstration purposes)
for i, preprocessed_text in enumerate(preprocessed_texts[:5]):
    print(f"Preprocessed Text {i+1}: {preprocessed_text}")

# tf-idf
# Create a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed text
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)

print("Numerical representation of the summarized content:")
print(tfidf_matrix)
# %%
