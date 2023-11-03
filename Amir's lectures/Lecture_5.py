#%%
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np

'''
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, 
partitioned (nearly) evenly across 20 different newsgroups. To the best of our knowledge, 
it was originally collected by Ken Lang, probably for his paper “Newsweeder: Learning 
to filter netnews,” though he does not explicitly mention this collection. The 20 
newsgroups collection has become a popular data set for experiments in text applications
of machine learning techniques, such as text classification and text clustering.
'''

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

print(twenty_train.target_names)
print(len(twenty_train.data))
print(len(twenty_train.filenames))

3
tfidf= TfidfVectorizer()
tfidf.fit(twenty_train.data)
X_train_tfidf =tfidf.transform(twenty_train.data)
print(X_train_tfidf.shape)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
X_test_tfidf = tfidf.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)

print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))
#%%
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np

'''
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, 
partitioned (nearly) evenly across 20 different newsgroups. To the best of our knowledge, 
it was originally collected by Ken Lang, probably for his paper “Newsweeder: Learning 
to filter netnews,” though he does not explicitly mention this collection. The 20 
newsgroups collection has become a popular data set for experiments in text applications
of machine learning techniques, such as text classification and text clustering.
'''

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

print(twenty_train.target_names)
print(len(twenty_train.data))
print(len(twenty_train.filenames))


tfidf= TfidfVectorizer()
tfidf.fit(twenty_train.data)
X_train_tfidf =tfidf.transform(twenty_train.data)
print(X_train_tfidf.shape)

clf = LogisticRegression().fit(X_train_tfidf, twenty_train.target)
X_test_tfidf = tfidf.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)

print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))

#%%
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

categories = ['rec.motorcycles', 'sci.electronics','comp.graphics', 'sci.med']

# sklearn provides us with subset data for training and testing
train_data = fetch_20newsgroups(subset='train',
                                categories=categories, shuffle=True, random_state=42)

print(train_data.target_names)

print("\n".join(train_data.data[0].split("\n")[:3]))
print(train_data.target_names[train_data.target[0]])

# Let's look at categories of our first ten training data
for t in train_data.target[:10]:
    print(train_data.target_names[t])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data.data)

# transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, train_data.target)

# Input Data to predict their classes of the given categories
docs_new = ['I have a Harley Davidson and Yamaha.', 'I have a GTX 1050 GPU']
# building up feature vector of our input
X_new_counts = count_vect.transform(docs_new)
# We call transform instead of fit_transform because it's already been fit
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicting the category of our input text: Will give out number for category
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train_data.target_names[category]))

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
# Fitting our train data to the pipeline
text_clf.fit(train_data.data, train_data.target)

# Test data
test_data = fetch_20newsgroups(subset='test',
                               categories=categories, shuffle=True, random_state=42)
docs_test = test_data.data
# Predicting our test data
predicted = text_clf.predict(docs_test)
print('We got an accuracy of',np.mean(predicted == test_data.target)*100, '% over the test data.')