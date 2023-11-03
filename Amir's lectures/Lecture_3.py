#%%
from spacy.lang.en import English
nlp = English()

doc = nlp("Hello world!")
for token in doc:
    print(token.text)

token = doc[1]
print(token.text)

span = doc[1:3]
print(span.text)

doc = nlp("It costs $5.")
print("Index:   ", [token.i for token in doc])
print("Text:    ", [token.text for token in doc])
print("is_alpha:", [token.is_alpha for token in doc])
print("is_punct:", [token.is_punct for token in doc])
print("like_num:", [token.like_num for token in doc])

#%%
import spacy
nlp = spacy.load("en_core_web_sm")

#%%
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("She ate the pizza")
for token in doc:
    print(token.text, token.pos_)

#%%
# https://spacy.io/usage/linguistic-features
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("She ate the pizza")
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)

#%%
# https://spacy.io/usage/linguistic-features
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(ent.text, ent.label_)

#%%
import spacy

print(spacy.explain("GPE"))
print(spacy.explain("NNP"))
print(spacy.explain("dobj"))

#%%
# Match exact token texts
[{"TEXT": "iPhone"}, {"TEXT": "X"}]

# Match lexical attributes
[{"LOWER": "iphone"}, {"LOWER": "x"}]

# Match any token attributes
[{"LEMMA": "buy"}, {"POS": "NOUN"}]

#%%
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]
matcher.add("IPHONE_PATTERN", [pattern])

doc = nlp("Upcoming iPhone X release date leaked")
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)
    
# %%

import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
pattern = [{"IS_DIGIT": True}, {"LOWER": "fifa"}, {"LOWER": "world"},
           {"LOWER": "cup"},  {"IS_PUNCT": True}]

matcher.add("FIFA", [pattern])
doc = nlp("2018 FIFA World Cup: France won!")
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)

#%%

import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [
    {"LEMMA": "love", "POS": "VERB"},
    {"POS": "NOUN"}
]
matcher.add("Other", [pattern])

doc = nlp("I loved dogs but now I love cats more.")
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)

#%%

import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

pattern = [{"LEMMA": "buy"}, {"POS": "DET", "OP": "?"},  {"POS": "NOUN"}]
matcher.add("Other", [pattern])

doc = nlp("I bought a smartphone. Now I'm buying apps.")
matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)

#%%

import spacy

nlp = spacy.load("en_core_web_sm")

print(nlp.vocab.strings.add("text"))

coffee_hash = nlp.vocab.strings["text"]

coffee_string = nlp.vocab.strings[coffee_hash]

print(coffee_string)

#%%

import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("I love natural language processing.")
lexeme = nlp.vocab["natural"]
print(lexeme.text, lexeme.orth, lexeme.is_alpha)


#%%

from spacy.tokens import Doc, Span
from spacy.lang.en import English
nlp = English()

words = ["Hello", "world", "!"]
spaces = [True, False, False]

doc = Doc(nlp.vocab, words=words, spaces=spaces)
span = Span(doc, 0, 2)
span_with_label = Span(doc, 0, 2, label="GREETING")
doc.ents = [span_with_label]; print(doc.ents)

#%%
# python -m spacy download en_core_web_md
import spacy
nlp = spacy.load("en_core_web_md")


doc1 = nlp("I like fast food")
doc2 = nlp("I like pizza")
print(doc1.similarity(doc2))

doc = nlp("I like pizza and pasta")
token1 = doc[2]
token2 = doc[4]
print(token1.similarity(token2))
#%%

import spacy
nlp = spacy.load("en_core_web_md")

doc = nlp("I like pizza")
token = nlp("soap")[0]

print(doc.similarity(token))
span = nlp("I like pizza and pasta")[2:5]
doc = nlp("McDonalds sells burgers")

print(span.similarity(doc))

#%%

import spacy
nlp = spacy.load("en_core_web_md")

doc = nlp("I have a banana")
print(doc[3].vector)

#%%

from spacy.matcher import Matcher
import spacy
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

matcher.add("CAR", [[{"LOWER": "golden"}, {"LOWER": "car"}]])
doc = nlp("I have a Golden Car")
for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print("Matched span:", span.text)
    # Get the span's root token and root head token
    print("Root token:", span.root.text)
    print("Root head token:", span.root.head.text)
    # Get the previous token and its POS tag
    print("Previous token:", doc[start - 1].text, doc[start - 1].pos_)

#%%

import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)

pattern = nlp("Golden Car")
matcher.add("CAR", [pattern])
doc = nlp("I have a Golden Car")

for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print("Matched span:", span.text)

#%%

import spacy
nlp = spacy.load("en_core_web_sm")
print(nlp.pipe_names)
print(nlp.pipeline)

#%%

import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.language import Language

@Language.component("component1")
def custom_component1(doc):
    print("Doc length:", len(doc))
    return doc
nlp.add_pipe('component1', name="component-info-1", first=True)
print("Pipeline:", nlp.pipe_names)

@Language.component("component2")
def custom_component2(doc):
    print("Doc length:", len(doc))
    return doc

nlp.add_pipe('component2', name="component-info-2", first=True)
doc = nlp("Hello world!")
print(doc)

#%%

from spacy.tokens import Doc, Token, Span
Doc.set_extension("title", default=None)
Token.set_extension("is_color", default=False)
Span.set_extension("has_color", default=False)

#%%

from spacy.tokens import Token
import spacy
nlp = spacy.load("en_core_web_sm")

def get_is_color(token):
    colors = ["red", "yellow", "blue"]
    return token.text in colors

Token.set_extension("is_color", getter=get_is_color)

doc = nlp("The sky is blue.")
print(doc[3]._.is_color, "-", doc[3].text)

#%%

from spacy.tokens import Doc
import spacy
nlp = spacy.load("en_core_web_sm")

def has_token(doc, token_text):
    in_doc = token_text in [token.text for token in doc]
    return in_doc

Doc.set_extension("has_token", method=has_token)

doc = nlp("The sky is blue.")
print(doc._.has_token("blue"), "- blue")
print(doc._.has_token("cloud"), "- cloud")

#%%

import spacy
nlp = spacy.load("en_core_web_sm")

# docs = [nlp(text) for text in LOTS_OF_TEXTS]---Slow
# docs = list(nlp.pipe(LOTS_OF_TEXTS))---Fast

# doc = nlp("Hello world")
# doc = nlp.make_doc("Hello world!")

text = 'I love performance'
with nlp.disable_pipes("tagger", "parser"):
    doc = nlp(text)
    print(doc.text)
    
#%%
