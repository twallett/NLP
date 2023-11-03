#%%
# Natural language toolkit

import nltk
nltk.download()

#%%

from nltk.book import *
import nltk

print(text1.vocab())
print(type(text1))
print(len(text1))

#%%

from nltk.corpus import gutenberg

print(gutenberg.fileids())
print(nltk.corpus.gutenberg.fileids())
hamlet = gutenberg.words('shakespeare-hamlet.txt')

from nltk.corpus import inaugural

print(inaugural.fileids())
print(nltk.corpus.inaugural.fileids())

from nltk.text import Text

former_president = Text(inaugural.words(inaugural.fileids()[-1]))
print(' '.join(former_president.tokens[0:1000]))
# %%

from nltk.book import text1
from nltk.book import text4
from nltk.book import text6

print(text1.concordance("monstrous"))
print(text1.similar("monstrous"))
print(text1.collocations())
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

print(text6.count("Very"))
print(text6.count('the') / float(len(text6)) * 100)
print(text4.count("bless"))
print(text4[100])
print(text4.index('the'))
print(text4[524])
print(text4.index('men'))
print(text4[0:len(text4)])

#%%
from nltk.book import text1
from nltk.book import text4

print(text4[100])
print(text4.index('the'))
print(text4[524])
print(text4.index('men'))
print(text4[0:len(text4)])

print(set(text4))
print(sorted(set(text4)))
print(sorted(set(text4)))
print(len(set(text4)))

T1_diversity = float(len(set(text1))) / float(len(text1))
print("The lexical diversity is: ", T1_diversity * 100, "%")
T4_diversity = float(len(set(text4))) / float(len(text4))
print("The lexical diversity is: ", T4_diversity * 100, "%")

#%%

from nltk.book import text1
from nltk.book import text4
from nltk import FreqDist
import nltk
Freq_Dist = FreqDist(text1)
print(Freq_Dist)
print(Freq_Dist.most_common(10))
print(Freq_Dist['his'])
Freq_Dist.plot(50, cumulative = False)
Freq_Dist.plot(50, cumulative = True)
Freq_Dist.hapaxes()
Once_happend= Freq_Dist.hapaxes() ; print(Once_happend)
print(text4.count('america') / float(len(text4) * 100))

Value_set = set(text1)
long_words = [words for words in Value_set if len(words) > 17]
print(sorted(long_words))
my_text = ["Here", "are", "some", "words", "that", "are", "in", "a", "list"]
vocab = sorted(set(my_text)) ; print(vocab)
word_freq = nltk.FreqDist(my_text); print(word_freq.most_common(5))

#%%

from nltk.corpus import gutenberg
import nltk

print(gutenberg.fileids())
emma = gutenberg.words('austen-emma.txt')
print(len(emma))
emma_Text = nltk.Text(gutenberg.words('austen-emma.txt'))
emma_Text.concordance("surprize")
for fileid in gutenberg.fileids():
   num_chars = len(gutenberg.raw(fileid))
   num_words = len(gutenberg.words(fileid))
   num_sents = len(gutenberg.sents(fileid))
   num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
   print(round(num_chars / num_words), round(num_words / num_sents),
         round(num_words / num_vocab), fileid)

macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt');
print(macbeth_sentences)
print(macbeth_sentences[1116])
longest_len = max(len(s) for s in macbeth_sentences); print(longest_len)
print( [s for s in macbeth_sentences if len(s) == longest_len])

#%%

from nltk.corpus import webtext
from nltk.corpus import nps_chat

for fileid in webtext.fileids():
   print(fileid, webtext.raw(fileid)[:65])

text = webtext.raw('firefox.txt')
print([i for i in range(len(text)) if text.startswith('a', i)])

chatroom = nps_chat.posts('10-19-20s_706posts.xml')
print(chatroom[123])

text2 = nps_chat.raw('11-09-teens_706posts.xml')

#%%

from nltk.corpus import brown
import nltk
print(brown.categories())
print(brown.words(categories='news'))
print( brown.words(fileids=['cg22']))
print(brown.sents(categories=['news', 'editorial', 'reviews']))
from nltk.corpus import brown
news_text = brown.words(categories='news')
fdist = nltk.FreqDist(w.lower() for w in news_text)
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m + ':', fdist[m], end=' ')

cfd = nltk.ConditionalFreqDist((genre, word)
            for genre in brown.categories()
            for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
print(); print(cfd.tabulate(conditions=genres, samples=modals))

#%%

from nltk.corpus import reuters
print(reuters.fileids())
print(reuters.categories())
print(reuters.categories('training/9865'))
print(reuters.categories(['training/9865', 'training/9880']))
print(reuters.fileids(['barley', 'corn']))

print(reuters.words('training/9865')[:14])
print(reuters.words(['training/9865', 'training/9880']))
print(reuters.words(categories='barley'))
print(reuters.words(categories=['barley', 'corn']))

#%%

from nltk.corpus import inaugural
print(inaugural.fileids())
print([fileid[:4] for fileid in inaugural.fileids()])

import nltk
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))
cfd.plot()

#%%

from nltk.corpus import PlaintextCorpusReader
import os

corpus_root = os.getcwd()
wordlists = PlaintextCorpusReader(corpus_root, 'Corpus.txt')

print(wordlists.fileids())
print(wordlists.words('Corpus.txt'))

#%%

from nltk.corpus import brown
import nltk

genre_word = [(genre, word)
           for genre in ['news', 'romance']
           for word in brown.words(categories=genre)]

print( genre_word[:4]); print(genre_word[-4:])
cfd = nltk.ConditionalFreqDist(genre_word)
print(cfd.conditions())

print(cfd['news'])
print(cfd['romance'])
cfd['romance'].most_common(20)

#%%
from nltk.corpus import inaugural
import nltk
cfd = nltk.ConditionalFreqDist((target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))
print(cfd['america'].most_common(20))
cfd.tabulate(conditions=['america', 'citizen']);cfd.plot()

from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch',
             'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
           (lang, len(word))
          for lang in languages
          for word in udhr.words(lang + '-Latin1'))
cfd.tabulate(conditions=['English', 'German_Deutsch'],
             samples=range(10), cumulative=True)

#%%

import nltk
sent = ['In', 'the', 'beginning', 'God', 'created',
        'the', 'heaven', 'and', 'the', 'earth', '.']
print(list(nltk.bigrams(sent)))

def generate_model(cfdist, word, num=10):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
text1 = text[0:]
bigrams = nltk.bigrams(text)
print(list(nltk.bigrams(text1))[0:20])
cfd = nltk.ConditionalFreqDist(bigrams)
print(cfd['living'])
generate_model(cfd, 'living')

#%%

import nltk
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)
print(unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt')))
print( unusual_words(nltk.corpus.nps_chat.words()))

from nltk.corpus import stopwords
print(stopwords.words('english'))

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)
print(content_fraction(nltk.corpus.reuters.words()))

#%%

import nltk
names = nltk.corpus.names
print(names.fileids())
male_names = names.words('male.txt')
female_names = names.words('female.txt')
print([w for w in male_names if w in female_names])

cfd = nltk.ConditionalFreqDist(
         (fileid, name[-1])
         for fileid in names.fileids()
         for name in names.words(fileid))
cfd.plot()

#%%

import nltk

entries = nltk.corpus.cmudict.entries()
print(len(entries))
for entry in entries[42371:42379]:
    print(entry)

for word, pron in entries:
    if len(pron) == 3:
        ph1, ph2, ph3 = pron
        if ph1 == 'P' and ph3 == 'T':
            print(word, ph2, end=' ')
            print()

syllable = ['N', 'IH0', 'K', 'S']
print([word for word, pron in entries if pron[-4:] == syllable])
print([w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n'])
print(sorted(set(w[:2] for w, pron in entries if pron[0] == 'N' and w[0] != 'n')))

#%%

import nltk
entries = nltk.corpus.cmudict.entries()

def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]
print( [w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']])
print([x[1] for x in entries if x[0]=='abbreviated'])

p3 = [(pron[0]+'-'+pron[2], word)
       for (word, pron) in entries
       if pron[0] == 'P' and len(pron) == 3]
cfd = nltk.ConditionalFreqDist(p3)
for template in sorted(cfd.conditions()):
     if len(cfd[template]) > 10:
        words = sorted(cfd[template])
        wordstring = ' '.join(words)
        print(template, wordstring[:70] + "...")
prondict = nltk.corpus.cmudict.dict()
prondict['blog'] = [['B', 'L', 'AA1', 'G']]
print(prondict['blog'])

#%%


from nltk.corpus import wordnet as wn
print(wn.synsets('motorcar'))
print(wn.synset('car.n.01').lemma_names())
print(wn.synset('car.n.01').definition())
print(wn.synset('car.n.01').examples())
print(wn.synset('car.n.01').lemmas())
print(wn.lemma('car.n.01.automobile'))
print(wn.lemma('car.n.01.automobile').synset())
print(wn.lemma('car.n.01.automobile').name())
print(wn.synsets('car'))
for synset in wn.synsets('car'):
    print(synset.lemma_names())
print(wn.lemmas('car'))

motorcar = wn.synset('car.n.01'); print(motorcar)
types_of_motorcar = motorcar.hyponyms()
sorted(lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas())
print(motorcar.hypernyms())
paths = motorcar.hypernym_paths()
print([synset.name() for synset in paths[0]])
print([synset.name() for synset in paths[1]])

#%%

from nltk.corpus import wordnet as wn
right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
minke = wn.synset('minke_whale.n.01')
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
print(right.lowest_common_hypernyms(minke))
print(right.lowest_common_hypernyms(orca))
print(right.lowest_common_hypernyms(tortoise))

print(right.path_similarity(minke))
print(wn.synset('whale.n.02').min_depth())
print(wn.synset('vertebrate.n.01').min_depth())
print(wn.synset('entity.n.01').min_depth())

#%%

from urllib import request
from nltk import word_tokenize
import  nltk

url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
print(type(raw));print(len(raw));print(raw[:75])

tokens = word_tokenize(raw)
print(type(tokens))
print(tokens[:10])
text = nltk.Text(tokens)
print(type(text))
print(text.collocations())

#%%

from urllib import request
from bs4 import BeautifulSoup
from nltk import word_tokenize
import nltk

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = request.urlopen(url).read().decode('utf8')
print(html[:60])


raw = BeautifulSoup(html, 'html.parser').get_text()
tokens = word_tokenize(raw); print(tokens)
print(tokens = tokens[110:390])
text = nltk.Text(tokens); print(text)
print(text.concordance('gene'))

#%%

from nltk import word_tokenize
from nltk import Text

f = open('Corpus.txt')
raw = f.read()
f = open('Corpus.txt', 'r')
for line in f:
    print(line.strip())

words_token = word_tokenize(raw)
text = Text(words_token)
text.dispersion_plot(['corpus'])

#%%

from nltk import word_tokenize
import nltk
from nltk import Text

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
 is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
print([porter.stem(t) for t in tokens])
print( [lancaster.stem(t) for t in tokens])

porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = Text(grail)
text.concordance('lie')

#%%

import nltk
from nltk import word_tokenize

raw = """DENNIS: Listen, strange women lying in ponds distributing swords
 is no basis for a system of government.  Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""

tokens = word_tokenize(raw)

wnl = nltk.WordNetLemmatizer()
print([wnl.lemmatize(t) for t in tokens])

#%%

import nltk

text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
sents = nltk.sent_tokenize(text)
print(sents[0])


sentences = text.split('. ')
words_in_sentences = [sentence.split(' ') for sentence in sentences]
print(sentences[0])

#%%

import nltk
from nltk import word_tokenize

text = word_tokenize("And now for something completely different")
print(text)
tagged = nltk.pos_tag(text)
print(tagged)
print(nltk.help.upenn_tagset('RB'))


text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
print(text.similar('woman'))
print(text.similar('bought'))
print(text.similar('over'))
print(text.similar('the'))

#%%

from nltk.corpus import brown
import nltk
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
print(tag_fd.most_common())

#%%

from nltk.corpus import brown
import nltk

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print(unigram_tagger.tag(brown_sents[2007]))

print(unigram_tagger.evaluate(brown_tagged_sents))

size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))

