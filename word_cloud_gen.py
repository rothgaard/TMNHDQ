import nltk
from wordcloud import WordCloud
from collections import Counter

f = open('./data/text_sample.txt')
raw = f.read()

stop_words_ita = set(nltk.corpus.stopwords.words('italian'))
stop_words_eng = set(nltk.corpus.stopwords.words('english'))

# specialization
stop_words_ita.add('dovrebbero')
stop_words_ita.add('fra')
stop_words_eng.add('')

tokens = nltk.word_tokenize(raw)
words = [w for w in tokens if not w in stop_words_eng and not w in stop_words_ita and w.isalpha()]
words_freq=Counter(words)
wc = WordCloud(max_words=100, width=800, height=600,
 background_color ='white').generate_from_frequencies(words_freq)

wc.to_file("hys_word_cloud.png")

