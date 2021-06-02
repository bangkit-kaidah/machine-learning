import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem.snowball import SnowballStemmer

stopword = StopWordRemoverFactory().create_stop_word_remover()
stemmer_indo = StemmerFactory().create_stemmer()
stemmer_eng = SnowballStemmer("english")

def clean_punc(sentence):
    """ Keep only words containing letters A-Z and a-z.
        Remove all punctuations, numbers, etc. """

    cleaned_sentence = ""
    for word in sentence.split():
        cleaned_word = re.sub('[^a-zA-Z]',' ', word)
        cleaned_sentence += cleaned_word.strip()
        cleaned_sentence += " "
    return cleaned_sentence.strip()

def remove_stop_words_indo(sentence):
    """ Remove stop words for Bahasa Indonesia sentence. """
    
    return stopword.remove(sentence)

def stemming_indo(sentence):
    """ Stemming for Bahasa Indonesia sentence. """

    return stemmer_indo.stem(sentence)

def stemming_eng(sentence):
    """ Stemming for English sentence. """

    stemmed_sentence = ""
    for word in sentence.split():
        stemmed_word = stemmer_eng.stem(word)
        stemmed_sentence += stemmed_word
        stemmed_sentence += " "
    return stemmed_sentence.strip()