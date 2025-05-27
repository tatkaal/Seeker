# core/text_processor.py
import spacy
import spacy.cli
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

# send spaCy to GPU
spacy.require_gpu()

DEFAULT_SPACY_MODEL = "en_core_web_trf"
# DEFAULT_SPACY_MODEL = "en_core_web_lg"

class TextProcessor:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.nlp_spacy = spacy.load(DEFAULT_SPACY_MODEL)
        except OSError:
            print(f"Downloading spaCy model {DEFAULT_SPACY_MODEL}…")
            spacy.cli.download(DEFAULT_SPACY_MODEL)
            self.nlp_spacy = spacy.load(DEFAULT_SPACY_MODEL)

    def preprocess(self, text:str, lemmatize:bool=True, remove_stopwords:bool=True)->str:
        if not text or not isinstance(text,str):
            return ""
        doc = self.nlp_spacy(text.lower())
        out=[]
        for tok in doc:
            if tok.is_punct or tok.is_space: continue
            if remove_stopwords and tok.text in self.stop_words: continue
            out.append(tok.lemma_ if lemmatize else tok.text)
        return " ".join(out)

    def extract_ngrams(self, text:str, n:int=2, top_k:int=10):
        tokens = word_tokenize(text)
        ngrams = nltk.ngrams(tokens, n)
        return Counter(ngrams).most_common(top_k)
