import pandas as pd
import numpy as np
import nltk
import unidecode
from string import punctuation

nltk.download('stopwords')
nltk.download('rslp')

class Data:
  def __init__(self, file):
    self.review = pd.read_csv(file, encoding='utf-8')
    self.irelevant_words = nltk.corpus.stopwords.words("portuguese")
    self.punctuation_stopwords = None
    self.whitespace_token = None
    self.punct_token = None

    np.random.seed(42)

    classification = self.review["sentiment"].replace(["neg", "pos"], [0,1])
    self.review["classification"] = classification

  def generate_whitespace_token(self):
    self.whitespace_token = nltk.tokenize.WhitespaceTokenizer()

    processed_phrase = list()

    for opinion in self.review.text_pt:
      new_phrase = list()
      text_words = self.whitespace_token.tokenize(opinion)
      for word in text_words:
        if word not in self.irelevant_words:
          new_phrase.append(word)
      processed_phrase.append(' '.join(new_phrase))
        
    self.review["treatment_1"] = processed_phrase

    return self.review, self.whitespace_token

  def generate_punct_token(self):
    self.punct_token = nltk.tokenize.WordPunctTokenizer()

    punctuation = list()
    for punct in punctuation:
      punctuation.append(punct)

    self.punctuation_stopwords = punctuation + self.irelevant_words

    processed_phrase = list()

    for opinion in self.review.text_pt:
      new_phrase = list()
      text_words = self.punct_token.tokenize(opinion)
      for word in text_words:
        if word not in self.punctuation_stopwords:
          new_phrase.append(word)
      processed_phrase.append(' '.join(new_phrase))
        
    self.review["treatment_2"] = processed_phrase

    return self.review, self.punct_token

  def generate_punct_token_unidecode(self):
    self.generate_punct_token()
    self.punct_token = nltk.tokenize.WordPunctTokenizer()

    self.review["treatment_3"] = [unidecode.unidecode(text) for text in self.review["treatment_2"]]
    stopwords_no_punct = [unidecode.unidecode(text) for text in self.punctuation_stopwords]

    processed_phrase = list()

    for opinion in self.review["treatment_3"]:
      new_phrase = list()
      text_words = self.punct_token.tokenize(opinion)
      for word in text_words:
        if word not in stopwords_no_punct:
          new_phrase.append(word)
      processed_phrase.append(' '.join(new_phrase))
        
    self.review["treatment_3"] = processed_phrase

    return self.review, self.punct_token

  def generate_punct_token_lower(self):
    self.punct_token = nltk.tokenize.WordPunctTokenizer()

    punctuation = list()
    for punct in punctuation:
      punctuation.append(punct)

    self.punctuation_stopwords = punctuation + self.irelevant_words

    processed_phrase = list()

    for opinion in self.review.text_pt:
      new_phrase = list()
      opinion = opinion.lower()
      text_words = self.punct_token.tokenize(opinion)
      for word in text_words:
        if word not in self.punctuation_stopwords:
          new_phrase.append(word)
      processed_phrase.append(' '.join(new_phrase))
        
    self.review["treatment_4"] = processed_phrase

    return self.review, self.punct_token

  def generate_punct_token_stemmed(self):
    self.punct_token = nltk.tokenize.WordPunctTokenizer()
    stemmer = nltk.RSLPStemmer()

    punctuation = list()
    for punct in punctuation:
      punctuation.append(punct)

    self.punctuation_stopwords = punctuation + self.irelevant_words

    processed_phrase = list()

    for opinion in self.review.text_pt:
      new_phrase = list()
      opinion = opinion.lower()
      text_words = self.punct_token.tokenize(opinion)
      for word in text_words:
        if word not in self.punctuation_stopwords:
          new_phrase.append(stemmer.stem(word))
      processed_phrase.append(' '.join(new_phrase))
        
    self.review["treatment_5"] = processed_phrase

    return self.review, self.punct_token

  def most_repeated_words(self, words, token, quantidade):
    token_phrase = token.tokenize(words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"word": list(frequency.keys()), "frequency": list(frequency.values())})
    
    return df_frequency.nlargest(columns = "frequency", n = quantidade)