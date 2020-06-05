import pandas as pd
import numpy as np
import nltk
from create_dataframe import Data
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
np.random.seed(SEED)

file = "./imdb-reviews-pt-br.csv"
data = Data(file)
data.generate_punct_token_stemmed()

def classify_text(text, text_column, column_classification):
  vectorize = CountVectorizer(lowercase=False, max_features=50)
  bag_of_words = vectorize.fit_transform(text[text_column])
  train, test, train_class, test_class = train_test_split(bag_of_words, text[column_classification])
  
  logistic_regression = LogisticRegression() #solver = "lbfgs"
  logistic_regression.fit(train, train_class)

  accuracy = logistic_regression.score(test, test_class)

  sparse_array = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vectorize.get_feature_names())

  return accuracy, sparse_array

def classify_text_tfidf(text, text_column, column_classification):
  tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2)) #max_features=50
  bag_of_words = tfidf.fit_transform(text[text_column])
  train, test, train_class, test_class = train_test_split(bag_of_words, text[column_classification])
  
  logistic_regression = LogisticRegression() #solver = "lbfgs"
  logistic_regression.fit(train, train_class)

  accuracy = logistic_regression.score(test, test_class)

  weights = pd.DataFrame(logistic_regression.coef_[0].T, index = tfidf.get_feature_names())

  return accuracy, weights

# print(weights.nlargest(50,0))
# print(classificar_text(review, "text_pt", "classificacao"))

def plot_wordcloud(wordcloud, title):
  plt.figure(figsize=(10,7))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.title(title)
  # plt.show()

def generate_wordcloud_neg(data, token, text, text_column, quantity):
  text_negative = text.query("sentiment == 'neg'")
  all_words = ' '.join([text for text in text_negative[text_column]])
  data_words = data.most_repeated_words(all_words, token, quantity)
  
  generate_wordcloud = WordCloud(width= 800, height= 500, max_font_size = 110, collocations = False).generate(all_words)

  plot_wordcloud(generate_wordcloud, 'Negative')

  return data_words

def generate_wordcloud_pos(data, token, text, text_column, quantity):
  text_positive = text.query("sentiment == 'pos'")
  all_words = ' '.join([text for text in text_positive[text_column]])
  data_words = data.most_repeated_words(all_words, token, quantity)
  
  generate_wordcloud = WordCloud(width= 800, height= 500, max_font_size = 110, collocations = False).generate(all_words)

  plot_wordcloud(generate_wordcloud, 'Positive')

  return data_words
  
def pareto(data, token, text, text_column, quantity):
  all_words = ' '.join([text for text in text[text_column]])
  df_frequency = data.most_repeated_words(all_words, token, quantity)

  plt.figure(figsize=(12,8))
  ax = sns.barplot(data = df_frequency, x = "word", y = "frequency", color = 'gray')
  ax.set(ylabel = "Count")
  # plt.show()

accuracy, sparse_array = classify_text_tfidf(data.review, "treatment_5", "classification")

pareto(data, data.punct_token, data.review, "treatment_5", 10)

most_repeated_negative = generate_wordcloud_neg(data, data.punct_token, data.review, "treatment_5", 10)
most_repeated_positive = generate_wordcloud_pos(data, data.punct_token, data.review, "treatment_5", 10)
# print(most_repeated_negative, most_repeated_positive, end='\n\n', sep='\n\n')

