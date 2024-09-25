##################################################
# Introduction to Text Mining and Natural Language Processing
##################################################

##################################################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
##################################################

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling

# !pip install nltk
# !pip install textblob
# !pip install wordcloud

import  string
from warnings import filterwarnings
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ImagePalette import random
from nltk.corpus import stopwords, words
from nltk.sentiment import SentimentIntensityAnalyzer
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################
# 1. Text Preprocessing
##################################################

df: DataFrame = pd.read_csv("C:/Users/Merkez/OneDrive/Masaüstü/nlp-220913-124842/nlp/datasets/amazon_reviews.csv", sep=",")
#print(df.head())


##################################################
#Normalizing Case Folding
##################################################

df['reviewText'] = df['reviewText'].str.lower() #hepsi küçük oldu


# Punctuations
df['reviewText'] = df['reviewText'].str.replace(r'[^\w\s]', '',regex=True) # noktalama işaretleri kaldırıldı olçum değeri taşımadığı için ve yerine boşluk eklendi
#print(df['reviewText'])

# numbers

df['reviewText'] = df['reviewText'].str.replace(r'\d', '',regex=True)
#print(df['reviewText'])

################################
# stopwords (bağlaç veya zamir gibi olçum değeri olmayan ifadeler örneğin: for, the, of vb. [sw] )
#############################

# import nltk kütüphanesinin içinde stopworda kelimelerin olduğu bir yerdir
#nltk.download('stopwords')

sw = stopwords.words('english')
#print(sw)

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
#print(df['reviewText'])


#############################
#Rare Words
#############################

temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()

drops = temp_df[temp_df <=1]
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))
#print(df['reviewText'])

###########################
#Tokenization
##########################

#nltk.download("punkt_tab")

#print(df['reviewText'].apply(lambda x : TextBlob(x).words).head())


#############################
# Lemmatization (Kelimeleri köklerine ayırma işlemidir. Takıları kaldırma) (stemming de vardır )

#nltk.download("wordnet")

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#print(df["reviewText"])


################################################
# 2. Text Visualization
################################################


########################################
# Terim Frekanslarının Hesaplanması
#########################################

tf = df["reviewText"].apply(lambda x : pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf.columns = ["words","tf"]
#print(tf.sort_values("tf",ascending=False))


#############################
#Barplot
#############################

tf[tf["tf"]>500].plot.bar(x="words", y ="tf")
#plt.show()


#############################
#WordCloud
#############################

text = " ".join(i for i in df.reviewText)

wordcloud = WordCloud(max_font_size= 50 , max_words=100, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation = "bilinear")
#plt.axis("off")
#plt.show()
#wordcloud.to_file("wordcloud.png")

#############################
# Şablonlara Göre Wordcloud
#############################
"""
tr_mask = np.array(Image.open("tr.png"))

wc = WordCloud(background_color= "white", max_words=1000, mask=tr_mask, contour_width=3, contour_color="firebrick")
wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation = "bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("tr_mask.png")
"""


################################################
# 2. Sentiment Analysis
################################################

#   print(df["reviewText"].head())
#nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# Exp.
x = sia.polarity_scores("I liked this music but it is not good as the other one")
#print(x)

a = df["reviewText"][0 : 10].apply(lambda x : sia.polarity_scores(x))
#print(a)
a = df["reviewText"][0 : 10].apply(lambda x : sia.polarity_scores(x)["compound"])
#print(a)

df["polarity_score"] = df["reviewText"].apply(lambda x : sia.polarity_scores(x)["compound"])
#print(df["polarity_score"])


################################################
# 3. Sentiment Modeling
################################################




################################################
# 4. Feature Engineering
################################################

a= df["reviewText"][0 :10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"]>0 else "neg")
#print(a)

df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"]>0 else "neg")
#print(df["sentiment_label"])

b= df["sentiment_label"].value_counts()
b=df.groupby("sentiment_label")["overall"].mean()

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

Z=df["sentiment_label"]
S=df["reviewText"]


################################################
# 4.1 Count Vectors
################################################


# count vektors : frekans temsiller
# TF-IDF : normalize edilmiş frekans temsiller
# Work Embedding (Word2Vec, GloVe, BERT vs.)

#words
#kelimelerin nümerik temsilleri

#characters
#karakterlerin nümerik temsilleri

# ngram
a = ("""Bu örneğin anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim
     N-gram' lar birlikte kullanılan kelimelerin kombinasyonlarını gösterir ve feature üretmek için kullanılır""")

#print(TextBlob(a).ngrams(3))

###########################

from sklearn.feature_extraction.text import CountVectorizer

corpus = ["This is the first document.",
          "This document is the second document",
          "And this is the third one.",
          "Is this the first document?"]
print(corpus)

"""
#word frequency
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X_c.toarray())
"""

"""
# n-gram frequency
vectorizer2 = CountVectorizer(analyzer="word", ngram_range=(2, 2))
X_c = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names_out())
print(X_c.toarray())
"""

#df üzerinde gerçek işlem
vectorizer = CountVectorizer()
S_count = vectorizer.fit_transform(S)
print(vectorizer.get_feature_names_out()[10:15])
print(S_count.toarray()[10:15])

################################################
# 4.2 TF-IDF
################################################

from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_word_vectorizer = TfidfVectorizer()
S_tf_idf_word = tf_idf_word_vectorizer.fit_transform(S)

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2,3))
S_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(S)


################################################
# 5. Sentiment Modeling
################################################

################################################
# 5.1 Logistic Regression
################################################

log_model = LogisticRegression().fit(S_tf_idf_word,Z)
a=cross_val_score ( log_model,S_tf_idf_word,Z ,scoring="accuracy",cv=5).mean()
print(a)

# >> 0.830111902339776 %80 DOĞRU TAHMİN YAPACAĞIZ


new_review = pd.Series("this product is great")
new_review = pd.Series("look at that shit very bad")
new_review = TfidfVectorizer().fit(S).transform(new_review)
#print(log_model.predict(new_review))

random_review =pd.Series(df["reviewText"].sample(1).values)
print(random_review)
random_review = TfidfVectorizer().fit(S).transform(random_review)
print(log_model.predict(random_review))

################################################
# 5.2 Random Forrest
################################################
# feature üretme yöntemi

# count vectors
rf_model1 = RandomForestClassifier().fit(S_count,Z)
b=cross_val_score(rf_model1, S_count, Z, cv=5,n_jobs=-1).mean()
print("Count vectors",b)

# TF IDF Word-Level
rf_model2 = RandomForestClassifier().fit(S_tf_idf_word,Z)
c=cross_val_score(rf_model2, S_tf_idf_word, Z, cv=5,n_jobs=-1).mean()
print("TF IDF Word-Level : ",c)

#TF IDF ngram
"""
rf_model3 = RandomForestClassifier().fit(S_tf_idf_ngram,Z)
d=cross_val_score(rf_model3, S_tf_idf_ngram, Z, cv=5,n_jobs=-1).mean()
print("TF IDF ngram : ",d)
"""

################################################
#  Hyperparameter Optimization
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_params ={"max_depth": [8,None],
            "max_features":[7,"auto"],
            "min_samples_split": [2,5,8],
            "n_estimators":[100,200]}

rf_best_grid =GridSearchCV(rf_model,
                           rf_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(S_count,Z)
print(rf_best_grid.best_params_)

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state =17).fit(S_count,Z)
b=cross_val_score(rf_final, S_count, Z, cv=5,n_jobs=-1).mean()
print(b)
