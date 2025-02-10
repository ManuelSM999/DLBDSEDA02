import re
import csv
import nltk
import pandas as pd
from tabulate import tabulate
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.corpus import words

# download wordl ist from nltk
nltk.download('words')


def print_topics(model, words_per_topic):
    # Iterate over topics = components
    for i, component in enumerate(model.components_):
        # combine words with their value within the component
        vocab_comp = zip(tfidf_vectorizer.get_feature_names_out(), component)

        # first sort words by their value within the component (reversed)
        # then choose the ten first words as representatives for the topic
        sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:words_per_topic]
        print(f"Topic {str(i)}: ")
        for t in sorted_words:
            print(t[0], end=" ")
        print()


reviews = []
word_list = set(words.words())

with open('Musical_instruments_reviews.csv', mode='r', newline='', encoding='utf-8') as file:
    content = csv.DictReader(file)
    for review in content:
        dirty_review_text = review['reviewText']
        # only letters and spaces to tokenize the words
        clean_review_text = re.sub(r'[^a-zA-Z\s]', '',
                                   dirty_review_text)
        tokenize_words = clean_review_text.split()
        # check if a word is included in the downloaded dictionary (word_list)
        valid_words = [word for word in tokenize_words if word in word_list]
        reviews.append(" ".join(valid_words))

stop_words = stopwords.words('english')
# vectorizers assign a value to all words in all reviews
# init vectorizers with Parameters (stopwords)
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
count_vectorizer = CountVectorizer(stop_words=stop_words)
# add data to vectorizers
tfidf_data = tfidf_vectorizer.fit_transform(reviews)
count_data = count_vectorizer.fit_transform(reviews)

# convert vectors to DataFrames
tfidf_data_frame = pd.DataFrame(tfidf_data.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
count_data_frame = pd.DataFrame(count_data.toarray(), columns=count_vectorizer.get_feature_names_out())
# sum of all words from the reviews; ascending=False to get most frequent words
tfidf_sums = tfidf_data_frame.sum().sort_values(ascending=False)
count_sums = count_data_frame.sum().sort_values(ascending=False)

# to pretty print the data with Pandas
print('-----------------Top 10 TF-IDF_Vector-----------------')
print(tabulate(tfidf_sums.head(10).reset_index(), tablefmt='pretty', headers=['Word', 'Value']))
print('-----------------Top 10 Count_Vector-----------------')
print(tabulate(count_sums.head(10).reset_index(), tablefmt='pretty', headers=['Word', 'Value']))
print()

number_of_topics = 5
number_of_words_per_topic = 10

# LSA Model
lsa_model = TruncatedSVD(n_components=number_of_topics, algorithm="randomized",
                         n_iter=10)  # building an SVD matrix using TruncatedSVD
lsa_model.fit_transform(tfidf_data)  # Use word values from tfidf for lsa model

print('-----------------LSA-----------------------')
print_topics(lsa_model, number_of_words_per_topic)

print()
# LDA
lda_model = LatentDirichletAllocation(n_components=number_of_topics, learning_method='online', max_iter=1,
                                      random_state=42)
lda_model.fit(tfidf_data)

print('-----------------LDA-----------------------')
print_topics(lda_model, number_of_words_per_topic)
