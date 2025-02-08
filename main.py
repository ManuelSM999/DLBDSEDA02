import re
import csv

from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def print_topics(model, number_of_words_per_topic):
    # Interate over topics = components
    for i, component in enumerate(model.components_):
        # combine words with their value within the component
        vocab_comp = zip(tfidf_vectorizer.get_feature_names_out(), component)

        # first sort words by their value within the component (reversed)
        # then choose the ten first words as representatives for the topic
        sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:number_of_words_per_topic]
        print(f"Topic {str(i)}: ")
        for t in sorted_words:
            print(t[0], end=" ")
        print()


reviews = []

stop_words = stopwords.words('english')

with open('Musical_instruments_reviews.csv', mode='r', newline='', encoding='utf-8') as file:
    content = csv.DictReader(file)
    for review in content:
        dirty_review_text = review['reviewText']
        clean_review_text = re.sub(r'[^a-zA-Z\s]', '',
                                   dirty_review_text)  # only letters and spaces to tokanize the words
        reviews.append(clean_review_text)

# vectorizers assign a value to all words in all reviews
# init vectorizers with Parameters
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
count_vectorizer = CountVectorizer(stop_words=stop_words)

# add data to vectorizers
tfidf_data = tfidf_vectorizer.fit_transform(reviews)
count_data = count_vectorizer.fit_transform(reviews)

# # to pretty print the data with Pandas
# tfidf_data_frame = pd.DataFrame(tfidf_data.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
# count_data_frame = pd.DataFrame(count_data.toarray(), columns=count_vectorizer.get_feature_names_out())


number_of_topics = 5
number_of_words_per_topic = 10

# LSA Model
lsa_model = TruncatedSVD(n_components=number_of_topics, algorithm="randomized",
                         n_iter=10)  # Aufbau einer SVD matrix mithilfe von TruncatedSVD
lsa_model.fit(tfidf_data, number_of_words_per_topic)  # Use word values from tfidf for lsa model

print('-----------------LSA-----------------------')
print_topics(lsa_model, number_of_words_per_topic)

print()
# LDA
lda_model = LatentDirichletAllocation(n_components=number_of_topics, learning_method='online', max_iter=1)
lda_model.fit(tfidf_data)

print('-----------------LDA-----------------------')
print_topics(lda_model, number_of_words_per_topic)
