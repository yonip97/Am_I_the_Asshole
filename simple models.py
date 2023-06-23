from utils import create_data

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
def bag_of_words(data):
    tf_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),
                                    strip_accents='unicode',
                                    stop_words='english',
                                    lowercase=True)
    vectors = np.array(tf_vectorizer.fit_transform(data.tolist()).todense())
    return vectors
def distribution_per_row(row,classes):
    row = row.dropna()
    distribution = np.zeros(classes)
    for entry in row:
        entry = int(entry)
        distribution[entry-1] += 1/len(row)
    return distribution
def transform_to_distribution(data,classes):
    return data.apply(lambda x:distribution_per_row(x,classes),axis = 1).tolist()


def main():
    full_data, annotators_names = create_data('data/labeled/full_annotation_team_1')
    features = bag_of_words(full_data['selftext'])
    labels = transform_to_distribution(full_data[annotators_names],classes = 5)

if __name__ == '__main__':
    main()