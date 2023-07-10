import math

import pandas as pd

from utils import create_data, isNaN

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split,KFold
import optuna

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


def distribution_per_row(row, classes):
    row = row[~np.isnan(row)]
    distribution = np.zeros(classes)
    for entry in row:
        entry = int(entry)
        distribution[entry - 1] += 1 / len(row)
    return distribution


def transform_to_distribution(data, classes):
    labels = [distribution_per_row(x, classes).reshape(1, -1) for x in data]
    # labels = data.apply(lambda x: distribution_per_row(x, classes), axis=1).tolist()
    # labels = [x.reshape(1, -1) for x in labels]
    return np.concatenate(labels)


def cross_entropy(array_one, array_two):
    eps = 1e-8
    entropy = 0
    for x, y in zip(array_one, array_two):
        for entry_x, entry_y in zip(x, y):
            entropy -= entry_x * math.log(entry_y + eps)
    return entropy / len(array_one)


class Softlabel_crossentropy():
    def __init__(self, classes, iterations, learning_rate=0.01, regularization=1, thres=1e-3,
                 print_each_x_iterations=100):
        """
        :param classes: number of classes in the distribution
        :param iterations: number of epochs for sgd
        :param learning_rate: learning rate of sgd
        :param regularization: L2 regularization coefficent
        :param thres: threshold to stop in SGD if L1 norm of the gradient is lower than that, the fit stops
        :param print_each_x_iterations: prints cross entropy each x iterations, if -1 does not print
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.thres = thres
        self.classes = classes
        self.params = None
        self.tf_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),
                                             strip_accents='unicode',
                                             stop_words='english',
                                             lowercase=True)
        self.print_each_x_iterations = print_each_x_iterations

    def _softmax(self, logits):
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def fit_text(self, X):
        self.tf_vectorizer.fit(X)

    def transform_text(self, X):
        X = np.array(self.tf_vectorizer.transform(X).todense())
        X = np.c_[np.ones((len(X), 1)), X]
        return X

    def transform_labels(self, y):
        return transform_to_distribution(y, classes=self.classes)

    def fit(self, X, y):
        self.fit_text(X)
        X = self.transform_text(X)
        self.params = np.random.randn(self.classes, X.shape[1])
        y = self.transform_labels(y)
        for iter in tqdm(range(self.iterations)):
            Z = -X @ self.params.T
            prob_y = self._softmax(Z)
            error = y - prob_y
            dW = 1 / X.shape[0] * (error.T @ X) + 2 * self.regularization * self.params
            self.params -= self.learning_rate * dW
            if np.abs(dW).max() < self.thres: break
            if self.print_each_x_iterations != -1 and iter % self.print_each_x_iterations == 0:
                print(cross_entropy(y, prob_y))

    def predict(self, X):
        new_X = self.transform_text(X)
        Z = -new_X @ self.params.T
        prob_y = self._softmax(Z)
        return prob_y

    def calculate_cross_entropy(self, real, pred):
        real = self.transform_labels(real)
        return cross_entropy(real, pred)


class Train_distribution():
    def __init__(self, classes=5):
        self.train_distribution = None
        self.classes = classes

    def transform_labels(self, y):
        return transform_to_distribution(y, classes=self.classes)

    def fit(self, X, y):
        unique_elements, element_counts = np.unique(y, return_counts=True)
        dis = element_counts[:-1] / sum(element_counts[:-1])
        self.train_distribution = dis

    def predict(self, X):
        predictions = [self.train_distribution for i in range(len(X))]
        return np.stack(predictions)

    def calculate_cross_entropy(self, labels, predictions):
        labels = self.transform_labels(labels)
        return cross_entropy(labels, predictions)


class Dominating_class():
    def __init__(self, classes, **kwargs):
        self.full_class = np.zeros(classes)
        self.classes = classes

    def transform_labels(self, y):
        return transform_to_distribution(y, classes=self.classes)

    def fit(self, X, y):
        dominant = None
        dominant_count = 0
        unique_elements, element_counts = np.unique(y, return_counts=True)
        for label, count in zip(unique_elements, element_counts):
            if isNaN(label):
                continue
            if dominant_count < count:
                dominant_count = count
                dominant = label
        dominant = int(dominant)
        self.full_class[dominant - 1] = dominant

    def predict(self, X):
        predictions = [self.full_class for i in range(len(X))]
        return np.stack(predictions)

    def calculate_cross_entropy(self, labels, predictions):
        labels = self.transform_labels(labels)
        return cross_entropy(labels, predictions)

def preprocess(data):
    data = data.replace('None', np.NAN)
    text = data['text'].to_numpy()
    annotators_names = list(data.columns)
    for col in ['example_id', 'batch', 'text']:
        annotators_names.remove(col)
    labels = data[annotators_names]
    labels = labels.dropna(thresh=2)
    text = text[labels.index]
    return text,labels.to_numpy().astype(float)
def main_for_trails(trial):
    #full_data, annotators_names = create_data('data/labeled/full_annotation_team_1')
    full_data = pd.read_csv('data/full_data.csv')
    text,labels = preprocess(full_data)
    classes = 5
    random_state = 42
    iterations = 2000
    lr = trial.suggest_float('lr',1e-5,1e-2,log=True)
    print_each_x_iterations = -1
    regularization = trial.suggest_float('regularization',0.1,5)
    thres = 1e-3
    folds = 5
    splitter = KFold(n_splits=folds,shuffle=True,random_state=random_state)
    basic_model_1_results = []
    basic_model_2_results = []
    basic_model_3_results = []
    for train_indexes,test_indexes in splitter.split(text):
        X_train = text[train_indexes]
        X_test = text[test_indexes]
        y_train = labels[train_indexes]
        y_test = labels[test_indexes]
        basic_model = Dominating_class(classes=classes)
        basic_model.fit(X_train, y_train)
        predictions = basic_model.predict(X_test)
        basic_model_1_results.append(basic_model.calculate_cross_entropy(y_test, predictions))
        basic_model_2 = Train_distribution(classes)
        basic_model_2.fit(X_train,y_train)
        predictions = basic_model_2.predict(X_test)
        basic_model_2_results.append(basic_model_2.calculate_cross_entropy(y_test,predictions))
        basic_model_3 = Softlabel_crossentropy(classes=classes, iterations=iterations, learning_rate=lr,
                                               regularization=regularization, thres=thres,
                                               print_each_x_iterations=print_each_x_iterations)
        basic_model_3.fit(X_train, y_train)
        y_pred = basic_model_3.predict(X_test)
        basic_model_3_results.append(basic_model_3.calculate_cross_entropy(y_test, y_pred))
    print(np.mean(basic_model_1_results))
    print(np.mean(basic_model_2_results))
    print(np.mean(basic_model_3_results))
    return np.mean(basic_model_3_results)
def main():
    #full_data, annotators_names = create_data('data/labeled/full_annotation_team_1')
    full_data = pd.read_csv('data/full_data.csv')
    text,labels = preprocess(full_data)
    classes = 5
    random_state = 42
    iterations = 2000
    lr = 5e-4
    print_each_x_iterations = -1
    regularization = 1
    thres = 1e-3
    folds = 5
    splitter = KFold(n_splits=folds,shuffle=True,random_state=random_state)
    basic_model_1_results_train = []
    basic_model_1_results_test = []
    basic_model_2_results_train = []
    basic_model_2_results_test = []
    basic_model_3_results_train = []
    basic_model_3_results_test = []
    for train_indexes,test_indexes in splitter.split(text):
        X_train = text[train_indexes]
        X_test = text[test_indexes]
        y_train = labels[train_indexes]
        y_test = labels[test_indexes]
        basic_model = Dominating_class(classes=classes)
        basic_model.fit(X_train, y_train)
        predictions = basic_model.predict(X_train)
        basic_model_1_results_train.append(basic_model.calculate_cross_entropy(y_train, predictions))
        predictions = basic_model.predict(X_test)
        basic_model_1_results_test.append(basic_model.calculate_cross_entropy(y_test, predictions))
        basic_model_2 = Train_distribution(classes)
        basic_model_2.fit(X_train,y_train)
        predictions = basic_model_2.predict(X_train)
        basic_model_2_results_train.append(basic_model_2.calculate_cross_entropy(y_train, predictions))
        predictions = basic_model_2.predict(X_test)
        basic_model_2_results_test.append(basic_model_2.calculate_cross_entropy(y_test,predictions))
        basic_model_3 = Softlabel_crossentropy(classes=classes, iterations=iterations, learning_rate=lr,
                                               regularization=regularization, thres=thres,
                                               print_each_x_iterations=print_each_x_iterations)
        basic_model_3.fit(X_train, y_train)
        predictions = basic_model_3.predict(X_train)
        basic_model_3_results_train.append(basic_model_3.calculate_cross_entropy(y_train,predictions))
        y_pred = basic_model_3.predict(X_test)
        basic_model_3_results_test.append(basic_model_3.calculate_cross_entropy(y_test, y_pred))
    print(np.mean(basic_model_1_results_train))
    print(np.mean(basic_model_1_results_test))
    print(np.mean(basic_model_2_results_train))
    print(np.mean(basic_model_2_results_test))
    print(np.mean(basic_model_3_results_train))
    print(np.mean(basic_model_3_results_test))
    #return np.mean(basic_model_3_results)

if __name__ == '__main__':
    main()
    # study = optuna.create_study(direction='minimize')
    # study.optimize(main_for_trails, n_trials=100)
    #main()
