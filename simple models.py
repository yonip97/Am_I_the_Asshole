import math

import pandas as pd

from utils import merge_with_metadata, isNaN, split_to_train_val_test

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import optuna
from utils import preprocess,preprocess_for_training,distribution_per_row_numpy


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]





def transform_to_distribution(data, classes):
    labels = [distribution_per_row_numpy(x, classes).reshape(1, -1) for x in data]
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




def main_for_trails(trial, model_name):
    simple_models = {'Dominating_class': Dominating_class, 'Train_distribution': Train_distribution,
                     'Softlabel_crossentropy': Softlabel_crossentropy}

    full_data = pd.read_csv('data/full_data.csv', index_col=0)
    full_data = preprocess(full_data, dropna_thres=4)
    seed = 42
    train_data, val_data, _ = split_to_train_val_test(seed=seed, data=full_data)
    train_text, train_labels = preprocess_for_training(train_data)
    val_text, val_labels = preprocess_for_training(val_data)
    classes = 5
    iterations = 2000
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    print_each_x_iterations = -1
    regularization = trial.suggest_float('regularization', 0.1, 5)
    thres = trial.suggest_float('thres', 1e-4, 1e-2, log=True)
    model = Softlabel_crossentropy(classes=classes, iterations=iterations, learning_rate=lr,
                                      regularization=regularization, thres=thres,
                                      print_each_x_iterations=print_each_x_iterations)
    model.fit(train_text, train_labels)
    predictions = model.predict(val_text)
    return model.calculate_cross_entropy(val_labels, predictions)


def main():
    full_data = pd.read_csv('data/full_data.csv', index_col=0)
    full_data = preprocess(full_data, dropna_thres=4)
    seed = 42
    train_data, val_data, test_data = split_to_train_val_test(seed=seed, data=full_data)
    train_text, train_labels = preprocess_for_training(train_data)
    val_text, val_labels = preprocess_for_training(val_data)
    train_val_text = np.concatenate([train_text, val_text])
    train_val_labels = np.concatenate([train_labels, val_labels])
    test_text, test_labels = preprocess_for_training(test_data)
    classes = 5
    iterations = 2000
    lr = 5e-4
    print_each_x_iterations = -1
    regularization = 1
    thres = 1e-3
    basic_model = Dominating_class(classes=classes)
    basic_model.fit(train_val_text, train_val_labels)
    predictions = basic_model.predict(train_val_text)
    basic_model_1_results_train = basic_model.calculate_cross_entropy(train_val_labels, predictions)
    predictions = basic_model.predict(test_text)
    basic_model_1_results_test = basic_model.calculate_cross_entropy(test_labels, predictions)
    basic_model_2 = Train_distribution(classes)
    basic_model_2.fit(train_val_text, train_val_labels)
    predictions = basic_model_2.predict(train_val_text)
    basic_model_2_results_train = basic_model_2.calculate_cross_entropy(train_val_labels, predictions)
    predictions = basic_model_2.predict(test_text)
    basic_model_2_results_test = basic_model_2.calculate_cross_entropy(test_labels, predictions)
    basic_model_3 = Softlabel_crossentropy(classes=classes, iterations=iterations, learning_rate=lr,
                                           regularization=regularization, thres=thres,
                                           print_each_x_iterations=print_each_x_iterations)
    basic_model_3.fit(train_val_text, train_val_labels)
    predictions = basic_model_3.predict(train_val_text)
    basic_model_3_results_train = basic_model_3.calculate_cross_entropy(train_val_labels, predictions)
    predictions = basic_model_3.predict(test_text)
    basic_model_3_results_test = basic_model_3.calculate_cross_entropy(test_labels, predictions)
    print(f"The cross entropy for the train val set for the dominating class model is {basic_model_1_results_train}")
    print(f"The cross entropy for the test set for the dominating class model is {basic_model_1_results_test}")
    print(f"The cross entropy for the train val set for the train distribution model is {basic_model_2_results_train}")
    print(f"The cross entropy for the test set for the train distribution model is {basic_model_2_results_test}")
    print(f"The cross entropy for the train val set for the softlabel model is {basic_model_3_results_train}")
    print(f"The cross entropy for the test set for the softlabel model is {basic_model_3_results_test}")


if __name__ == '__main__':
    main()
    # study = optuna.create_study(direction='minimize')
    # study.optimize(main_for_trails, n_trials=100)
    # main()
