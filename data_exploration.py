import string
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from utils import merge_annotation_results, check_correlation
import json
import pickle
import math
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


def extract_relevant_attributes():
    attributes = []
    return


def isNaN(num):
    return num != num


def temp(path):
    onlyfiles = [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and 'xlsx' in f)]
    files_series = [pd.read_excel(f, index_col=0).reset_index(drop=True) for f in onlyfiles]
    annotation_data = merge_annotation_results(files_series)
    annotation_data = annotation_data.replace('None', np.NAN)
    annotation_data = annotation_data.dropna(thresh=3)
    annotators_names = list(annotation_data.columns)
    annotation_data = annotation_data.astype(float)
    relevant_attributes = {'upvote_ratio': [], 'score': [], 'num_comments': [], 'selftext': [], 'title': []}
    with open("data/raw/raw_output_updated.pickle", 'rb') as f:
        full_raw_data = pickle.load(f)
    full_raw_data = [x['data'] for x in full_raw_data]
    count = 0
    for k in full_raw_data[0].keys():
        attriubte = []
        for i in range(len(full_raw_data)):
            if k in full_raw_data[i]:
                attriubte.append(full_raw_data[i][k])
        try:
            x = Counter(attriubte)
            if x.most_common(1)[0][1] > 0.9 * len(attriubte):
                count += 1
            else:
                print(k)
            # else:
            #     print(k)
        except:
            pass
            # print("Cant hash see later")
            # print(k)
    print(count)
    for sample in full_raw_data:
        for key in relevant_attributes:
            relevant_attributes[key].append(sample[key])
    x = pd.DataFrame.from_dict(relevant_attributes)
    x['example_id'] = x.index
    full_data = annotation_data.join(x, on='example_id').reset_index(drop=True)
    full_data = full_data.rename({'score': 'ups'}, axis=1)
    full_data['total_votes'] = full_data['ups'] / full_data['upvote_ratio']
    full_data['total_votes'] = full_data['total_votes'].astype(int)
    full_data['downs'] = full_data['total_votes'] - full_data['ups']
    # full_data = annotation_data.join(full_data_no_annotation,on='example_id')
    return full_data, annotators_names


def words_importance(col, data):
    lemmatizer = WordNetLemmatizer()
    sentences = data[col]
    importance = data['mean_score']
    words_importance_dict = {}
    for sentence, score in zip(sentences, importance):
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
        words = sentence.split()
        for word in words:

            if word in stopwords.words('english'):
                continue
            word = lemmatizer.lemmatize(word)
            if word not in words_importance_dict:
                words_importance_dict[word] = []
            words_importance_dict[word].append(score)
    words_appearance_dict = {k: len(v) for k, v in words_importance_dict.items()}
    words_importance_list = [(k, np.mean(v)) for k, v in words_importance_dict.items() if words_appearance_dict[k] > 10]
    words_importance_list = sorted(words_importance_list, reverse=True, key=lambda x: x[1])
    words_appearance_list = [(k, len(v)) for k, v in words_importance_dict.items()]
    words_appearance_list = sorted(words_appearance_list, reverse=True, key=lambda x: x[1])
    c = 4


def mean_score_distribution(full_data):
    plt.hist(full_data['mean_score'], bins=10, range=(1, 4), weights=np.ones(len(full_data)) / len(full_data))
    plt.title('mean score distribution')
    plt.xlabel('Post mean score')
    plt.show()


def single_annotation_histogram(full_data,annotators_names):
    all_annotations = []
    for name in annotators_names:
        all_annotations.append(full_data[name])
    all_annotations = pd.concat(all_annotations)
    bars = (all_annotations.value_counts() / len(all_annotations)).plot.bar(rot=0)
    colors = ['b', 'r', 'g', 'y']
    for patch, color in zip(bars.patches, colors):
        patch.set_facecolor(color)
    plt.show()


def pair_difference_histogram(full_data, annotators_names):
    all_diffs = []
    for i in range(len(annotators_names)):
        for j in range(i + 1, len(annotators_names)):
            temp_df = full_data[[annotators_names[i], annotators_names[j]]].dropna()
            diff = abs(temp_df[annotators_names[i]] - temp_df[annotators_names[j]])
            all_diffs.append(diff)
    all_diffs = pd.concat(all_diffs)
    bars = (all_diffs.value_counts() / len(all_diffs)).plot.bar(rot=0)
    colors = ['b', 'r', 'g', 'y']
    for patch, color in zip(bars.patches, colors):
        patch.set_facecolor(color)
    plt.show()


def check_correlation_each_pair(data):
    columns = data.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            a = data[columns[i]]
            b = data[columns[j]]
            print(f"The correlation between {columns[i]} and {columns[j]}")
            p, k = check_correlation(a, b)
            print(f"pearson correlation {p:.4f}")
            print(f"kendell tau correlation {k:.4f}")

def diff_each_pair(row):
    row = row.dropna()
    names = row.index
    diff = []
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            diff.append(abs(row[names[i]]-row[names[j]]))
    return np.mean(diff)

def main():
    full_data, annotators_names = temp('data/labeled/full_annotation_team_1')
    full_data['mean_score'] = full_data[annotators_names].mean(skipna=True, axis=1)
    full_data['text_length'] = full_data['selftext'].apply(lambda x: len(x))
    full_data['mean_disagreement'] = full_data[annotators_names].apply(diff_each_pair,axis=1)
    check_correlation_each_pair(
        full_data[['mean_score', 'text_length', 'num_comments', 'upvote_ratio', 'ups', 'total_votes', 'downs','mean_disagreement']])
    # print(f"Expected value {np.mean(full_data['mean_score'])}")
    # print(f"Variance of mean score {np.var(full_data['mean_score'])}")
    # mean_score_distribution(full_data)
    # single_annotation_histogram(full_data)
    # pair_difference_histogram(full_data,annotators_names)
    # full_data.to_excel('full_data_all_features.xlsx')


if __name__ == "__main__":
    main()
