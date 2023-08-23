import os
from functools import reduce
import pandas as pd
from scipy.stats import pearsonr, kendalltau
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
import math
from collections import Counter
from collections import Counter


def check_correlation(a, b):
    p = pearsonr(a, b)[0]
    k = kendalltau(a, b)[0]
    return p, k


def isNaN(num):
    return num != num


def merge_with_metadata(annotation_data_path, raw_data_path):
    annotation_data = pd.read_csv(annotation_data_path)
    annotation_data = preprocess(annotation_data, dropna_thres=4)
    relevant_attributes = {'upvote_ratio': [], 'score': [], 'num_comments': [], 'title': []}
    with open(raw_data_path, 'rb') as f:
        full_raw_data = pickle.load(f)
    full_raw_data = [x['data'] for x in full_raw_data]
    count = 0
    for k in full_raw_data[0].keys():
        attribute = []
        for i in range(len(full_raw_data)):
            if k in full_raw_data[i]:
                attribute.append(full_raw_data[i][k])
        try:
            meta_data = Counter(attribute)
            if meta_data.most_common(1)[0][1] > 0.9 * len(attribute):
                count += 1
        except:
            pass
    for sample in full_raw_data:
        for key in relevant_attributes:
            relevant_attributes[key].append(sample[key])
    meta_data = pd.DataFrame.from_dict(relevant_attributes)
    meta_data['example_id'] = meta_data.index
    full_data = annotation_data.merge(meta_data, on='example_id').reset_index(drop=True)
    full_data = full_data.rename({'score': 'ups'}, axis=1)
    full_data['total_votes'] = full_data['ups'] / full_data['upvote_ratio']
    full_data['total_votes'] = full_data['total_votes'].astype(int)
    full_data['downs'] = full_data['total_votes'] - full_data['ups']
    annotators_names = list(annotation_data.columns)
    annotators_names = [x for x in annotators_names if x not in ['text', 'example_id']]
    return full_data, annotators_names


def cross_entropy(pred, real):
    eps = 1e-8
    entropy = 0
    for x, y in zip(pred, real):
        for entry_x, entry_y in zip(x, y):
            entropy -= entry_x * math.log(entry_y + eps)
    return entropy / len(pred)


def preprocess(data, dropna_thres=4):
    data = data.replace('None', np.NAN)
    data = data.dropna(thresh=dropna_thres)
    return data


def distribution_per_row_series(row):
    row = row.dropna()
    distribution = np.zeros(5)
    for entry in row:
        distribution[int(entry) - 1] += 1 / len(row)
    return distribution

def distribution_per_row_numpy(row, classes):
    row = row[~np.isnan(row)]
    distribution = np.zeros(classes)
    for entry in row:
        entry = int(entry)
        distribution[entry - 1] += 1 / len(row)
    return distribution
def data_merging(dir_path):
    files = os.listdir(dir_path)
    pds = [pd.read_excel(os.path.join(dir_path, f), index_col=0) for f in files]
    full_data = pds[0]
    name = full_data['annotator_id'].iloc[0]
    full_data['label'] = full_data['label'].replace('None', np.NAN).astype(float)
    full_data = full_data.rename({"label": name}, axis=1).drop(['annotator_id'], axis=1)
    for index in range(1, len(pds)):
        pd_frame = pds[index]
        name = pd_frame['annotator_id'].iloc[0]
        pd_frame['label'] = pd_frame['label'].replace('None', np.NAN).astype(float)
        pd_frame = pd_frame.rename({"label": name}, axis=1).drop(['annotator_id'], axis=1)
        full_data = full_data.merge(pd_frame, on=['example_id', 'text'], how='outer')
    full_data = full_data.sort_values('example_id')
    return full_data
def split_to_train_val_test(seed, data, train_percentage=0.6, val_percentage=0.2):
    assert 1 - train_percentage - val_percentage > 0
    ids = data['example_id']
    train_size = int(train_percentage * len(ids))
    val_size = int(val_percentage * len(ids))
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(ids)
    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:train_size + val_size]
    test_ids = shuffled_ids[train_size + val_size:]
    train_data = data[data['example_id'].isin(train_ids)]
    val_data = data[data['example_id'].isin(val_ids)]
    test_data = data[data['example_id'].isin(test_ids)]
    return train_data, val_data, test_data
def preprocess_for_training(data):
    text = data['text'].to_numpy()
    annotators_names = list(data.columns)
    for col in ['example_id', 'text']:
        annotators_names.remove(col)
    labels = data[annotators_names]
    return text, labels.to_numpy().astype(float)
