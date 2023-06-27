from functools import reduce
import pandas as pd
from scipy.stats import pearsonr,kendalltau
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle
from collections import Counter

def merge_annotation_results(annotators_data):
    # texts = [annotators_data[i][['example_id','text']] for i in range(len(annotators_data))]
    # texts = pd.concat(texts).drop_duplicates().sort_values('example_id').reset_index(drop = True)
    # texts = texts.drop(index=[96]).reset_index(drop =True)
    # texts.index = texts['example_id']
    # texts = texts.drop(columns=['example_id'])
    if 'example_id' not in annotators_data[0].columns:
        for i in range(len(annotators_data)):
            annotators_data[i]['example_id'] = annotators_data[i].index
            annotators_data[i]['annotator_id'] = i
    for i in range(len(annotators_data)):
        annotators_data[i].index = annotators_data[i]['example_id']
        annotator = annotators_data[i]['annotator_id'].iloc[0]
        annotators_data[i] = annotators_data[i].rename(columns={'label':annotator})
        annotators_data[i] = annotators_data[i][[annotator]]
    df_merged = reduce(lambda left, right: pd.merge(left, right,left_index=True, right_index=True,how='outer'), annotators_data)
    #df_merged = df_merged.join(texts)
    return df_merged


def check_correlation(a, b):
    p = pearsonr(a,b)[0]
    k = kendalltau(a,b)[0]
    return p,k
def isNaN(num):
    return num != num


def create_data(path):
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
        except:
            pass
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

