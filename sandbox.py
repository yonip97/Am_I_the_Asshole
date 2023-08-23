import numpy as np
import pandas as pd
import ast
from utils import distribution_per_row_series
import json
import string
from utils import cross_entropy


def remove_punctuation(text):
    for punctuation in ['%', '.']:
        text = text.replace(punctuation, '')
    return text


def check_if_valid(y):
    sum_ = 0
    for k, v in y.items():
        sum_ += v
    if sum_ != 100:
        return False
    return True


def process_row(original_row_data):
    original_row_data = ast.literal_eval(original_row_data)
    row_data = []
    for sample in original_row_data:
        try:
            y = ast.literal_eval(remove_punctuation(sample))
            if not check_if_valid(y):
                continue
            row_data.append(y)
        except:
            pass
    averaged_row_data = np.zeros(5)
    for sample_dict in row_data:
        for k, v in sample_dict.items():
            k = int(k)
            averaged_row_data[k - 1] += v / 100 / len(row_data)
    return averaged_row_data



