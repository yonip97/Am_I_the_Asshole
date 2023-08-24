import ast
import numpy as np
import pandas as pd
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


def process_row_llms(original_row_data):
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


def main():
    data = pd.read_csv('data/prompted_data.csv',index_col=0)
    data['chatgpt_labels'] = data['gpt-3.5-turbo'].apply(lambda x: process_row_llms(x))
    data['gpt_4_labels'] = data['gpt-4'].apply(lambda x: process_row_llms(x))
    data['labels'] = data['labels'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
    print(f"The cross entropy of chatgpt labels is {cross_entropy(data['labels'], data['chatgpt_labels']):.3f}")
    print(f"The cross entropy of gpt4 labels is {cross_entropy(data['labels'], data['gpt_4_labels']):.3f}")

main()