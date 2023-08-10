from utils import preprocess, distribution_per_row
import pandas as pd


def create_input(post_text, guidelines):
    input_text = 'Post: ' +'\n'
    input_text += post_text +'\n'
    input_text += guidelines
    return input_text


def create_prompted_data(path):
    data = pd.read_csv(path, index_col=0)
    data = preprocess(data,dropna_thres=4)
    guidelines = open('data/Task Guidelines.txt', encoding='utf-8').read()
    data['labels'] = data.drop(['example_id','text'],axis=1).apply(lambda x:distribution_per_row(x),axis=1)
    data = data[['example_id','text','labels']]
    data['prompted'] = data.apply(lambda x: create_input(x['text'], guidelines), axis=1)
    return data[['example_id','text','prompted','labels']]
