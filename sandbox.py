import pandas as pd
import ast
from utils import distribution_per_row

data = pd.read_csv('data/prompted_data.csv',index_col=0)
y = pd.read_csv('data/prompted.csv',index_col=0)
gpt4_predictions = data['gpt-4'].tolist()
chatgpt_predictions = data['gpt-3.5-turbo'].tolist()
def process(data):
    new_data = []
    import json
    for x in data:
        x = ast.literal_eval(x)
        for y in x:
            y = json.loads(y)
            y = ast.literal_eval(y.replace('%',''))
            print(y)
        #new_x = ast.literal_eval(x[1:])
    c = 1

process(gpt4_predictions)



