from os import listdir
from os.path import isfile, join
import pandas as pd
from IAA_statistics import merge_results
import numpy as np

mypath = 'data/labeled/0-24'
onlyfiles = [join(mypath, f) for f in listdir(mypath) if (isfile(join(mypath, f)) and 'csv' in f)]
files_series = [pd.read_csv(f, index_col=0).reset_index(drop=True) for f in onlyfiles]
x = merge_results(files_series)
x = x.replace('None', np.NaN)
print(sum(x == 1))
max_diff = 0
for i in range(len(x.columns)):
    for j in range(i + 1, len(x.columns)):
        y = x[[i, j]].set_axis([0, 1], axis=1, inplace=False)
        y = y.dropna().astype('int')
        max_diff = max(max(y[0] - y[1]), max_diff)
print(max_diff)
