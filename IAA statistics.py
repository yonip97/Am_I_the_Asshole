import pandas as pd
from statsmodels.stats import inter_rater as irr
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.stats import pearsonr


class IAA_metric():
    def calculate_observed_agreement(self, data: pd.DataFrame):
        count_agreement = 0
        count_options = 0
        for index, row in data.iterrows():
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    count_options += 1
                    if row[i] == row[j]:
                        count_agreement += 1
        return count_agreement / count_options

    def calculate_chance_agreement(self, data):
        return 0

    def calculate(self, data: pd.DataFrame):
        A_0 = self.calculate_observed_agreement(data)
        A_e = self.calculate_chance_agreement(data)
        return (A_0 - A_e) / (1 - A_e)


class Bennet_s(IAA_metric):
    def __init__(self, num_of_classes):
        self.num_of_classes = num_of_classes

    def calculate_chance_agreement(self, data):
        return 1 / self.num_of_classes


class Scott_pi(IAA_metric):

    def calculate_chance_agreement(self, data):
        categories_count = {}
        annotation_count = 0
        for index, row in data.iterrows():
            for i in range(len(row)):
                if row[i] not in categories_count.keys():
                    categories_count[row[i]] = 0
                categories_count[row[i]] += 1
                annotation_count += 1
        counter = 0
        for category in categories_count.keys():
            counter += categories_count[category] ** 2
        return counter / annotation_count ** 2


class Cohen_kappa(IAA_metric):
    def calculate_chance_agreement(self, data):
        categories_per_annotator_count = {}
        annotation_count = 0
        for index, row in data.iterrows():
            for i in range(len(row)):
                if row[i] not in categories_per_annotator_count.keys():
                    categories_per_annotator_count[row[i]] = {0: 0, 1: 0}
                categories_per_annotator_count[row[i]][i] += 1
                annotation_count += 1
        counter = 0
        for category in categories_per_annotator_count.keys():
            counter += categories_per_annotator_count[category][0] * categories_per_annotator_count[category][1]
        return counter / annotation_count ** 2


class Feliss_kappa():
    def __init__(self, num_of_classes):
        self.num_pf_classes = num_of_classes

    def calculate(self, data: pd.DataFrame):
        data -= 1
        agg_data, _ = irr.aggregate_raters(data.T, self.num_pf_classes)
        return irr.fleiss_kappa(agg_data)


def merge_results(annotators_series):
    annotators_series_new = []
    for i in range(len(annotators_series)):
        annotators_series_new.append(annotators_series[i].rename(columns={'label':i}))
    return pd.concat(annotators_series_new,axis=1)
#
# a = pd.Series(data=[1,1,1,1,1,2,3,4,3,2,1,2],index=[0,1,2,3,4,5,6,7,8,9,10,11])
# b = pd.Series(data=[4,1,2,1,1,2,6,4,3,2,1,3],index=[0,1,2,3,4,5,6,7,8,9,10,11])
# c = pd.Series(data=[4,5,1,1,1,1,1,4,3,8,1,3],index=[0,1,2,3,4,5,6,7,8,9,10,11])
# x = merge_results([a,b,c])
mypath = 'data'
onlyfiles = [join(mypath, f) for f in listdir(mypath) if (isfile(join(mypath, f)) and 'csv' in f)]
files_series = [pd.read_csv(f,index_col=0).reset_index(drop=True)for f in onlyfiles]
x = merge_results(files_series)
x = x.replace('None',np.NaN)
metric_bennet_s = Bennet_s(5)
metric_scott_pi = Scott_pi()
metric_cohen_kappa = Cohen_kappa()
metric_feliss_kappa = Feliss_kappa(5)
pairwise_agreement_bennet_s = []
pairwise_agreement_scott_pi = []
pairwise_agreement_cohen_kappa = []
for i in range(len(x.columns)):
    for j in range(i+1,len(x.columns)):
        y = x[[i,j]].set_axis([0,1],axis=1,inplace=False)
        y = y.dropna().astype('int')
        pairwise_agreement_bennet_s.append(metric_bennet_s.calculate(y))
        pairwise_agreement_scott_pi.append(metric_scott_pi.calculate(y))
        pairwise_agreement_cohen_kappa.append(metric_cohen_kappa.calculate(y))

for i in range(len(x.columns)):
    for j in range(i+1,len(x.columns)):
        y = x[[i,j]].set_axis([0,1],axis=1,inplace=False)
        y = y.dropna().astype('int')
        print(pearsonr(y[0],y[1]))
        print(onlyfiles[i])
        print(onlyfiles[j])
        print(metric_bennet_s.calculate(y))
        print(metric_scott_pi.calculate(y))
        print(metric_cohen_kappa.calculate(y))


x = x.dropna().astype('int')
feliss_kappa_result = metric_feliss_kappa.calculate(x.copy())

print(f"bennet s average: {np.mean(pairwise_agreement_bennet_s):.4f}")
print(f"scott pi average {np.mean(pairwise_agreement_scott_pi):.4f}")
print(f"cohen kappa average {np.mean(pairwise_agreement_cohen_kappa):.4f}")
print(f"feliss kappa {feliss_kappa_result:.4f}")

