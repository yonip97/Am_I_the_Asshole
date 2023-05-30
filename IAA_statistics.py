import pandas as pd
from statsmodels.stats import inter_rater as irr
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.stats import pearsonr,kendalltau
from functools import reduce

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
        data = data.copy()
        data -= 1
        agg_data, _ = irr.aggregate_raters(data.T, self.num_pf_classes)
        return irr.fleiss_kappa(agg_data)


def merge_results(annotators_data):
    annotators_data_new = []
    for i in range(len(annotators_data)):
        annotators_data[i].index = annotators_data[i]['example_id']
        annotator = annotators_data[i]['annotator_id'].iloc[0]
        annotators_data[i] = annotators_data[i].rename(columns={'label':annotator})
        annotators_data[i] = annotators_data[i][[annotator]]
        annotators_data_new.append(annotators_data[i] )
    df_merged = reduce(lambda left, right: pd.merge(left, right,left_index=True, right_index=True,how='outer'), annotators_data_new)
    return df_merged
def get_stats(path,classes):
    onlyfiles = [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and 'xlsx' in f)]
    files_series = [pd.read_excel(f,index_col=0).reset_index(drop=True)for f in onlyfiles]
    full_data = merge_results(files_series)
    full_data = full_data.replace('None',np.NAN)
    metric_bennet_s = Bennet_s(classes)
    metric_scott_pi = Scott_pi()
    metric_cohen_kappa = Cohen_kappa()
    metric_feliss_kappa = Feliss_kappa(classes)
    pairwise_agreement_bennet_s = []
    pairwise_agreement_scott_pi = []
    pairwise_agreement_cohen_kappa = []
    threewise_feliss_kappa = []
    pearson_correlations = []
    kendell_correlations = []
    individual_results = {}
    cols = list(full_data.columns)
    for i in range(len(full_data.columns)):
        for j in range(i+1,len(full_data.columns)):
            annotator_1 = cols[i]
            annotator_2 = cols[j]
            if (annotator_1,annotator_2) not in individual_results.keys():
                individual_results[(annotator_1,annotator_2)] = {}
            pair_data = full_data[[annotator_1,annotator_2]].set_axis([0,1],axis=1,copy=False)
            pair_data = pair_data.dropna().astype('int')
            bennet_s = round(metric_bennet_s.calculate(pair_data),4)
            scott_pi = round(metric_scott_pi.calculate(pair_data),4)
            cohen_kappa = round(metric_cohen_kappa.calculate(pair_data),4)
            pearson_correlation = round(pearsonr(pair_data[0], pair_data[1])[0],4)
            kendalltau_correlation = round(kendalltau(pair_data[0], pair_data[1])[0],4)
            individual_results[(annotator_1,annotator_2)]['bennet_S'] = bennet_s
            individual_results[(annotator_1, annotator_2)]['scott_pi'] = scott_pi
            individual_results[(annotator_1, annotator_2)]['cohen_kappa'] = cohen_kappa
            individual_results[(annotator_1, annotator_2)]['pearson_correlation'] = pearson_correlation
            individual_results[(annotator_1, annotator_2)]['kendalltau_correlation'] = kendalltau_correlation
            pairwise_agreement_bennet_s.append(bennet_s)
            pairwise_agreement_scott_pi.append(scott_pi)
            pairwise_agreement_cohen_kappa.append(cohen_kappa)
            pearson_correlations.append(pearson_correlation)
            kendell_correlations.append(kendalltau_correlation)
    results_dict = {}
    results_dict['mean_bennet_S'] = np.mean(pairwise_agreement_bennet_s)
    results_dict['mean_scott_pi'] = np.mean(pairwise_agreement_scott_pi)
    results_dict['mean_cohen_kappa'] = np.mean(pairwise_agreement_cohen_kappa)
    results_dict['mean_pearson_correlation'] = np.mean(pearson_correlations)
    results_dict['mean_kendell_tau_correlation'] = np.mean(kendell_correlations)
    for i in range(len(full_data.columns)):
        for j in range(i+1,len(full_data.columns)):
            for k in range(j+1,len(full_data.columns)):
                annotator_1 = cols[i]
                annotator_2 = cols[j]
                annotator_3 = cols[k]
                three_data = full_data[[annotator_1,annotator_2,annotator_3]].set_axis([0,1,2],axis=1,copy=False)
                three_data = three_data.dropna().astype('int')
                if len(three_data) == 0:
                    continue
                else:
                    if (annotator_1, annotator_2,annotator_3) not in individual_results.keys():
                        individual_results[(annotator_1, annotator_2,annotator_3)] = {}
                    feliss_kappa = round(metric_feliss_kappa.calculate(three_data),4)
                    individual_results[(annotator_1, annotator_2, annotator_3)]['feliss_kappa'] = feliss_kappa
                    threewise_feliss_kappa.append(feliss_kappa)
    results_dict['mean_felis_kappa'] = np.mean(threewise_feliss_kappa)
    return results_dict,individual_results

results_dict,individual_results = get_stats('data/labeled/full_annotation_team_1',5)
for key,value in individual_results.items():
    print(key)
    print(value)