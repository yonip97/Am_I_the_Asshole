from functools import reduce
import pandas as pd



def merge_annotation_results(annotators_data):
    annotators_data_new = []
    if 'example_id' not in annotators_data[0].columns:
        for i in range(len(annotators_data)):
            annotators_data[i]['example_id'] = annotators_data[i].index
            annotators_data[i]['annotator_id'] = i
    for i in range(len(annotators_data)):
        annotators_data[i].index = annotators_data[i]['example_id']
        annotator = annotators_data[i]['annotator_id'].iloc[0]
        annotators_data[i] = annotators_data[i].rename(columns={'label':annotator})
        annotators_data[i] = annotators_data[i][[annotator]]
        annotators_data_new.append(annotators_data[i] )
    df_merged = reduce(lambda left, right: pd.merge(left, right,left_index=True, right_index=True,how='outer'), annotators_data_new)
    return df_merged

