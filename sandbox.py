from os import listdir
from os.path import isfile, join
import pandas as pd
from IAA_statistics import merge_results
import numpy as np
import pickle

# mypath = 'data/labeled/25-49'
# onlyfiles = [join(mypath, f) for f in listdir(mypath) if (isfile(join(mypath, f)) and 'csv' in f)]
# files_series = [pd.read_csv(f, index_col=0).reset_index(drop=True) for f in onlyfiles]
# x = merge_results(files_series)
# x = x.replace('None', np.NaN)
# x = x.astype(float)
# x.to_excel('data/labeled/25-49/annotation_25-49.xlsx')
# print(sum(x == 1))
# max_diff = 0
# for i in range(len(x.columns)):
#     for j in range(i + 1, len(x.columns)):
#         y = x[[i, j]].set_axis([0, 1], axis=1, inplace=False)
#         y = y.dropna().astype('int')
#         max_diff = max(max(y[0] - y[1]), max_diff)
# print(max_diff)
with open('data/raw/raw_output_updated.pickle','rb') as f:
    raw_data = pickle.load(f)
records = []
for post in raw_data:
    post = post['data']
    records.append((post['title'], post['selftext'], post['score'], post['upvote_ratio'], post['num_comments']))
df = pd.DataFrame.from_records(records, columns=['title', 'post_text', 'score', 'upvote_ratio', 'num_comments'])
df = df[['post_text']]
df.to_excel('data/full_data.xlsx')
full_data_for_annotation = df[25:350].reset_index()
full_data_for_annotation['batch'] = 'exploration'
full_data_for_annotation['batch'][50:125] = 'evaluation'
full_data_for_annotation['batch'][125:325] = 'part3'
full_data_for_annotation['text'] = full_data_for_annotation['post_text']
full_data_for_annotation['original_text'] = full_data_for_annotation['post_text']
full_data_for_annotation['label_1'] = np.NAN
full_data_for_annotation['label_2'] = np.NAN
full_data_for_annotation['label_3'] = np.NAN
full_data_for_annotation['label_4'] = np.NAN
full_data_for_annotation['example_id'] = full_data_for_annotation['index']
full_data_for_annotation = full_data_for_annotation.drop(['index','post_text'],axis=1)
full_data_for_annotation = full_data_for_annotation[['example_id','batch','original_text','text','label_1','label_2','label_3','label_4']]
full_data_for_annotation['label'] = np.nan
full_data_for_annotation['annotator_id'] = np.nan
full_data_for_annotation = full_data_for_annotation[['example_id','annotator_id','text','label']]
full_data_for_annotation[125:155].to_excel('data/Part3A.xlsx')
full_data_for_annotation[155:325].to_excel('data/Part3B.xlsx')

full_data_for_annotation = full_data_for_annotation[:125]
full_data_for_annotation_gony = full_data_for_annotation.copy()
full_data_for_annotation_gony['annotator_id'] = 'Gony Idan'
full_data_for_annotation_ofek = full_data_for_annotation.copy()
full_data_for_annotation_ofek['annotator_id'] = 'Ofek Glick'
full_data_for_annotation_eitan = full_data_for_annotation.copy()
full_data_for_annotation_eitan['annotator_id'] = 'Eitan Greenberg'
full_data_for_annotation_yoni = full_data_for_annotation.copy()
full_data_for_annotation_yoni['annotator_id'] = 'Yehonatan Peisakhovsky'
yoni_done = pd.read_excel('data/labeled/exploration/50-74/exploration_data_50-74_yoni.xlsx',index_col=0)
ofek_done = pd.read_excel('data/labeled/exploration/50-74/exploration_data_50-74_ofek.xlsx',index_col=0)
gony_done =pd.read_csv('data/labeled/evaluation/evaluation_data_gony_100-150.csv',index_col=0)
eitan_done = pd.read_excel('data/labeled/evaluation/eitan_100-150.xlsx',index_col=0)
eitan_done = eitan_done.replace('None',np.nan)
eitan_done = eitan_done.astype(float)
full_data_for_annotation_yoni['label'][25:50] = yoni_done['label']
full_data_for_annotation_ofek['label'][25:50] = ofek_done['label']
full_data_for_annotation_gony['label'][75:125] = gony_done['label'].loc[100:149]
full_data_for_annotation_eitan['label'][75:125] = eitan_done['label']
full_data_for_annotation_eitan = full_data_for_annotation_eitan.drop(range(44,75)).reset_index(drop = True)
full_data_for_annotation_gony = full_data_for_annotation_gony.drop(list(range(13,44))).reset_index(drop = True)
full_data_for_annotation_ofek = full_data_for_annotation_ofek[13:106].reset_index(drop = True)
full_data_for_annotation_yoni = full_data_for_annotation_yoni.drop(range(75,106)).reset_index(drop = True)
c = 6
full_data_for_annotation_gony.to_excel('data/gony_part2.xlsx')
full_data_for_annotation_eitan.to_excel('data/eitan_part2.xlsx')
full_data_for_annotation_yoni.to_excel('data/yoni_part2.xlsx')
full_data_for_annotation_ofek.to_excel('data/ofek_part2.xlsx')
# group1 = df[100:235][['post_text']]
# group2 = df[235:370][['post_text']]
# group1.to_excel('data/to_be_labeled/records_100-234.xlsx')
# group2.to_excel('data/to_be_labeled/records_234-369.xlsx')
# import os
# mypath = 'data/labeled'
# names = {'yoni':[],'gony':[],'eitan':[],'ofek':[]}
# for path, subdirs, files in os.walk(mypath):
#     for name in files:
#         name = os.path.join(path, name)
#         x = pd.read_excel(name)
#         for key in names:
#             if key in name:
#                 names[key].append(name)
# for key in names:
#     all_annotations = []
#     cols = []
#     for name in names[key]:
#         x = pd.read_excel(name,index_col=0).reset_index(drop=True)
#         all_annotations.append(x)
#         if '0-24' in name:
#             cols.append('0-24')
#         elif '25-49' in name:
#             cols.append('25-49')
#         else:
#             cols.append('50-74')
#     full_annotation_by_annotator = pd.concat(all_annotations,axis=1).astype(float)
#     full_annotation_by_annotator.columns = cols
#     full_annotation_by_annotator.to_excel(f'data/labeled/all_exploration_annotations_{key}.xlsx')
