#
# import pandas as pd
# from utils import preprocess, distribution_per_row
#
# def process_data(path):
#     data = pd.read_csv(path)
#     data = preprocess(data,dropna_thres=4)
#     ann_cols = [col for col in data.columns if col not in ['text', 'example_id']]
#     distributions = data[ann_cols].apply(lambda x: distribution_per_row(x), axis=1)
#     return data,distributions
#
# # guidelines_text = open('few_shot_prompt.txt', encoding='utf-8').read()
# data,distributions = process_data('data/full_data.csv')
# #data.to_csv('data/full_data_filtered.csv')
# data['distribution'] = distributions
# y = []
# x = pd.read_csv('data/prompted.csv')
# c = 4
# # for i, row in x.iterrows():
# #     post_text = row['text']
# #     dis = x['distribution']
# #     text = guidelines_text + '\n'
# #     text += 'post: ' + post_text + '\n' + 'score: '
# #     print(text)
# #     break
# #     print(len(post_text.split( )))
# #     y.append(len(post_text.split()))
# # plt.hist(y)
# # plt.show()
# # print(sum(np.array(y)>=600))
# # print(guidelines_text)
# text = """
# Task Guidelines:
# Generate a normalized distribution representing how much  the person who wrote the post is described by the task classes. The output format should be just: {1- ?% , 2-?%, 3-?%, 4-?%,5-?%}.
# Do not write anything beside the classes distribution"""
# print(len(text.split(' ')))
# text = """
# Guidelines for Behavior Levels:
#
# 1 - Morally Good: Considerate, empathetic, and respectful of others. Mindful of impact on others.
#
# 2 - Mildly Selfish: Mostly considerate, occasional insensitivity. Open to feedback and improvement.
#
# 3 - Occasional Rudeness: Sometimes inconsiderate, displays unkindness. Disregards others' feelings at times.
#
# 4 - Frequent Inappropriateness: Often insensitive, self-centered. Prioritizes own needs, lacks consideration.
#
# 5 - Deliberately Cruel: Empathy-free, self-important. Displays intentional malice and cruelty.
# """
# text = """
# Post:
#
# Task Guidelines:
# Generate a normalized distribution representing how much  the person who wrote the post is described by the task classes. The output format should be just: {1:?% , 2:?%, 3:?%, 4:?%,5:?%}.
# Do not write anything beside the classes distribution
#
#
#
# Task Classes:
# Guidelines for Behavior Levels:
#
# 1 - Morally Good: Considerate, empathetic, and respectful of others. Mindful of impact on others.
#
# 2 - Mildly Selfish: Mostly considerate, occasional insensitivity. Open to feedback and improvement.
#
# 3 - Occasional Rudeness: Sometimes inconsiderate, displays unkindness. Disregards others' feelings at times.
#
# 4 - Frequent Inappropriateness: Often insensitive, self-centered. Prioritizes own needs, lacks consideration.
#
# 5 - Deliberately Cruel: Empathy-free, self-important. Displays intentional malice and cruelty.
# """
# print(len(text.split(' ')))
#



