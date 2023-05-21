import math

import numpy as np
import pandas as pd
import requests
import pickle


def extract_from_reddit():
    subreddit_name = "AmItheAsshole"
    url = f"https://www.reddit.com/r/{subreddit_name}/hot.json?limit=100"

    headers = {
        "User-Agent": "your_user_agent"
    }

    posts = []
    after = None
    while len(posts) < 2000 and (after is not None or len(posts) == 0):
        if after is not None:
            url = f"https://www.reddit.com/r/{subreddit_name}/new.json?limit=100&after={after}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()["data"]
            posts += data["children"]
            after = data["after"]
            print(f"Retrieved {len(posts)} posts from r/{subreddit_name}")
        else:
            print(f"Error retrieving posts: {response.status_code}")
            break
    raw_posts = posts[1:]
    posts_users = set()
    posts = []
    for post in raw_posts:
        if post['data']['author'] in posts_users:
            continue
        posts.append(post)
        posts_users.add(post['data']['author'])
    records = []
    with open('data/raw/raw_output.pickle', 'wb') as handle:
        pickle.dump(posts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for post in posts:
        post = post['data']
        records.append((post['title'], post['selftext'], post['score'], post['upvote_ratio'], post['num_comments']))
    df = pd.DataFrame.from_records(records, columns=['title', 'post_text', 'score', 'upvote_ratio', 'num_comments'])
    already_labeled = pd.read_csv('data/AITA_exploration.csv', index_col=0)[:25]
    already_labeled[['title', 'score', 'upvote_ratio', 'num_comments']] = np.NAN
    already_labeled.to_csv("data/unlabeled/old_posts_already_labeled.csv",encoding="utf-8-sig")
    df = pd.concat((already_labeled, df))[['title', 'post_text', 'score', 'upvote_ratio', 'num_comments']]
    df.to_csv('data/unlabeled/full_data.csv', encoding='utf-8-sig')

    chosen_posts = df[:475]
    exploration_posts = chosen_posts[:100]
    exploration_posts.to_csv('data/unlabeled/exploration_data.csv', encoding="utf-8-sig")
    evaluation_posts = chosen_posts[100:200]
    evaluation_posts.to_csv('data/unlabeled/evaluation_data.csv', encoding="utf-8-sig")
    rest = chosen_posts[200:]
    rest.to_csv('data/unlabeled/remaining_data.csv', encoding="utf-8-sig")


def divide_to_chunks(path, num):
    df = pd.read_csv(path,index_col=0)
    chunks = math.ceil(len(df) / num)
    for i in range(chunks):
        temp = df[i * num:(i + 1) * num]
        temp.to_csv(f'data/unlabeled/exploration_data_{i * num}-{(i + 1) * num - 1}.csv',encoding="utf-8-sig")

def read_raw_data(path):
    with open(path,'rb') as f:
        x = pickle.load(f)
    return x
def give_time(samples):
    import datetime
    for sample in samples:
        time_ = datetime.datetime.fromtimestamp(sample['data']['created_utc']).strftime('%Y-%m-%d')
        print(time_)

#extract_from_reddit()
#divide_to_chunks('data/unlabeled/exploration_data.csv', 25)
# x = read_raw_data('data/raw/raw_output.pickle')
# give_time(x)
# print(len(x))
# c = 5
# x = pd.read_csv(f"data/unlabeled/full_data.csv",index_col=0)
# c = 5
# file_50_to_74 = x[50:75][['post_text']]
# file_50_to_74.to_excel("data/to_be_labeled/exploration_data_50-74.xlsx")
# file_75_to_99 = x[75:100][['post_text']]
# file_75_to_99.to_excel("data/to_be_labeled/exploration_data_75-99.xlsx")
# file_100_to_370 = x[100:370][['post_text']]
# file_100_to_370.to_excel("data/to_be_labeled/data_100-369.xlsx")