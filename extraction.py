import math

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
    with open('data/raw/raw_output.pickle', 'wb') as handle:
        pickle.dump(posts, handle, protocol=pickle.HIGHEST_PROTOCOL)

