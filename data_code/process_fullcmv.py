import json
import re
import numpy as np
import pandas as pd

'''
    Process original dataset as a text file and convert to a dict (format is not right to directly load as dictionary).
'''
# dataset_path = 'datasets/original/CMV/'

# conversation_file = dataset_path + 'conversations.json'
# corpus_file = dataset_path + 'corpus.json'
# index_file = dataset_path + 'index.json'
# user_file = dataset_path + 'users.json'
# utterance_file = dataset_path + 'utterances.jsonl'

# with open(utterance_file, 'r') as infile:
#     utterances = infile.read()

# # Use the timestamp as a delimiter between utterances. 
# regex_for_split = re.compile(r'\"timestamp\":\s([0-9]*|null)\}')
# utterances = re.split(regex_for_split, utterances)
# utterances = utterances[0::2]
# utterances = [u[::-1].replace(',', '}', 1)[::-1] for u in utterances][:len(utterances)-1]

# #Convert to dict.
# utterances = [json.loads(u) for u in utterances]

# with open('datasets/processed/CMV/cmv_usable.jsonl', 'w') as outfile:
#     json.dump(utterances, outfile, indent = 4)

#Loading from the file for now.
with open('../datasets/processed/CMV/cmv_usable.jsonl', 'r') as infile:
    utterances = json.load(infile)

print('Loaded utterances.')

'''
    Split into null, successful, and unsuccessful.
'''

successful = []
unsuccessful = []
neutral = []

for u in utterances:
    if u['id'] != u['root']: #Filtering out the original post. No information on its strength.
        if u['meta']['success'] == 1:
            successful.append({
                'id': u['id'],
                'text': u['text'],
                'root': u['root'],
                'user': u['user'],
                'ups': abs(u['meta']['ups']) if u['meta']['ups'] else 0, 
            })
        elif u['meta']['success'] == 0:
            unsuccessful.append({
                'id': u['id'],
                'text': u['text'],
                'root': u['root'],
                'user': u['user'],
                'ups': abs(u['meta']['ups']) if u['meta']['ups'] else 0,
            })
        else:
            neutral.append({
                'id': u['id'],
                'text': u['text'],
                'root': u['root'],
                'user': u['user'],
                'ups': abs(u['meta']['ups']) if u['meta']['ups'] else 0,
            })

print('Split by success.')
print()

'''
    Some upvote metrics.
'''

s_ups = [value for u in successful for key, value in u.items() if key == 'ups']
us_ups = [value for u in unsuccessful for key, value in u.items() if key == 'ups']

print('Upvote metrics:')
print(np.median(s_ups), np.median(us_ups))
print(np.max(s_ups), np.max(us_ups))
print()


'''
    Conversation-level metrics
'''

posts = [u['id'] for u in utterances if u['root'] == u['id']]

conversation_metrics = []
conversations = {post: {
    'successful':[],
    'unsuccessful':[],
    'neutral':[]
} for post in posts}

for u in successful:
    conversations[u['root']]['successful'].append(u)

for u in unsuccessful:
    conversations[u['root']]['unsuccessful'].append(u)

for u in neutral:
    conversations[u['root']]['neutral'].append(u)

for post in posts:
    s_commenters = [u['user'] for u in conversations[post]['successful']]
    us_commenters = [u['user'] for u in conversations[post]['unsuccessful']]
    commenters = s_commenters + us_commenters
    unique_commenters = list(set(commenters))
    comment_counts = {commenter : commenters.count(commenter) for commenter in unique_commenters}
    total_comments = sum([])

    s_replies = [u['id'] for u in conversations[post]['successful']]
    us_replies = [u['id'] for u in conversations[post]['unsuccessful']]
    n_replies = [u['id'] for u in conversations[post]['neutral']]
    s_upvotes = [u['ups'] for u in conversations[post]['successful']]
    us_upvotes = [u['ups'] for u in conversations[post]['unsuccessful']]
    
    unique_s_up_vals = list(set(s_upvotes))
    upvote_counts = {up_val : s_upvotes.count(up_val) for up_val in unique_s_up_vals}
    unique_us_up_vals = list(set(us_upvotes))
    us_upvote_counts = {up_val : us_upvotes.count(up_val) for up_val in unique_us_up_vals}
    for up_val in us_upvote_counts:
        if up_val not in upvote_counts:
            upvote_counts[up_val] = us_upvote_counts[up_val]
        else:
            upvote_counts[up_val] += us_upvote_counts[up_val]

    conv_info = {
        'id': post,
        'num_s_replies': len(s_replies),
        'num_us_replies': len(us_replies),
        'num_n_replies': len(n_replies),
        'total_s_upvotes': sum(s_upvotes),
        'avg_s_upvotes': np.mean(s_upvotes),
        'med_s_upvotes': np.median(s_upvotes),
        'std_s_upvotes': np.std(s_upvotes),
        'total_us_upvotes': sum(us_upvotes),
        'avg_us_upvotes': np.mean(us_upvotes),
        'med_us_upvotes': np.median(us_upvotes),
        'std_us_upvotes': np.std(us_upvotes),
        'comment_counts': comment_counts,
        'total_comments': sum([value for key,value in comment_counts.items()]),
        'upvote_counts': upvote_counts
    }
    conversation_metrics.append(conv_info)

print('Conversation metrics:')
metric_list = {
    'num_s_replies': [],
    'num_us_replies': [],
    'num_n_replies': [],
    'total_s_upvotes':[],
    'total_us_upvotes': []
}
for c in conversation_metrics:
    for metric in c:
        if metric in metric_list.keys():
            metric_list[metric].append(c[metric])

for metric, values in metric_list.items():
    print(metric, ": ", end = " ")
    print(np.mean(values), np.median(values))
print()


'''
    Write processed data.
'''

with open('../datasets/processed/CMV/successful.json', 'w') as outfile:
    json.dump(successful, outfile, indent = 4)

with open('../datasets/processed/CMV/unsuccessful.json', 'w') as outfile:
    json.dump(unsuccessful, outfile, indent = 4)

with open('../datasets/processed/CMV/neutral.json', 'w') as outfile:
    json.dump(neutral, outfile, indent = 4)

conversation_metrics = pd.DataFrame(conversation_metrics)
conversation_metrics.to_csv('../datasets/processed/CMV/conversation_metrics.csv', header=True, index = False)

print('Written to files.')



    

