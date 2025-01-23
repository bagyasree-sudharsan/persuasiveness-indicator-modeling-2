import pandas as pd
import json
import numpy as np


### THIS SCRIPT DOES NOT RUN PROPERLY. THERE ARE ERRORS. 

'''
    Conversation-level metrics
'''

with open('../datasets/processed/AnnotatedCMV/successful_unlabeled.json', 'r') as infile:
    successful = json.load(infile)
with open('../datasets/processed/AnnotatedCMV/unsuccessful_unlabeled.json', 'r') as infile:
    unsuccessful = json.load(infile)
with open('../datasets/processed/AnnotatedCMV/neutral_unlabeled.json', 'r') as infile:
    neutral = json.load(infile)

posts = list(set([u['root'] for u in successful + unsuccessful + neutral]))
print(len(posts))

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

for post in conversations:
    print('POST: ', post, end = '-----')
    print(len(conversations[post]['successful']), len(conversations[post]['unsuccessful']), len(conversations[post]['neutral']))

# for post in posts:
#     s_commenters = [u['user'] for u in conversations[post]['successful']]
#     us_commenters = [u['user'] for u in conversations[post]['unsuccessful']]
#     commenters = s_commenters + us_commenters
#     unique_commenters = list(set(commenters))
#     comment_counts = {commenter : commenters.count(commenter) for commenter in unique_commenters}
#     total_comments = sum([])

#     s_replies = [u['id'] for u in conversations[post]['successful']]
#     us_replies = [u['id'] for u in conversations[post]['unsuccessful']]
#     n_replies = [u['id'] for u in conversations[post]['neutral']]
#     s_upvotes = [u['ups'] for u in conversations[post]['successful']]
#     us_upvotes = [u['ups'] for u in conversations[post]['unsuccessful']]

#     conv_info = {
#         'id': post,
#         'num_s_replies': len(s_replies),
#         'num_us_replies': len(us_replies),
#         'num_n_replies': len(n_replies),
#         'total_s_upvotes': sum(s_upvotes),
#         'avg_s_upvotes': np.mean(s_upvotes),
#         'med_s_upvotes': np.median(s_upvotes),
#         'std_s_upvotes': np.std(s_upvotes),
#         'total_us_upvotes': sum(us_upvotes),
#         'avg_us_upvotes': np.mean(us_upvotes),
#         'med_us_upvotes': np.median(us_upvotes),
#         'std_us_upvotes': np.std(us_upvotes),
#         'comment_counts': comment_counts,
#         'total_comments': sum([value for key,value in comment_counts.items()])
#     }
#     conversation_metrics.append(conv_info)

# print('Conversation metrics:')
# metric_list = {
#     'num_s_replies': [],
#     'num_us_replies': [],
#     'num_n_replies': [],
#     'total_s_upvotes':[],
#     'total_us_upvotes': []
# }
# for c in conversation_metrics:
#     for metric in c:
#         if metric in metric_list.keys():
#             metric_list[metric].append(c[metric])

# for metric, values in metric_list.items():
#     print(metric, ": ", end = " ")
#     print(np.mean(values), np.median(values))
# print()

conversation_metrics = pd.DataFrame(conversation_metrics)
conversation_metrics.to_csv('../datasets/processed/AnnotatedCMV/conversation_metrics.csv', header=True, index = False)

print('Written to file.')