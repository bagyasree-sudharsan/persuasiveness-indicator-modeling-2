'''
- Weight upvotes by successful/unsuccessful, number of comments on the post (this can be an estimation of engagement), 
number of comments the commenter has made vs. average number of comments each commenter makes (estimation of how much 
back and forth was needed to convince - assumption would be that most commenters will not use multiple comments to 
make their argument without some sort of interference from the OP), number of total upvotes for all successful/unsuccessful 
comments on the post. Higher success score → more successful.
- Create an inverse of this for unsuccessful, i.e., the more the upvotes, the less unsuccessful it is. 
Higher success score → less unsuccessful → more successful.
'''


'''
Create between 0-100, and then scale down. 
1. 50 points for being successful.
2. (1 - num_s_replies/total_num_replies) * 10. The idea is that if the comment is one of very few successful comments, more points 
    are awarded. If unsuccessful, no points.
    (alternatively, try combining 1 and 2. So 1 - num_s/total_num * 60, Unsuccessful comments don't get any points.)
3. Number of upvotes > median successful upvotes -> 10 points (irrespective of s or us)
4. Some function of number of upvotes, (total num of upvotes and total comments), (total number of successful upvotes and successful comments)
5. (1- number of comments by commenter/total number of comments) * 10. The fewer the comments needed to persuade, the more persuasive 
each individual comment. - 10 points (should I be accounting for length of comment here? Idts really. shouldn't be penalized 
for making a strong argument by giving examples etc.)
'''

import json
import csv
import random
import numpy as np
import ast

with open('../datasets/processed/CMV/successful.json', 'r') as infile:
    successful = json.load(infile)
with open('../datasets/processed/CMV/unsuccessful.json', 'r') as infile:
    unsuccessful = json.load(infile)
with open('../datasets/processed/CMV/neutral.json', 'r') as infile:
    neutral = json.load(infile)

# df = pd.read_csv('../datasets/processed/CMV/conversation_metrics.csv')
with open('../datasets/processed/CMV/conversation_metrics.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    conv_metrics = {row['id']: row for row in csv_reader}

final_data = []

for utterance in successful:
    metrics = conv_metrics[utterance['root']]
    #For being successful.
    score = 50
    
    #The fewer the number of successful replies in the thread, the higher the score.
    num_s_replies = int(metrics['num_s_replies'])
    total_num_replies = int(metrics['num_s_replies']) + int(metrics['num_us_replies']) + int(metrics['num_n_replies'])
    score += (1 - (num_s_replies/total_num_replies)) * 10

    #Points for being higher than the median and/or average number of successful upvotes.
    utterance_ups = utterance['ups']

    med_s_upvotes = float(metrics['med_s_upvotes'])
    avg_s_upvotes = float(metrics['avg_s_upvotes'])
    if (utterance_ups >= med_s_upvotes and utterance_ups >= avg_s_upvotes):
        score += 10
    elif ((utterance_ups >= med_s_upvotes and utterance_ups < avg_s_upvotes) or (utterance_ups >= avg_s_upvotes and utterance_ups < med_s_upvotes)):
        score += 5

    # Weight the upvotes received by distance from the median for that conversation (in both count and magnitude)
    # For some reason this seems to provide scores that are a bit lower than expected. Will investigate later.
    upvote_counts = ast.literal_eval(metrics['upvote_counts'])
    count_num = 0
    val_num = 0
    for up_val, up_count in upvote_counts.items():
        if up_val < utterance_ups:
            count_num += up_count
            val_num += (up_val * up_count)
        elif up_val == utterance_ups:
            count_num += up_count//2
            val_num += (up_val * up_count)//2

    total_count = int(metrics['num_s_replies']) + int(metrics['num_us_replies'])
    total_count = total_count if total_count > 0 else 1
    total_vals = int(metrics['total_s_upvotes']) + int(metrics['total_us_upvotes'])
    total_vals = total_vals if total_vals > 0 else 1

    score += ((8 * (count_num/total_count)) + (12 * (val_num/total_vals)))

    #Lesser back-and-forth required -> Higher persuasiveness
    comment_counts = json.loads(metrics['comment_counts'].replace("'", '"'))
    total_num_comments = int(metrics['total_comments'])
    comment_count = int(comment_counts[utterance['user']])
    score += (1 - (comment_count/total_num_comments)) * 10

    utterance['is_successful'] = 1
    utterance['score'] = score
    final_data.append(utterance)


#Unsuccessful scores
for utterance in unsuccessful:
    metrics = conv_metrics[utterance['root']]
    
    #Since it is unsuccessful
    score = 0

    #Points for being higher than the median and/or average number of successful upvotes.
    utterance_ups = utterance['ups']

    med_s_upvotes = float(metrics['med_s_upvotes'])
    avg_s_upvotes = float(metrics['avg_s_upvotes'])
    if (utterance_ups >= med_s_upvotes and utterance_ups >= avg_s_upvotes):
        score += 10
    elif ((utterance_ups >= med_s_upvotes and utterance_ups < avg_s_upvotes) or (utterance_ups >= avg_s_upvotes and utterance_ups < med_s_upvotes)):
        score += 5

    # Weight the upvotes received by distance from the median for that conversation (in both count and magnitude)
    # For some reason this seems to provide scores that are a bit lower than expected. Will investigate later.
    upvote_counts = ast.literal_eval(metrics['upvote_counts'])
    count_num = 0
    val_num = 0
    for up_val, up_count in upvote_counts.items():
        if up_val < utterance_ups:
            count_num += up_count
            val_num += (up_val * up_count)
        elif up_val == utterance_ups:
            count_num += up_count//2
            val_num += (up_val * up_count)//2

    total_count = int(metrics['num_s_replies']) + int(metrics['num_us_replies'])
    total_count = total_count if total_count > 0 else 1
    total_vals = int(metrics['total_s_upvotes']) + int(metrics['total_us_upvotes'])
    total_vals = total_vals if total_vals > 0 else 1

    score += ((8 * (count_num/total_count)) + (12 * (val_num/total_vals)))

    #Lesser back-and-forth required -> Higher persuasiveness
    comment_counts = json.loads(metrics['comment_counts'].replace("'", '"'))
    total_num_comments = int(metrics['total_comments'])
    comment_count = int(comment_counts[utterance['user']])
    score += (1 - (comment_count/total_num_comments)) * 10

    utterance['is_successful'] = 0
    utterance['score'] = score
    final_data.append(utterance)

#Neutral scores
for utterance in neutral:
    utterance['is_successful'] = 2
    utterance['score'] = 50
    final_data.append(utterance)

# Normalize scores
scores = [u['score'] for u in final_data]
min_score = np.min(scores)
max_score = np.max(scores)

for utterance in final_data:
    if utterance['is_successful'] != 2:
        utterance['score'] = (utterance['score'] - min_score)/(max_score-min_score)

with open('../datasets/processed/CMV/final.json', 'w') as outfile:
    json.dump(final_data, outfile, indent = 4)

print('Written final data to file.')