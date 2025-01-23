
import os
import json


with open('negative_annotated_original.json', 'r') as infile:
    negative = json.load(infile)
with open('positive_annotated_original.json', 'r') as infile:
    positive = json.load(infile)

neg_reply_ids = [reply['id'] for reply_thread in negative for reply in reply_thread['replies']]
pos_reply_ids = [reply['id'] for reply_thread in positive for reply in reply_thread['replies']]
neg_replies = [(reply_thread['root_id'], reply) for reply_thread in negative for reply in reply_thread['replies']]
pos_replies = [(reply_thread['root_id'], reply) for reply_thread in positive for reply in reply_thread['replies']]
reply_ids = neg_reply_ids + pos_reply_ids
replies = neg_replies + pos_replies

with open('../datasets/processed/CMV/successful.json', 'r') as infile:
    pos = json.load(infile)
with open('../datasets/processed/CMV/unsuccessful.json', 'r') as infile:
    neg = json.load(infile)
with open('../datasets/processed/CMV/neutral.json', 'r') as infile:
    neutral = json.load(infile)

successful = [u for u in pos for r in reply_ids if r in u['id']]
unsuccessful = [u for u in neg for r in reply_ids if r in u['id']]
neutral = [u for u in neutral for r in reply_ids if r in u['id']]

with open('../datasets/processed/AnnotatedCMV/successful_unlabeled.json', 'w') as outfile:
    json.dump(successful, outfile, indent = 4)
with open('../datasets/processed/AnnotatedCMV/unsuccessful_unlabeled.json', 'w') as outfile:
    json.dump(unsuccessful, outfile, indent = 4)
with open('../datasets/processed/AnnotatedCMV/neutral_unlabeled.json', 'w') as outfile:
    json.dump(neutral, outfile, indent = 4)



