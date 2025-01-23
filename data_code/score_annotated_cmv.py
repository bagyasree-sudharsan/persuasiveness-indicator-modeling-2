import json

with open('../datasets/processed/AnnotatedCMV/successful_unlabeled.json', 'r') as infile:
    successful = json.load(infile)
with open('../datasets/processed/AnnotatedCMV/unsuccessful_unlabeled.json', 'r') as infile:
    unsuccessful = json.load(infile)
with open('../datasets/processed/AnnotatedCMV/neutral_unlabeled.json', 'r') as infile:
    neutral = json.load(infile)

with open('../datasets/processed/CMV/final.json', 'r') as infile:
    final_full = json.load(infile)

successful_ids = [u['id'] for u in successful]
unsuccessful_ids = [u['id'] for u in unsuccessful]
neutral_ids = [u['id'] for u in neutral]
final_ids = successful_ids + unsuccessful_ids + neutral_ids

final_data = [u for u in final_full if u['id'] in final_ids]

with open('../datasets/processed/AnnotatedCMV/final.json', 'w') as outfile:
    json.dump(final_data, outfile, indent = 4)