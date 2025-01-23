import json

datasets = []

with open('datasets/processed/AnnotatedCMV/final.json', 'r') as infile:
    datasets.append(json.load(infile))
with open('datasets/processed/CMV/final.json', 'r') as infile:
    datasets.append(json.load(infile))
with open('datasets/processed/SCOA/final.json', 'r') as infile:
    datasets.append(json.load(infile))

for dataset in datasets:
    labels = [d['is_successful'] for d in dataset]
    print('Successful: ', labels.count(1))
    print('Unsuccessful: ', labels.count(0))
    print('Neutral: ', labels.count(2))
    print('Total: ', len(labels))
    print()



