import json

with open('datasets/processed/CMV/CMV_relations_annotated_karthik.json', 'r') as infile:
    data = json.load(infile)

label2id = {
    'a': 0,
    's': 1,
    'n': 2,
    'A': 0,
    'S': 1,
    'N': 2,
    0: 0,
    1: 1,
    2: 2
}

for d in data:
    if 'root_relation' in d and d['root_relation']:
        d['root_relation'] = label2id[d['root_relation']]
    if 'reply_relation' in d and d['reply_relation']:
        d['reply_relation'] = label2id[d['reply_relation']]


with open('datasets/processed/CMV/CMV_relations_annotated_karthik.json', 'w') as outfile:
    json.dump(data, outfile, indent = 4)

