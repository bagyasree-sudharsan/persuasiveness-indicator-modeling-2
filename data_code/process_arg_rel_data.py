import pandas as pd
import os
import json

folder_paths = [
    '../datasets/original/tacl_arg_rel-main/tacl_arg_rel-main/data/debate',
    '../datasets/original/tacl_arg_rel-main/tacl_arg_rel-main/data/kialo'
]

data = []

for folder in folder_paths:
    claim_premise_pairs = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        print(file_path)
        df = pd.read_csv(file_path)
        if 'propid_to' in df.columns:
            for index, row in df.iterrows():
                # print(row['propid_to'], row['propid_from'])
                if (row['propid_to'], row['propid_from']) not in claim_premise_pairs:
                    data.append({
                        'claim_id': row['propid_to'],
                        'claim': row['text_to'],
                        'premise': row['text_from'],
                        'premise_id': row['propid_from'],
                        'relation': row['relation']
                    })
                    claim_premise_pairs.append((row['propid_to'], row['propid_from']))
    print('--------Finished a folder.-------------')


num_support = 0
num_attack = 0
num_neutral = 0
for d in data:
    if d['relation'] == 1:
        num_support += 1
    elif d['relation'] == -1:
        d['relation'] = 0
        num_attack += 1
    else:
        d['relation'] = 2
        num_neutral += 1

print(len(data))
print(num_support, num_attack, num_neutral)

with open('../datasets/processed/ArgRelData/arg_rel_data.json', 'w') as outfile:
    json.dump(data, outfile, indent = 4)
