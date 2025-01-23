import os
import json
import pandas as pd

directory = '../datasets/original/UKP1/UKPConvArg1Strict-CSV/'

final_data = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        df = pd.read_csv(f, sep='\t')
        df.rename(columns = {'#id':"id", "label":"winner"}, inplace = True)
        data = df.to_dict(orient = 'records')
        final_data.extend(data)
    else:
        print('Not a valid file.', filename)

with open('../datasets/processed/UKP/final.json', 'w') as outfile:
    json.dump(final_data, outfile, indent = 4)