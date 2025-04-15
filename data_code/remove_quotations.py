import json
import re

with open('datasets/processed/CMV/CMV_relations_annotated_karthik.json', 'r') as infile:
    data = json.load(infile)

for d in data:
    for key, value in d.items():
        if key in ['text', 'root_text', 'reply_to_text'] and value:
            quotations = re.findall('&gt(.+)\n\n', value)
            for q in quotations:
                value = value.replace(q, '')
            d[key] = value


with open('datasets/processed/CMV/CMV_relations_annotated_karthik.json', 'w') as outfile:
    json.dump(data, outfile, indent = 4)
