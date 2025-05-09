import json

with open('datasets/processed/CMV/CMV_relations_annotated.json', 'r') as infile:
    bagyasree_annotations = json.load(infile)

with open('datasets/processed/CMV/CMV_relations_annotated_karthik.json', 'r') as infile:
    karthik_annotations = json.load(infile)

bagyasree_annotations_dict = {}

for a in bagyasree_annotations:
    if a['root_text']:
        bagyasree_annotations_dict[(a['text'], a['root_text'])] = a['root_relation']
    if a['reply_to_text']:
        bagyasree_annotations_dict[(a['text'], a['reply_to_text'])] = a['reply_relation']

karthik_annotations_dict = {}
for a in karthik_annotations:
    karthik_annotations_dict[(a['text'], a['reply_to_text'])] = a['reply_relation']

common_count = 0
agreement_count = 0
disagreement_count = 0
disagreements = []
for key in bagyasree_annotations_dict:
    if key in karthik_annotations_dict:
        common_count += 1
        if karthik_annotations_dict[key] == bagyasree_annotations_dict[key]:
            agreement_count += 1
        else:
            disagreement_count += 1
            disagreements.append(key)

annotator_disagreements = [{
    "text": d[0],
    "parent_text": d[1],
    "annotator_1_label": bagyasree_annotations_dict[d],
    "annotator_2_label": karthik_annotations_dict[d]
} for d in disagreements]

with open('annotator_disagreements.json', 'w') as outfile:
    json.dump(annotator_disagreements, outfile, indent = 4)

