from constants import TEXT_SEG_ARG_COMP_TAGS, SEMANTIC_TYPE_TAGS
from common_functions import add_tags_to_text
import json

with open('datasets/processed/CMV/final.json', 'r') as infile:
    data_cmv = json.load(infile)

with open('datasets/processed/SCOA/final.json', 'r') as infile:
    data_scoa = json.load(infile)

with open('datasets/processed/UKP/final.json', 'r') as infile:
    data_ukp = json.load(infile)

cmv_texts = [d['text'] for d in data_cmv if isinstance(d['text'], str)]
scoa_texts = [d['text'] for d in data_scoa if isinstance(d['text'], str)]
# ukp_arg_pairs = [(d['a1'], d['a2']) for d in data_ukp]
# arg1_list = [pair[0] for pair in arg_pairs]
# arg2_list = [pair[1] for pair in arg_pairs]

id2label_argcomps = {i: v for i, v in enumerate(TEXT_SEG_ARG_COMP_TAGS)}
id2label_semtypes = {i: v for i, v in enumerate(SEMANTIC_TYPE_TAGS)}
cmv_texts_with_tags = add_tags_to_text(cmv_texts, id2label_argcomps, id2label_semtypes)
scoa_texts_with_tags = add_tags_to_text(scoa_texts, id2label_argcomps, id2label_semtypes)
# ukp_arg_pairs_with_tags = [(add_tags_to_text([arg1_list[i]], id2label_argcomps, id2label_semtypes), add_tags_to_text([arg2_list[i]]), id2label_argcomps, id2label_semtypes) for i in range(0, len(arg_pairs))]

with open('cmv_tagged.json', 'w') as outfile:
    json.dump(cmv_texts_with_tags, outfile, indent = 4)
with open('scoa_tagged.json', 'w') as outfile:
    json.dump(scoa_texts_with_tags, outfile, indent = 4)
# with open('ukp_tagged.json', 'w') as outfile:
#     json.dump(ukp_arg_pairs_with_tags, outfile, indent = 4)