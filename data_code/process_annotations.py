import json
import os
import xml.etree.ElementTree as ET

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import ARG_COMP_TAGS, SEMANTIC_TYPE_DICT, COMBINED_DICT, SEMANTIC_TYPE_TAGS, COMBINED_TAGS

# '''
# Attach claim and premise tags to data. 
# '''

def attach_correct_arg_comp_tags(arg_comp):
    text = arg_comp.text
    text_words = text.split(' ')
    arg_comp_tags = []
    semantic_type_tags = []
    combined_tags = []
    
    if arg_comp.tag == 'premise':
        arg_comp_tags.append(ARG_COMP_TAGS[3])
        arg_comp_tags.extend([ARG_COMP_TAGS[4]] * (len(text_words) - 1))
        
    else:
        arg_comp_tags.append(ARG_COMP_TAGS[1])
        arg_comp_tags.extend([ARG_COMP_TAGS[2]] * (len(text_words) - 1))
    
    semantic_type_tags.extend([SEMANTIC_TYPE_DICT[arg_comp.attrib['type']]] * len(text_words))
    combined_tags.append(COMBINED_DICT[arg_comp.attrib['type']][0])
    combined_tags.extend([COMBINED_DICT[arg_comp.attrib['type']][1]] * (len(text_words) - 1))
    
    if arg_comp.tail:
        untagged_text = arg_comp.tail
        untagged_text_words = untagged_text.split(' ')
        text += untagged_text
        text_words.extend(untagged_text_words)
        arg_comp_tags.extend([ARG_COMP_TAGS[0]] * len(untagged_text_words))
        semantic_type_tags.extend([SEMANTIC_TYPE_TAGS[0]] * len(untagged_text_words))
        combined_tags.extend([COMBINED_TAGS[0]] * len(untagged_text_words))
    
    return text, text_words, arg_comp_tags, semantic_type_tags, combined_tags


def parse_annotations(thread):
    annotated_data = []
    root_dict = {
       
    }
    for tag in thread:
        if tag.tag == 'title' or tag.tag == 'OP' or tag.tag == 'reply':
            tagged_text = {
                'text': '',
                'text_words': [],
                'arg_comp_tags':[],
                'semantic_type_tags':[],
                'combined_tags':[]
            }
            tagged_text['id'] = thread.attrib['ID'] if (tag.tag == 'OP' or tag.tag == 'title') else tag.attrib['id']
            for arg_comp in tag:
                text, text_words, arg_comp_tags, semantic_type_tags, combined_tags = attach_correct_arg_comp_tags(arg_comp)
                tagged_text['text'] += text
                tagged_text['text_words'].extend(text_words)
                tagged_text['arg_comp_tags'].extend(arg_comp_tags)
                tagged_text['semantic_type_tags'].extend(semantic_type_tags)
                tagged_text['combined_tags'].extend(combined_tags)

            if len(tagged_text['text']):
                annotated_data.append(tagged_text)
            
    return annotated_data

def parse_xml_files(directory):
    unparseable = []
    data = []
    file_count = 0
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            try:
                tree = ET.parse(f)
            except:
                unparseable.append(filename)
                continue
            root = tree.getroot()
            d = parse_annotations(root)
            data.extend(d)
            file_count += 1
        else:
            unparseable.append(filename)

    return data

directory_neg = '../datasets/original/AnnotatedCMV/negative/'
directory_pos = '../datasets/original/AnnotatedCMV/positive/'

neg = parse_xml_files(directory_neg)
pos = parse_xml_files(directory_pos)
final = neg + pos

arg_comp_tag_ids = {k: v for v, k in enumerate(ARG_COMP_TAGS)}
semantic_type_tag_ids = {k: v for v, k in enumerate(SEMANTIC_TYPE_TAGS)}
combined_tag_ids = {k: v for v, k in enumerate(COMBINED_TAGS)}
for sample in final:
    sample['arg_comp_tags'] = [arg_comp_tag_ids[tag] for tag in sample['arg_comp_tags']]
    sample['semantic_type_tags'] = [semantic_type_tag_ids[tag] for tag in sample['semantic_type_tags']]
    sample['combined_tags'] = [combined_tag_ids[tag] for tag in sample['combined_tags']]

print(len(final))

with open('../datasets/processed/AnnotatedCMV/annotated_arg_comps.json', 'w') as outfile:
    json.dump(final, outfile, indent = 4)

print('Written to file.')
