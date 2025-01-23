import json
import os
import xml.etree.ElementTree as ET

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import ARG_COMP_TAGS, SEMANTIC_TYPE_DICT, COMBINED_DICT, SEMANTIC_TYPE_TAGS, COMBINED_TAGS, TEXT_SEG_ARG_COMP_TAGS, TEXT_SEG_COMBINED_TAGS, TEXT_SEG_COMBINED_DICT
from split_arguments import split_text

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

def attach_text_seg_tags(arg_comp):
    segment = arg_comp.text
    arg_comp_tag = TEXT_SEG_ARG_COMP_TAGS[1] if arg_comp.tag == 'claim' else TEXT_SEG_ARG_COMP_TAGS[2]
    sem_type_tag = SEMANTIC_TYPE_DICT[arg_comp.attrib['type']]
    combined_tag = TEXT_SEG_COMBINED_DICT[arg_comp.attrib['type']]
    untagged_segment = arg_comp.tail if (arg_comp.tail and len(arg_comp.tail) > 3) else None

    return segment, arg_comp_tag, sem_type_tag, combined_tag, untagged_segment
    
def parse_annotations(thread):
    annotated_data = []
    for tag in thread:
        if tag.tag == 'title' or tag.tag == 'OP' or tag.tag == 'reply':
            tagged_text = {
                'text': '',
                'text_words': [],
                'arg_comp_tags':[],
                'semantic_type_tags':[],
                'combined_tags':[],
                'text_segments': [],
                'text_seg_arg_comp_tags':[],
                'text_seg_sem_type_tags':[],
                'text_seg_combined_tags':[]
            }
            tagged_text['id'] = thread.attrib['ID'] if (tag.tag == 'OP' or tag.tag == 'title') else tag.attrib['id']
            
            for arg_comp in tag:
                text, text_words, arg_comp_tags, semantic_type_tags, combined_tags = attach_correct_arg_comp_tags(arg_comp)
                tagged_text['text'] += text
                tagged_text['text_words'].extend(text_words)
                tagged_text['arg_comp_tags'].extend(arg_comp_tags)
                tagged_text['semantic_type_tags'].extend(semantic_type_tags)
                tagged_text['combined_tags'].extend(combined_tags)

                segment, arg_comp, sem_type, combined, untagged = attach_text_seg_tags(arg_comp)
                tagged_text['text_segments'].append(segment)
                tagged_text['text_seg_arg_comp_tags'].append(arg_comp)
                tagged_text['text_seg_sem_type_tags'].append(sem_type)
                tagged_text['text_seg_combined_tags'].append(combined)
                if untagged:
                    tagged_text['text_segments'].append(untagged)
                    tagged_text['text_seg_arg_comp_tags'].append('NA')
                    tagged_text['text_seg_sem_type_tags'].append('NA')
                    tagged_text['text_seg_combined_tags'].append('NA')

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

text_seg_arg_comp_tag_ids = {k: v for v, k in enumerate(TEXT_SEG_ARG_COMP_TAGS)}
text_seg_combined_tag_ids = {k: v for v, k in enumerate(TEXT_SEG_COMBINED_TAGS)}

for sample in final:
    sample['arg_comp_tags'] = [arg_comp_tag_ids[tag] for tag in sample['arg_comp_tags']]
    sample['semantic_type_tags'] = [semantic_type_tag_ids[tag] for tag in sample['semantic_type_tags']]
    sample['combined_tags'] = [combined_tag_ids[tag] for tag in sample['combined_tags']]
    sample['text_seg_arg_comp_tags'] = [text_seg_arg_comp_tag_ids[tag] for tag in sample['text_seg_arg_comp_tags']]
    sample['text_seg_sem_type_tags'] = [semantic_type_tag_ids[tag] for tag in sample['text_seg_sem_type_tags']]
    sample['text_seg_combined_tags'] = [text_seg_combined_tag_ids[tag] for tag in sample['text_seg_combined_tags']]


print(len(final))

with open('../datasets/processed/AnnotatedCMV/annotated_arg_comps.json', 'w') as outfile:
    json.dump(final, outfile, indent = 4)

print('Written to file.')


# '''
#     Adding text segment annotations to dataset.
# '''


# with open('../datasets/processed/AnnotatedCMV/annotated_arg_comps.json', 'r') as infile:
#     data = json.load(infile)

# for sample in data[:5]:
#     text_segments = split_text([sample['text']])[0]
#     # print(text_segments)
#     # text_seg_id = 0
#     # text_seg_arg_comp_tags = []
#     # text_seg_sem_type_tags = []
#     # for i in range(0, len(sample['arg_comp_tags'])):
#     #     arg_comp = sample['arg_comp_tags'][i]
#     #     if arg_comp == 1 or arg_comp == 3:
#     #         while sample['text_words'][i] not in text_segments[text_seg_id]:
#     #             text_seg_id += 1
#     #             text_seg_arg_comp_tags.append('Not found')
#     #             text_seg_sem_type_tags.append('Not found')
#     #         text_seg_arg_comp_tags.append(arg_comp)
#     #         text_seg_sem_type_tags.append(sample['semantic_type_tags'][i])

#     # num_words = 0
#     # all_words = []
#     # for segment in text_segments:
#     #     words = segment.split(' ')
#     #     for word in words:
#     #         newline_separated = word.split('\n\n')
#     #         for n in newline_separated:
#     #             all_words.append(n)
#     #             all_words.append('\n\n')
#     #         all_words.pop()
    
#     # for i in range(0, len(all_words) - 3):
#     #     if all_words[i:i+3] == ['', '\n\n', '']:

#     # print(len(all_words), len(sample['text_words']))

#     # if len(all_words) != len(sample['text_words']):
#     #     print(all_words)
#     #     print(sample['text_words'])
#     #     print('-------')
#     arg_comp_tags = []
#     sem_type_tags = []

#     prev_tag = None
#     for tag in sample['arg_comp_tags']:
#         tag = tag if tag not in (1, 3) else tag + 1
#         if tag != prev_tag:
#             arg_comp_tags.append(tag)
#             prev_tag = tag 

#     prev_tag = None
#     for tag in sample['semantic_type_tags']:
#         if tag != prev_tag:
#             sem_type_tags.append(tag)
#             prev_tag = tag

#     arg_comp_tags = [tag for tag in arg_comp_tags if tag != 0]
#     arg_comp_tags = [tag for tag in sem_type_tags if tag != 0]

#     print(len(arg_comp_tags), len(sem_type_tags), len(text_segments))
    