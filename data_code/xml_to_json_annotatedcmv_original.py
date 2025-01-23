import json 
import os
import xml.etree.ElementTree as ET

def convert_xml_to_dict(thread):
    d = {
        'root_id':thread.attrib['ID'],
        'replies': []
    }
    for tag in thread:
        if tag.tag == 'title':
            d['root_title'] = tag.text
        if tag.tag == 'OP':
            d['root_author'] = tag.attrib['author']
            d['root_text'] = tag.text
        if tag.tag == 'reply':
            d['replies'].append({
                'id': tag.attrib['id'],
                'author': tag.attrib['author'],
                'text': tag.text
            })
    return d
    
def parse_xml_files(directory):
    unparseable = []
    data = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            try:
                tree = ET.parse(f)
            except:
                unparseable.append(filename)
                break
            root = tree.getroot()
            d = convert_xml_to_dict(root)
            data.append(d)
    return data

directory_neg = '../datasets/original/AnnotatedCMV/original/negative/'
directory_pos = '../datasets/original/AnnotatedCMV/original/positive/'

neg = parse_xml_files(directory_neg)
pos = parse_xml_files(directory_pos)

with open('negative_annotated_original.json', 'w') as outfile:
    json.dump(neg, outfile, indent = 4)

with open('positive_annotated_original.json', 'w') as outfile:
    json.dump(pos, outfile, indent = 4)