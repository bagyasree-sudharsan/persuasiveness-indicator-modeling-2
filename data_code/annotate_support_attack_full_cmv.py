'''
    To annotate support/attack/neutral relations between 
    1. Text and the original post.
    2. Text and the reply immediately above the text, i.e., the comment that the current text is directly replying to. 

    Change the list slices based on which records you want to annotate. If the last one annotated is 35, for example, you must start
    new annotations at record number 35 or earlier, or there will be issues. 

    The script will display the current text, followed by the root (OP) for the text. Enter a/s/n to annotate. If the text is a direct
    reply to another comment, that comment will then show up as 'REPLY TO', which can also be annotated with a/s/n. 

    The horizontal divider indicates that a certain text has been completely annotated, i.e., for both the OP and its parent comment.

    Run process_relation_annotations.py to get the id for each relation, to use in models/predictions.
'''

import json

with open('datasets/processed/CMV/cmv_usable.jsonl', 'r') as infile:
    utterances = json.load(infile)

print('Loaded utterances.')

with open('data_code/annotation_middleware_data.json', 'r') as infile:
    annotation_middleware_data = json.load(infile)

roots = annotation_middleware_data['roots']
replies = annotation_middleware_data['replies']
annotated = []

for utterance in utterances[30:35]:
    if utterance['id'] == utterance['root']:
        roots[utterance['id']] = utterance['text']
        replies[utterance['id']] = utterance['text']
    else:
        try:
            root = roots[utterance['root']]
        except:
            print('Root not available at this time.')
            continue
        
        text = utterance['text']
        print('TEXT: ')
        print(text)

        print('ROOT: ')
        print(root)

        root_relation = input()
        
        reply_relation = None
        reply_to = None
        if utterance['reply-to'] and utterance['reply-to'] != utterance['root']:
            try:
                reply_to = replies[utterance['reply-to']]
                print('REPLY TO:')
                print(replies[utterance['reply-to']])
                reply_relation = input()
            except:
                print('Reply-to text not available at this time.')
        print()    
        print('----------------------------------------------------------------------------------')
        print()
        replies[utterance['id']] = text

        annotated.append({
            'text': text,
            'root_relation': root_relation,
            'reply_relation': reply_relation,
            'root_text': root,
            'reply_to_text': reply_to
        })

with open('datasets/processed/CMV/CMV_relations_annotated.json', 'r') as infile:
    data = json.load(infile)

data.extend(annotated)

with open('datasets/processed/CMV/CMV_relations_annotated.json', 'w') as outfile:
    json.dump(data, outfile, indent = 4)

annotation_middleware_data = {
    'roots': roots,
    'replies': replies
}
with open('data_code/annotation_middleware_data.json', 'w') as outfile:
    json.dump(annotation_middleware_data, outfile, indent = 4)