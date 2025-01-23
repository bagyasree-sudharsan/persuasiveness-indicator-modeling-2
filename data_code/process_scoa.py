import re
import json

with open('../datasets/original/SCOA/utterances.jsonl', 'r') as infile:
    utterances = infile.read()

utterances = utterances.replace('\\u2019', '')
# Use the timestamp as a delimiter between utterances. 
regex_for_split = re.compile(r'(\"speaker\":\s\"([A-Z]*[a-z]*[0-9]*|\_|\>|\<|\'|\:|\-)*\"\})')
utterances = re.split(regex_for_split, utterances)
speakers = [u[12:-2] for u in utterances[1::3]]
utterances = [u[::-1].replace(',', '}', 1)[::-1] for u in utterances[0::3]][:len(utterances)-1][:-1]
utterances = [json.loads(u) for u in utterances]

with open('../datasets/original/SCOA/conversations.json', 'r') as infile:
    conv_original = json.load(infile)


'''
    Combine utterances for each speaker in a conversation and organize data by conversation and speaker. 
    This is only done if the conversation has a clear win side. 
'''
conversations = {}
for i in range(0, len(utterances)):
    conv_id = utterances[i]['conversation_id']
    win_side = conv_original[conv_id]['win_side']
    if win_side in (0,1):
        speaker = speakers[i]
        
        if conv_id not in conversations:
            conversations[conv_id] = {
                'speaker_utterances': {
                    speaker: (utterances[i]['meta']['side'], [utterances[i]['text']])
                },
                'win_side': conv_original[conv_id]['win_side'],
                '0_votes': list(conv_original[conv_id]['votes_side'].values()).count(0) if conv_original[conv_id]['votes_side'] else None,
                '1_votes': list(conv_original[conv_id]['votes_side'].values()).count(1) if conv_original[conv_id]['votes_side'] else None,

            }
                    
        else:
            if speaker not in conversations[conv_id]['speaker_utterances']:
                conversations[conv_id]['speaker_utterances'][speaker] = (utterances[i]['meta']['side'], [utterances[i]['text']])
            else:
                conversations[conv_id]['speaker_utterances'][speaker][1].append(utterances[i]['text'])

with open('../datasets/processed/SCOA/conv_level_processed.json', 'w') as outfile:
    json.dump(conversations, outfile, indent = 4)

'''
    Separate utterances into successful, unsuccessful, and neutral.
'''

successful = []
unsuccessful = []
neutral = []

for conv_id in conversations:
    win_side = conversations[conv_id]['win_side']
    for speaker, utterances in conversations[conv_id]['speaker_utterances'].items():
        if utterances[0] == win_side:
            successful.append({
                'root': conv_id,
                'text': utterances[1],
                'for_votes': conversations[conv_id]['{}_votes'.format(str(win_side))],
                'against_votes': conversations[conv_id]['{}_votes'.format(str(abs(win_side-1)))]
            })
        elif (utterances[0] in (0, 1)):
            unsuccessful.append({
                'root': conv_id,
                'text': utterances[1],
                'for_votes': conversations[conv_id]['{}_votes'.format(str(abs(win_side-1)))],
                'against_votes': conversations[conv_id]['{}_votes'.format(str(win_side))]
            })
        else:
            neutral.append({
                'root': conv_id,
                'text': utterances[1],
                'for_votes': 4.5,
                'against_votes': 4.5
            })

with open('../datasets/processed/SCOA/successful.json', 'w') as outfile:
    json.dump(successful, outfile, indent = 4)

with open('../datasets/processed/SCOA/unsuccessful.json', 'w') as outfile:
    json.dump(unsuccessful, outfile, indent = 4)

with open('../datasets/processed/SCOA/neutral.json', 'w') as outfile:
    json.dump(neutral, outfile, indent = 4)