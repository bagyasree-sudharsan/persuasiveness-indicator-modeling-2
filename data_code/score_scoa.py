import json


## Probably need to weight this scoring by something else as well. 

with open('../datasets/processed/SCOA/successful.json', 'r') as infile:
    successful = json.load(infile)
with open('../datasets/processed/SCOA/unsuccessful.json', 'r') as infile:
    unsuccessful = json.load(infile)
with open('../datasets/processed/SCOA/neutral.json', 'r') as infile:
    neutral = json.load(infile)


final_data = []

for utterance in successful:
    for_votes = utterance['for_votes']
    against_votes = utterance['against_votes']
    neutral_votes = 9 - for_votes - against_votes

    score = (((for_votes + (0.5 * neutral_votes))/9 * 50) + 50)/100
    utterance['score'] = score
    utterance['is_successful'] = 1

    final_data.append(utterance)


for utterance in unsuccessful:
    for_votes = utterance['for_votes']
    against_votes = utterance['against_votes']
    neutral_votes = 9 - for_votes - against_votes

    score = ((for_votes + (0.5 * neutral_votes))/9 * 50)/100
    utterance['score'] = score
    utterance['is_successful'] = 0
    
    final_data.append(utterance)


for utterance in neutral:
    for_votes = utterance['for_votes']
    against_votes = utterance['against_votes']
    neutral_votes = 9 - for_votes - against_votes

    score = 50
    utterance['score'] = 0.5
    utterance['is_successful'] = 2

    final_data.append(utterance)

with open('../datasets/processed/SCOA/final.json', 'w') as outfile:
    json.dump(final_data, outfile, indent = 4)

print('Written final data to file.')




   