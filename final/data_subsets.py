import json
import random

def create_ukp_subset(file_path, num_per_category, output_file):
    with open(file_path, 'r') as infile:
        data = json.load(infile)
    
    a1_winner = [{**d, 
        'arg_comps_a1': [],
        'arg_comps_a2': [],
        'sem_types_a2': [],
        'sem_types_a2': [],
        'text_segments_a1': [],
        'text_segments_a2': []} for d in data if d['winner'] == 'a1']
    a2_winner = [{**d, 
        'arg_comps_a1': [],
        'arg_comps_a2': [],
        'sem_types_a1': [],
        'sem_types_a2': [],
        'text_segments_a1': [],
        'text_segments_a2': []} for d in data if d['winner'] == 'a2']
    random.shuffle(a1_winner)
    random.shuffle(a2_winner)

    required_data = a1_winner[:num_per_category] + a2_winner[:num_per_category]
    random.shuffle(required_data)

    with open('datasets/{}_test.json'.format(output_file), 'w') as outfile:
        json.dump(required_data, outfile, indent = 4)

def create_subsets(file_path, num_per_category, output_file, is_scoa = False):
    with open(file_path, 'r') as infile:
        data = json.load(infile)

    if not is_scoa:
        successful = [(d['text'], d['is_successful'], d['score']) for d in data if (isinstance(d['text'], str) and d['is_successful'] == 1)]
        unsuccessful = [(d['text'], d['is_successful'], d['score']) for d in data if (isinstance(d['text'], str) and d['is_successful'] == 0)]
        neutral = [(d['text'], d['is_successful'], d['score']) for d in data if (isinstance(d['text'], str) and d['is_successful'] == 2)]
    else:
        successful = [('\n'.join(d['text']), d['is_successful'], d['score']) for d in data if d['is_successful'] == 1]
        unsuccessful = [('\n'.join(d['text']), d['is_successful'], d['score']) for d in data if d['is_successful'] == 0]
        neutral = [('\n'.join(d['text']), d['is_successful'], d['score']) for d in data if d['is_successful'] == 2]

    random.shuffle(successful)
    random.shuffle(unsuccessful)
    random.shuffle(neutral)

    data_tuples = successful[:num_per_category] + unsuccessful[:num_per_category] + neutral[:num_per_category]
    random.shuffle(data_tuples)
    train_data = [{
        'text': text, 
        'is_successful': is_successful, 
        'score': score,
        'arg_comps': [],
        'sem_types': [],
        'text_segments': []} for text, is_successful, score in data_tuples]

    data_tuples = successful[num_per_category: num_per_category+num_per_category] + unsuccessful[num_per_category: num_per_category+num_per_category] + neutral[num_per_category: num_per_category+num_per_category]
    random.shuffle(data_tuples)
    test_data = [{'text': text, 
        'is_successful': is_successful, 
        'score': score,
        'arg_comps': [],
        'sem_types': [],
        'text_segments': []} for text, is_successful, score in data_tuples]

    with open('datasets/{}_train.json'.format(output_file), 'w') as outfile:
        json.dump(train_data, outfile, indent = 4)
    with open('datasets/{}_test.json'.format(output_file), 'w') as outfile:
        json.dump(test_data, outfile, indent = 4)


# create_subsets('../datasets/processed/CMV/final.json', 1500, 'CMV')
# create_subsets('../datasets/processed/SCOA/final.json', 1500, 'SCOA', is_scoa = True)
create_ukp_subset('../datasets/processed/UKP/final.json', 1500, 'UKP')