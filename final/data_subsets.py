import json
import random
import numpy as np
from constants import FINAL_DATA_PATH

def create_combined_datasets_for_training(train_dataset_path, test_dataset_path, percent, new_dataset_file):
    with open(train_dataset_path, 'r') as infile:
        train_data = json.read(infile)
    with open(test_dataset_path, 'r') as infile:
        test_data = json.read(infile)
    
    num_test = np.ceil(len(test_data) * percent)
    test_data_to_train = test_data[:num_test]

    merged_data = train_data + test_data_to_train
    random.shuffle(merged_data)

    with open(FINAL_DATA_PATH + f'{new_dataset_file}', 'w') as outfile:
        json.dump(merged_data, outfile, indent = 4)

def create_ukp_subset(file_path, num_per_category, output_file):
    with open(file_path, 'r') as infile:
        data = json.load(infile)
    
    a1_winner = [{**d, 
        'arg_comps_a1': [],
        'arg_comps_a2': [],
        'sem_types_a1': [],
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

    with open(FINAL_DATA_PATH + '{}_test.json'.format(output_file), 'w') as outfile:
        json.dump(required_data, outfile, indent = 4)

def create_ap_subset(file_path, output_file, train_test_split = 0.75):
    with open(file_path, 'r') as infile:
        data = json.load(infile)
    
    successful = [(d['text'], d['is_successful'], d['score']) for d in data if (isinstance(d['text'], str) and d['is_successful'] == 1)]
    unsuccessful = [(d['text'], d['is_successful'], d['score']) for d in data if (isinstance(d['text'], str) and d['is_successful'] == 0)]
    
    random.shuffle(successful)
    random.shuffle(unsuccessful)

    num_successful_train = int(len(successful) * train_test_split)
    num_unsuccessful_train = int(len(unsuccessful) * train_test_split)

    train_data_tuples = successful[:num_successful_train] + unsuccessful[:num_unsuccessful_train]
    random.shuffle(train_data_tuples)
    train_data = [{
        'text': text, 
        'is_successful': is_successful, 
        'score': score,
        'arg_comps': [],
        'sem_types': [],
        'text_segments': []} for text, is_successful, score in train_data_tuples]

    test_data_tuples = successful[num_successful_train:] + unsuccessful[num_unsuccessful_train:]
    random.shuffle(test_data_tuples)
    test_data = [{
        'text': text, 
        'is_successful': is_successful, 
        'score': score,
        'arg_comps': [],
        'sem_types': [],
        'text_segments': []} for text, is_successful, score in test_data_tuples]

    print(f'Train data size: {len(train_data)}, Test data size: {len(test_data)}')
    with open(FINAL_DATA_PATH + '{}_train.json'.format(output_file), 'w') as outfile:
        json.dump(train_data, outfile, indent = 4)
    with open(FINAL_DATA_PATH + '{}_test.json'.format(output_file), 'w') as outfile:
        json.dump(test_data, outfile, indent = 4)

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

    with open(FINAL_DATA_PATH + '{}_train.json'.format(output_file), 'w') as outfile:
        json.dump(train_data, outfile, indent = 4)
    with open(FINAL_DATA_PATH + '{}_test.json'.format(output_file), 'w') as outfile:
        json.dump(test_data, outfile, indent = 4)


create_subsets('datasets/processed/CMV/final.json', 2500, 'CMV') #Original results in NeurSym paper used subsets of 1500 each.
create_subsets('datasets/processed/SCOA/final.json', 2500, 'SCOA', is_scoa = True) #Original results in NeurSym paper used subsets of 1500 each.
create_ap_subset('datasets/processed/AP/final.json', 'AP') #AP dataset does not have "neutral" values.
create_ukp_subset('datasets/processed/UKP/final.json', 2500, 'UKP') #Original results in NeurSym paper used subsets of 1500 each.

# create_combined_datasets_for_training(FINAL_DATA_PATH + 'CMV_train_tagged.json', FINAL_DATA_PATH + 'SCOA_train_tagged.json', 0.15, 'CMV_SCOA15.json')
# create_combined_datasets_for_training(FINAL_DATA_PATH + 'CMV_train_tagged.json', FINAL_DATA_PATH + 'SCOA_train_tagged.json', 0.30, 'CMV_SCOA30.json')
# create_combined_datasets_for_training(FINAL_DATA_PATH + 'SCOA_train_tagged.json', FINAL_DATA_PATH + 'CMV_train_tagged.json', 0.15, 'SCOA_CMV15.json')
# create_combined_datasets_for_training(FINAL_DATA_PATH + 'SCOA_train_tagged.json', FINAL_DATA_PATH + 'CMV_train_tagged.json', 0.30, 'SCOA_CMV30.json')