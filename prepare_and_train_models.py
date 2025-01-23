from symbolic_regressor import symbolic_regressor
import json
import random

def load_data(path_to_data):
    with open(path_to_data, 'r') as infile:
        data = json.load(infile)

    data_tuples = [(d['text'], d['score']) for d in data if isinstance(d['text'], str)]
    random.shuffle(data_tuples)

    texts = [d[0] for d in data_tuples]
    labels = [d[1] for d in data_tuples]
    
    return texts, labels

def load_and_combine_datasets(path_to_train_dataset, path_to_test_dataset, percent_of_test_data):
    with open(path_to_train_dataset, 'r') as infile:
        train_data = json.load(infile)

    with open(path_to_test_dataset, 'r') as infile:
        test_data = json.load(infile)

    num_test_examples = int(np.floor(len(test_data) * percent_of_test_data))
    test_indices = random.sample(range(0, len(test_data)), num_test_examples)
    test_data_to_add = [test_data[i] for i in range(0, test_indices)]
    train_data += test_data_to_add

    data_tuples = [(d['text'], d['score']) for d in train_data if isinstance(d['text'], str)]
    random.shuffle(data_tuples)

    texts = [d[0] for d in data_tuples]
    labels = [d[1] for d in data_tuples]
    
    return texts, labels


training = [
    [
        'AnnotatedCMV',
        'datasets/processed/AnnotatedCMV/final.json',
        '',
        None,
        'annotated_cmv',
        'AnnotatedCMV_TextSegTagged'
    ],
    [
        'Full CMV',
        'datasets/processed/CMV/final.json',
        '',
        None,
        'full_cmv',
        'CMV_TextSegTagged'
    ],
    [
        'SCOA',
        'datasets/processed/SCOA/final.json',
        '',
        None,
        'scoa',
        'SCOA_TextSegTagged'
    ],
    [
        'Full CMV + 15% SCOA',
        'datasets/processed/CMV/final.json',
        'datasets/processed/SCOA/final.json',
        0.15,
        'cmv_scoa_15',
        'CMV_SCOA_15_TextSegTagged'
    ],
    [
        'Full CMV + 30% SCOA',
        'datasets/processed/CMV/final.json',
        'datasets/processed/SCOA/final.json',
        0.30,
        'cmv_scoa_30',
        'CMV_SCOA_30_TextSegTagged'
    ],
     [
        'SCOA  + 15% CMV',
        'datasets/processed/SCOA/final.json',
        'datasets/processed/CMV/final.json',
        0.15,
        'scoa_cmv_15',
        'SCOA_CMV_15_TextSegTagged'
    ],
    [
        'SCOA  + 30% CMV',
        'datasets/processed/SCOA/final.json',
        'datasets/processed/CMV/final.json',
        0.30,
        'scoa_cmv_30',
        'SCOA_CMV_30_TextSegTagged'
    ],

]

for t in training:
    print(t[0], '----------------------------------------------------------')
    if t[3]:
        texts, labels = load_and_combine_datasets(t[1], t[2], t[3])
    else:
        texts, labels = load_data(t[1])
    symbolic_regressor(texts, labels, 0.85, t[4], t[5])
    print('---------------------------------------------------------------------------')


