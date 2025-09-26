# from metrics import Metrics
# from common import add_tags_to_text, get_tokenizer
# from transformers import AutoModelForSequenceClassification, pipeline
# import json
# from constants import FINAL_DATA_PATH

from evaluate_models import evaluate_model

import sys 
sys.stdout = open('evaluation_results.txt','wt')

def evaluate(train_dataset, test_dataset, symbolic):
    if symbolic is None:
        model_name = f'Baseline{train_dataset}Classifier'
        test_dataset_file = f'{test_dataset}_test.json'
        model_substring = 'Baseline'
    else:
        model_substring = 'ArgComps' if symbolc == 'arg_comps' else 'SemTypes'
        model_name = f'{train_dataset}Classifier{model_substring}'
        test_dataset_file = f'{test_dataset}_test_tagged.json'
    
    print(f'Evaluating {train_dataset} ({model_substring}) model on {test_dataset}...')
    evaluate_model(f'models/{model_name}', FINAL_DATA_PATH + f'{test_dataset_file}_test_tagged.json', is_regressor = False, symbolic = symbolic, ukp_evaluation = False)
    print('----------------------------------------------------------------')

if __name__ == "__main__":
    CMV = 'CMV'
    SCOA = 'SCOA'
    AP = 'AP'
    ARG_COMPS = 'arg_comps'
    SEM_TYPES = 'sem_types'

    # CMV evaluations
    evaluate(CMV, CMV, None)
    evaluate(CMV, CMV, ARG_COMPS)
    evaluate(CMV, CMV, SEM_TYPES)
    evaluate(CMV, SCOA, None)
    evaluate(CMV, SCOA, ARG_COMPS)
    evaluate(CMV, SCOA, SEM_TYPES)
    evaluate(CMV, AP, None)
    evaluate(CMV, AP, ARG_COMPS)
    evaluate(CMV, AP, SEM_TYPES)

    # SCOA evaluations
    evaluate(SCOA, SCOA, None)
    evaluate(SCOA, SCOA, ARG_COMPS)
    evaluate(SCOA, SCOA, SEM_TYPES)
    evaluate(SCOA, CMV, None)
    evaluate(SCOA, CMV, ARG_COMPS)
    evaluate(SCOA, CMV, SEM_TYPES)
    evaluate(SCOA, AP, None)
    evaluate(SCOA, AP, ARG_COMPS)
    evaluate(SCOA, AP, SEM_TYPES)

    # AP evaluations
    evaluate(AP, AP, None)
    evaluate(AP, AP, ARG_COMPS)
    evaluate(AP, AP, SEM_TYPES)
    evaluate(AP, CMV, None)
    evaluate(AP, CMV, ARG_COMPS)
    evaluate(AP, CMV, SEM_TYPES)
    evaluate(AP, SCOA, None)
    evaluate(AP, SCOA, ARG_COMPS)
    evaluate(AP, SCOA, SEM_TYPES)

    

