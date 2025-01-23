from evaluation_classes.predictions import Prediction
from evaluation_classes.metrics import Metrics 
from common_functions import add_tags_to_text
from globals import CUSTOM_TOKENIZER
from constants import TEXT_SEG_ARG_COMP_TAGS, SEMANTIC_TYPE_TAGS

import sys 
sys.stdout = open('evaluation_results.txt','wt')

evaluation = Metrics()
prediction = Prediction()

def evaluate_model(model_path, dataset_path, evaluation_title, num_labels):
    print('EVALUATION: ', evaluation_title)

    texts, actual_values, actual_scores = prediction.prepare_data_for_prediction(dataset_path)
    if num_labels == 1:
        predicted_values = prediction.predict(model_path, texts, num_labels, predict_score = True)
        # predicted_values = prediction.predict_regression(model_path, texts, tokenizer = CUSTOM_TOKENIZER)
    else:
        predicted_values = prediction.predict(model_path, texts, num_labels)

    predicted_labels = prediction.get_predicted_labels(predicted_values)
    evaluation.f1(predicted_labels, actual_values)
    evaluation.precision(predicted_labels, actual_values)
    evaluation.recall(predicted_labels, actual_values)
    print()

    predicted_scores = prediction.get_predicted_scores(predicted_values)
    mse = evaluation.mse(predicted_scores, actual_scores)
    mae = evaluation.mae(predicted_scores, actual_scores)
    evaluation.evaluate_scores(predicted_scores, actual_scores)
    print()

    # arg_pairs, actual_better = prediction.prepare_ukp_data('datasets/processed/UKP/final.json')
    # predicted_better = prediction.predict_better_argument(arg_pairs, model_path)
    # print('Stronger argument', end = ' ')
    # evaluation.f1(predicted_better, actual_better)
    # print()

def evaluate_arg_comp_model(dataset_path, num_labels, is_sem_type = False, label_key = None):
    texts, actual_values, arg_comps = prediction.prepare_arg_comp_data_for_prediction(dataset_path, label_key)
    arg_comps = arg_comps if is_sem_type else None
    predicted_values = prediction.predict_arg_comps(texts, arg_comps, actual_values)
    predicted_labels = prediction.get_predicted_labels(predicted_values)
    f1_score = evaluation.f1(predicted_labels, actual_values)
    precision = evaluation.precision(predicted_labels, actual_values)
    recall = evaluation.recall(predicted_labels, actual_values)

    print()
    print('F1, Precision, Recall: ', end = '')
    print(f1_score, precision, recall)

    evaluation.conf_matrix(predicted_labels, actual_values)


def evaluate_final_model(model_path, dataset_path, evaluation_title, num_labels):
    print('EVALUATION: ', evaluation_title)

    texts, actual_values, actual_scores = prediction.prepare_data_for_prediction(dataset_path)
    id2label_argcomps = {i: v for i, v in enumerate(TEXT_SEG_ARG_COMP_TAGS)}
    id2label_semtypes = {i: v for i, v in enumerate(SEMANTIC_TYPE_TAGS)}
    texts_with_tags = add_tags_to_text(texts, id2label_argcomps, id2label_semtypes)

    if num_labels == 1:
        predicted_values = prediction.predict(model_path, texts_with_tags, num_labels, tokenizer = CUSTOM_TOKENIZER, predict_score = True)
    else:
        predicted_values = prediction.predict(model_path, texts_with_tags, num_labels, tokenizer = CUSTOM_TOKENIZER)

    predicted_labels = prediction.get_predicted_labels(predicted_values)
    evaluation.f1(predicted_labels, actual_values)
    evaluation.precision(predicted_labels, actual_values)
    evaluation.recall(predicted_labels, actual_values)
    print()

    predicted_scores = prediction.get_predicted_scores(predicted_values)
    mse = evaluation.mse(predicted_scores, actual_scores)
    mae = evaluation.mae(predicted_scores, actual_scores)
    evaluation.evaluate_scores(predicted_scores, actual_scores)
    print()

    arg_pairs, actual_better = prediction.prepare_ukp_data('datasets/processed/UKP/final.json')
    arg1_list = [pair[0] for pair in arg_pairs]
    arg2_list = [pair[1] for pair in arg_pairs]
    arg_pairs_with_tags = [(add_tags_to_text(arg1_list[i]), add_tags_to_text(arg2_list[i])) for i in range(0, len(arg_pairs))]
    print(arg_pairs_with_tags)
    predicted_better = prediction.predict_better_argument(arg_pairs_with_tags, model_path, tokenizer = CUSTOM_TOKENIZER)
    print('Stronger argument', end = ' ')
    evaluation.f1(predicted_better, actual_better)
    print()


# model_path = 'models/AnnotatedCMV_Classifier'
# dataset_path = 'datasets/processed/AnnotatedCMV/final.json'
# evaluate_model(model_path, dataset_path, 'Baseline AnnotatedCMV on AnnotatedCMV', 3)

# model_path = 'models/AnnotatedCMV_Regressor'
# dataset_path = 'datasets/processed/SCOA/final.json'
# evaluate_model(model_path, dataset_path, 'Baseline AnnotatedCMV on SCOA - Regression', 1)

# model_path = 'models/BaselineRegressor'
# dataset_path = 'datasets/processed/AnnotatedCMV/final.json'
# evaluate_model(model_path, dataset_path, 'Baseline AnnotatedCMV on AnnotatedCMV - Regression', 1)

# dataset_path = 'datasets/processed/AnnotatedCMV/annotated_arg_comps.json'
# evaluate_arg_comp_model(dataset_path, 3, label_key = 'text_seg_arg_comp_tags')


# dataset_path = 'datasets/processed/AnnotatedCMV/annotated_arg_comps.json'
# evaluate_arg_comp_model(dataset_path, 13, is_sem_type = True, label_key = 'text_seg_sem_type_tags')

# model_path = 'models/AnnotatedCMV_TextSegTagged'
# dataset_path = 'datasets/processed/AnnotatedCMV/final.json'
# evaluate_final_model(model_path, dataset_path, 'Final', 1)

evaluating = [
    [
        'models/AnnotatedCMV_TextSegTagged',
        'datasets/processed/AnnotatedCMV/final.json',
        'Annotated on Annotated',
        1
    ]
]

for e in evaluating:
    evaluate_final_model(e[0], e[1], e[2], e[3])