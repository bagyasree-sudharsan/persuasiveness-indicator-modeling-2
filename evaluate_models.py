from evaluation_classes.predictions import Prediction
from evaluation_classes.metrics import Metrics 

evaluation = Metrics()
prediction = Prediction()

def evaluate_model(model_path, dataset_path, evaluation_title, num_labels):
    print('EVALUATION: ', evaluation_title)

    # texts, actual_values, actual_scores = prediction.prepare_data_for_prediction(dataset_path)
    # if num_labels == 1:
    #     predicted_values = prediction.predict(model_path, texts, num_labels, predict_score = True)
    # else:
    #     predicted_values = prediction.predict(model_path, texts, num_labels)

    # predicted_labels = prediction.get_predicted_labels(predicted_values)
    # f1_score = evaluation.f1(predicted_labels, actual_values)
    # precision = evaluation.precision(predicted_labels, actual_values)
    # recall = evaluation.recall(predicted_labels, actual_values)

    # print()
    # print('F1, Precision, Recall: ', end = '')
    # print(f1_score, precision, recall)

    # predicted_scores = prediction.get_predicted_scores(predicted_values)
    # p10, p20, p30 = evaluation.evaluate_scores(predicted_scores, actual_scores)

    # print('Within 10 percent, within 20 percent, within 30 percent: ', end = '')
    # print(p10, p20, p30)

    # mse = evaluation.mse(predicted_scores, actual_scores)
    # mae = evaluation.mae(predicted_scores, actual_scores)
    # print('Mean Square Error, Mean Absolute Error: ', end = '')
    # print(mse, mae)

    predicted_better, actual_better = prediction.predict_better_argument(model_path, 'datasets/processed/UKP/final.json')
    f1_score_better = evaluation.f1(predicted_better, actual_better)
    print('F1 score for predicting the stronger argument: ', f1_score_better)

def evaluate_arg_comp_model(model_path, dataset_path, num_labels, label_key = None):
    texts, actual_values = prediction.prepare_arg_comp_data_for_prediction(dataset_path, label_key)
    predicted_values = prediction.predict_arg_comps(model_path, texts, num_labels)
    predicted_labels = prediction.get_predicted_labels(predicted_values)
    f1_score = evaluation.f1(predicted_labels, actual_values)
    precision = evaluation.precision(predicted_labels, actual_values)
    recall = evaluation.recall(predicted_labels, actual_values)

    print()
    print('F1, Precision, Recall: ', end = '')
    print(f1_score, precision, recall)


# model_path = 'models/AnnotatedCMV_Classifier'
# dataset_path = 'datasets/processed/AnnotatedCMV/final.json'
# evaluate_model(model_path, dataset_path, 'Baseline AnnotatedCMV on AnnotatedCMV', 3)

# model_path = 'models/AnnotatedCMV_Regressor'
# dataset_path = 'datasets/processed/SCOA/final.json'
# evaluate_model(model_path, dataset_path, 'Baseline AnnotatedCMV on SCOA - Regression', 1)

model_path = 'models/TextSegArgComps'
dataset_path = 'datasets/processed/AnnotatedCMV/annotated_arg_comps.json'
evaluate_arg_comp_model(model_path, dataset_path, 3, 'text_seg_arg_comp_tags')
