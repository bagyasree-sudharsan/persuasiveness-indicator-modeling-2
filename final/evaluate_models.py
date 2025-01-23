from metrics import Metrics
from common import add_tags_to_text, get_tokenizer
from transformers import AutoModelForSequenceClassification, pipeline
import json

metrics = Metrics()

def predict(model_path, texts, is_regressor = True, symbolic = None):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = get_tokenizer(symbolic)
    tokenizer_kwargs = {"truncation": True, "padding": "max_length"}
    if symbolic is not None:
        model.resize_token_embeddings(len(tokenizer))

    model_pipeline = pipeline('text-classification', model, tokenizer = tokenizer, device = 0)
    outputs = [model_pipeline(example, **tokenizer_kwargs,  top_k = None) for example in texts]

    if not is_regressor:
        for output in outputs: 
            label_scores = [(label_score['label'], label_score['score']) for label_score in output]
            predictions.append(max(label_scores, key=lambda label_score: label_score[1]))
        
        labels, scores = zip(*predictions)
    
    else:
        scores = [output[0]['score'] for output in outputs]
        labels = [] 
        for score in scores:
            if 0.45 <= score <= 0.55:
                labels.append(2)
            elif score > 0.55:
                labels.append(1)
            else:
                labels.append(0)

    return labels, scores           

def predict_ukp(model_path, text_pairs, symbolic = None):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = get_tokenizer(symbolic)
    tokenizer_kwargs = {"truncation": True, "padding": "max_length"}
    if symbolic is not None:
        model.resize_token_embeddings(len(tokenizer))

    model_pipeline = pipeline('text-classification', model, tokenizer = tokenizer, device = 0)
    predictions = []

    for pair in text_pairs:
        a1_score = model_pipeline(pair[0], **TOKENIZER_KWARGS, top_k = None)[0]['score']
        a2_score = model_pipeline(pair[1], **TOKENIZER_KWARGS, top_k = None)[0]['score']
        if a1_score > a2_score:
            predictions.append(0)
        else:
            predictions.append(1)
    
    return predictions

def evaluate_on_ukp(model_path, symbolic = 'sem_types'):
    with open('datasets/UKP_test.json', 'r') as infile:
        data = json.load(infile)
    print(data[:5])
    text_tuples = [(
        d['a1'],
        d['a2'],
        d['winner'],
        d['arg_comps_a1'],
        d['sem_types_a1'],
        d['arg_comps_a2'],
        d['sem_types_a2'],
        d['text_segments_a1'],
        d['text_segments_a2']
    ) for d in data]

    a1_texts, a2_texts, actual_winners, arg_comps_a1, sem_types_a1, arg_comps_a2, sem_types_a2, text_segments_a1, text_segments_a2 = zip(*text_tuples)
    text_pairs = list(zip(a1_texts, a2_texts))
    if symbolic is not None:
        use_sem_types = True if symbolic == 'sem_types' else False
        a1_with_tags = add_tags_to_text(text_segments_a1, arg_comps, sem_types, use_sem_types)
        a1_with_tags = add_tags_to_text(text_segments_a2, arg_comps, sem_types, use_sem_types)
        text_pairs = list(zip(a1_with_tags, a2_with_tags))
    
    predicted_winners = predict_ukp(model_path, text_pairs, symbolic)
    actual_winners = [0 if winner == 'a1' else 1 for winner in actual_winners]
    metrics.f1(predicted_winners, actual_winners)
    metrics.accuracy(predicted_winners, actual_winners)


def evaluate_model(model_path, test_data_path, is_regressor = True, symbolic = None, ukp_evaluation = True):
    with open(test_data_path, 'r') as infile:
        data = json.load(infile)
    
    text_tuples = [(
        d['text'],
        d['is_successful'],
        d['score'],
        d['arg_comps'],
        d['sem_types'],
        d['text_segments']
    ) for d in data if (isinstance(d['text'], str))]

    texts, actual_labels, actual_scores, arg_comps, sem_types, text_segments = zip(*text_tuples)
    if symbolic is not None:
        use_sem_types = True if symbolic == 'sem_types' else False
        texts = add_tags_to_text(text_segments, arg_comps, sem_types, use_sem_types)
    
    predicted_labels, predicted_scores = predict(model_path, texts, is_regressor, symbolic)

    metrics.f1(predicted_labels, actual_labels)
    metrics.precision(predicted_labels, actual_labels)
    metrics.recall(predicted_labels, actual_labels)
    metrics.conf_matrix(predicted_labels, actual_labels)
    print()

    if is_regressor:
        metrics.mse(predicted_scores, actual_scores)
        metrics.mae(predicted_scores, actual_scores)
        print()
        metrics.evaluate_scores(predicted_scores, actual_scores)
        print()

        if ukp_evaluation:
            evaluate_on_ukp(model_path, symbolic)


evaluate_model('models/BaselineCMVRegressor', 'datasets/CMV_test.json', is_regressor = True, symbolic = None, ukp_evaluation = True)