import json
from common import split_text, get_tokenizer

def predict_arg_comps(text_segments):
    model = AutoModelForSequenceClassification.from_pretrained('models/ArgCompClassifier')
    tokenizer = get_tokenizer(None)
    tokenizer_kwargs = {"truncation": True, "padding": "max_length"}
    model_pipeline = pipeline('text-classification', model, tokenizer = tokenizer, device = 0)
    
    outputs = [model_pipeline(example, **TOKENIZER_KWARGS, top_k = None) for example in text_segments]
    for output in outputs:
        label_scores = [(label_score['label'], label_score['score']) for label_score in output]
        predictions.append(max(label_scores, key=lambda label_score: label_score[1]))
    
    labels, scores = zip(*predictions)
    return labels, scores

def predict_sem_types(text_segments, arg_comps):
    tokenizer = get_tokenizer(None)
    tokenizer_kwargs = {"truncation": True, "padding": "max_length"}
    claim_model = AutoModelForSequenceClassification.from_pretrained('models/ClaimClassifier')
    claim_pipeline = pipeline('text-classification', claim_model, tokenizer = tokenizer, device = 0)
    premise_model = AutoModelForSequenceClassification.from_pretrained('models/PremiseClassifier')
    premise_pipeline = pipeline('text-classification', premise_model, tokenizer = tokenizer, device = 0)

    predictions = []
    for i in range(len(text_segments)):
        if arg_comps[i] == 0:
            predictions.append((0, 1))
        elif arg_comps[i] == 1:
            output = claim_pipeline(texts[i], **tokenizer_kwargs, top_k = None)
            label_scores = [(label_score['label'] + 8, label_score['score']) for label_score in output]
            predictions.append(max(label_scores, key=lambda label_score: label_score[1]))
        else:
            output = premise_pipeline(texts[i], **tokenizer_kwargs, top_k = None)
            label_scores = [(label_score['label'] + 8, label_score['score']) for label_score in output]
            predictions.append(max(label_scores, key=lambda label_score: label_score[1]))
    
    labels, scores = zip(*predictions)
    return labels, scores

def predict_arg_comps(self, texts, arg_comps = None):
        predictions = []

        if not arg_comps:
            model_pipeline = pipeline('text-classification', 'models/ArgCompClassifier', tokenizer = PRETRAINED_TOKENIZER, device = 0)
            outputs = [model_pipeline(example, **TOKENIZER_KWARGS, top_k = None) for example in texts]

            for output in outputs:
                label_scores = [(label_score['label'], label_score['score']) for label_score in output]
                predictions.append(max(label_scores, key=lambda label_score: label_score[1]))

        else:
            claim_model = pipeline('text-classification', 'models/ClaimClassifier', tokenizer = PRETRAINED_TOKENIZER, device = 0)
            premise_model = pipeline('text-classification', 'models/PremiseClassifier', tokenizer = PRETRAINED_TOKENIZER, device = 0)
            for i in range(len(texts)):
                if arg_comps[i] == 0:
                    label_scores = [(0, 1)]
                elif arg_comps[i] == 1:
                    output = claim_model(texts[i], **TOKENIZER_KWARGS, top_k = None)
                    label_scores = [(label_score['label'] + 8, label_score['score']) for label_score in output]
                else:
                    output = premise_model(texts[i], **TOKENIZER_KWARGS, top_k = None)
                    label_scores = [(label_score['label'] + 1, label_score['score']) for label_score in output]

                predictions.append(max(label_scores, key=lambda label_score: label_score[1]))
            
        return predictions



def tag_data(file_path, is_ukp = False):
    with open(file_path, 'r') as infile:
        data = json.load(infile)

    final_data = []

    if not is_ukp:
        for d in data:
            text = d['text']
            text_segments = split_text(text)
            arg_comps, scores_ac = predict_arg_comps(text_segments)
            sem_types, scores_st = predict_sem_types(text_segments, arg_comps)
            final_data.append({
                **d,
                'text_segments': text_segments,
                'arg_comps': arg_comps,
                'sem_types': sem_types
            })
    
    else:
        for d in data:
            text_segments_a1 = split_text(d['a1'])
            text_segments_a2 = split_text(d['a2'])
            arg_comps_a1, scores_ac_a1 = predict_arg_comps(text_segments_a1)
            sem_types_a1, scores_st_a1 = predict_sem_types(text_segments_a1, arg_comps_a1)
            arg_comps_a2, scores_ac_a2 = predict_arg_comps(text_segments_a2)
            sem_types_a2, scores_st_a2 = predict_sem_types(text_segments_a2, arg_comps_a2)
            final_data.append({
                **d,
                'text_segments_a1': text_segments_a1,
                'arg_comps_a1': arg_comps_a1,
                'sem_types_a1': sem_types_a1,
                'text_segments_a2': text_segments_a2,
                'arg_comps_a2': arg_comps_a2,
                'sem_types_a2': sem_types_a2
            })

tag_data('datasets/CMV_train.json')
tag_data('datasets/CMV_test.json')
tag_data('datasets/SCOA_train.json')
tag_data('datasets/SCOA_test.json')
tag_data('datasets/UKP_test.json')
        