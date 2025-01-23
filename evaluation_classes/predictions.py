import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoModel
from globals import TextDataset, PRETRAINED_TOKENIZER, TOKENIZER_KWARGS
from split_arguments import split_text

class Prediction:
    def __init__(self):
        pass

    def predict_regression(self, model_path, texts, tokenizer = PRETRAINED_TOKENIZER):
        model = AutoModel.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
        tokenized_input = tokenizer(texts)
        print(model(**tokenized_input))

    def predict(self, model_path, texts, num_labels, tokenizer = PRETRAINED_TOKENIZER, predict_score = False):

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
        model_pipeline = pipeline('text-classification', model, tokenizer = tokenizer, device = 0)
        outputs = [model_pipeline(example, **TOKENIZER_KWARGS,  top_k = None) for example in texts]

        predictions = []
        if not predict_score:
            for output in outputs:
                if not predict_score:
                    label_scores = [(label_score['label'], label_score['score']) for label_score in output]
                    predictions.append(max(label_scores, key=lambda label_score: label_score[1]))
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
            predictions = [(label, score) for label, score in zip(labels, scores)]

        return predictions
    
    def prepare_data_for_prediction(self, path_to_data):
        with open(path_to_data, 'r') as infile:
            data = json.load(infile)

        eval_texts = ['\n'.join(d['text'])for d in data if d is not None]
        eval_labels = [d['is_successful'] for d in data if d is not None]
        eval_scores = [d['score'] for d in data if d is not None]

        return eval_texts, eval_labels, eval_scores
    
    def get_predicted_labels(self, predictions, base_on_score = False):
        return [prediction[0] for prediction in predictions]
    
    def get_predicted_scores(self, predictions):
        return [prediction[1] for prediction in predictions]

    def predict_better_argument(self, arg_pairs, model_path, tokenizer = PRETRAINED_TOKENIZER):
        model_pipeline = pipeline('text-classification', model_path, tokenizer = tokenizer, device = 0)
        predictions = []

        for pair in arg_pairs:
            a1_score = model_pipeline(pair[0], **TOKENIZER_KWARGS, top_k = None)[0]['score']
            a2_score = model_pipeline(pair[1], **TOKENIZER_KWARGS, top_k = None)[0]['score']
            if a1_score > a2_score:
                predictions.append(0)
            else:
                predictions.append(1)
        
        return predictions

    def prepare_ukp_data(self, path_to_data):
        with open(path_to_data, 'r') as infile:
            data = json.load(infile)

        arg_pairs = [(record['a1'], record['a2']) for record in data]
        actual_labels = [0 if record['winner'] == 'a1' else 1 for record in data]
        
        return arg_pairs, actual_labels
    
    def prepare_arg_comp_data_for_prediction(self, path_to_data, label_key = None):
        with open(path_to_data, 'r') as infile:
            data = json.load(infile)
        
        if label_key:
            eval_texts = [segment for d in data for segment in d['text_segments']]
            eval_labels = [tag for d in data for tag in d[label_key]]
            arg_comps = [tag for d in data for tag in d['text_seg_arg_comp_tags']]
        else:
            eval_texts = [segment for text in eval_texts for segment in split_text(text)] 
            eval_labels = []
            arg_comps = None

        return eval_texts, eval_labels, arg_comps
    
    def predict_arg_comps(self, texts, arg_comps = None, actual = None):
        predictions = []

        if not arg_comps:
            model_pipeline = pipeline('text-classification', 'models/TextSegArgComps', tokenizer = PRETRAINED_TOKENIZER, device = 0)
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

    def prepare_data_for_prediction_final(self, dataset_path):
        with open(path_to_data, 'r') as infile:
            data = json.load(infile)

        eval_texts = ['\n'.join(d['text'])for d in data if d is not None]
        eval_labels = [d['is_successful'] for d in data if d is not None]
        eval_scores = [d['score'] for d in data if d is not None]
    

    