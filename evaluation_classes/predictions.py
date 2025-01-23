import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from globals import TextDataset, PRETRAINED_TOKENIZER, TOKENIZER_KWARGS

class Prediction:
    def __init__(self):
        pass

    def predict(self, model_path, tokenized_texts, num_labels, predict_score = False):
        # eval_dataset = TextDataset(tokenized_texts, train_labels)
        # model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = num_labels).to('cuda')
        model_pipeline = pipeline('text-classification', model_path, tokenizer = PRETRAINED_TOKENIZER, device = 0)
        outputs = [model_pipeline(example, **TOKENIZER_KWARGS, top_k = None) for example in tokenized_texts]

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

        # tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
        # eval_tokenized = tokenizer(eval_texts, padding="max_length", truncation = True, return_tensors = 'pt')
        return eval_texts, eval_labels, eval_scores
    
    def get_predicted_labels(self, predictions, base_on_score = False):
        return [prediction[0] for prediction in predictions]
    
    def get_predicted_scores(self, predictions):
        return [prediction[1] for prediction in predictions]
    