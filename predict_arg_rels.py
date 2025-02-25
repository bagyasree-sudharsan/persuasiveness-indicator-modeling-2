from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizer

def predict_arg_rels_bert(texts):
    model = DistilBertForSequenceClassification.from_pretrained('models/ArgRelClassifier')
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    tokenizer_kwargs = {"truncation": True, "padding": "max_length"}
    model_pipeline = pipeline('text-classification', model, tokenizer = tokenizer, device = 0)
    
    predictions = []
    outputs = [model_pipeline(example, **tokenizer_kwargs, top_k = None) for example in texts]
    for output in outputs:
        label_scores = [(label_score['label'], label_score['score']) for label_score in output]
        predictions.append(max(label_scores, key=lambda label_score: label_score[1]))
    
    if len(predictions):
        labels, scores = zip(*predictions)
        return labels, scores
    else:
        return [], []

def predict_arg_rels_llama(texts):
    pass
