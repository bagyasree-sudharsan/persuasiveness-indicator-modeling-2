import json
from statistics import mode
from transformers import AutoModelForSequenceClassification, pipeline, DistilBertTokenizer
from predict_arg_rels import predict_arg_rels_bert, predict_arg_rels_llama

ARG_COMPS = ['NA', 'claim', 'premise']

SEM_TYPES = [
    'NA', 
    'ethos', 
    'logos', 
    'pathos', 
    'ethos-logos',
    'logos-pathos', 
    'ethos-pathos',
    'ethos-logos-pathos',
    'interpretation', 
    'evaluation-emotional', 
    'evaluation-rational', 
    'disagreement', 
    'agreement']

CONJUNCTION_LIST = ['but', 'because', 'therefore', 'thus', 'hence', 'however', 'since']

def get_tokenizer(symbolic = None):
    new_tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    if symbolic is not None:
        additional_special_tokens = ['[{}]'.format(tag.upper()) for tag in ARG_COMPS]
        additional_special_tokens.extend(['[{}]'.format(tag.upper()) for tag in SEM_TYPES])
        additional_special_tokens = list(set(additional_special_tokens))
        special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
        new_tokenizer.add_special_tokens(special_tokens_dict)
    
    return new_tokenizer

def split_text(texts):
    sentence_list = [text.split('.') for text in texts]
    sentence_list = [[sentence.strip() for sentence in sentences if len(sentence.strip())] for sentences in sentence_list ]
    text_segment_list = []
    for sentences in sentence_list:
        text_segments = []
        for sentence in sentences:
            words = sentence.split(' ')
            segments = []
            segment = ''
            for word in words:
                if word.lower() not in CONJUNCTION_LIST:
                    segment = segment + ' ' + word if len(segment) else segment + word
                else:
                    if len(segment):
                        segments.append(segment)
                    segment = word
            if len(segment):
                segments.append(segment)
            text_segments.extend(segments)
        text_segment_list.append(text_segments)
    return text_segment_list

def predict_arg_comps(text_segments):
    model = AutoModelForSequenceClassification.from_pretrained('final/models/ArgCompClassifier')
    tokenizer = get_tokenizer(None)
    tokenizer_kwargs = {"truncation": True, "padding": "max_length"}
    model_pipeline = pipeline('text-classification', model, tokenizer = tokenizer, device = 0)
    
    predictions = []
    outputs = [model_pipeline(example, **tokenizer_kwargs, top_k = None) for example in text_segments]
    for output in outputs:
        label_scores = [(label_score['label'], label_score['score']) for label_score in output]
        predictions.append(max(label_scores, key=lambda label_score: label_score[1]))
    
    if len(predictions):
        labels, scores = zip(*predictions)
        return labels, scores
    else:
        return [], []

with open('datasets/processed/CMV/CMV_relations_annotated.json', 'r') as infile:
    data = json.load(infile)

relations = []
claim_pairs = []

for d in data:
    # claim_pairs.append((d['root_text'], d['text']))
    # relations.append(d['root_relation'])
    if d['reply_relation']:
        text = d['text']
        text_segments = split_text(text)
        # arg_comps, scores_ac = predict_arg_comps(text_segments)
        # claim_indices = []
        # for i in range(len(arg_comps)):
        #     if arg_comps[i] == 1:
        #         claim_indices.append(i)
        # claim_segments = [text_segments[i] for i in claim_indices]
        claim_segments = text_segments
        claim_pairs.append((d['reply_to_text'], claim_segments))
        relations.append(d['reply_relation'])


# #------------Llama Predictions------------------------------------------------------------------------
print('Predicting using Llama...')
predicted_relations_llama = []
for pair in claim_pairs:
    segment_pairs = [(pair[0], segment) for segment in pair[1]]
    segment_predicted_relations = predict_arg_rels_llama(segment_pairs)
    most_common_relation = mode(segment_predicted_relations)
    predicted_relations_llama.append(most_common_relation)
print('Llama F1: ', f1_score(relations, predicted_relations_llama, average = 'macro'))

# #------------ DistilBERT Predictions --------------------------------------------------------
predicted_relations_distilbert = []
print('Predicting using DistilBERT...')
for pair in claim_pairs:
    segment_pair_texts = [pair[0] + ' ' + segment for segment in pair[1]]
    segment_predicted_relations, scores = predict_arg_rels_bert(segment_pair_texts)
    most_common_relation = mode(segment_predicted_relations)
    predicted_relations_distilbert.append(most_common_relation)
print('DistilBERT F1: ', f1_score(relations, predicted_relations_distilbert, average = 'macro'))