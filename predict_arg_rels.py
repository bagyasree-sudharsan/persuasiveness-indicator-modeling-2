from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizer
import json
from ollama import chat
from ollama import ChatResponse
from sklearn.metrics import f1_score

def get_summaries(claims1, claims2):
    texts = []
    for i in range(0, len(claims1)):
        claim1 = claims1[i]
        claim2 = claims2[i]
        question = f'I will show you two pieces of text, "Claim 1" and "Claim 2." \
        Claim 2 may provide additional support for claim 1, or may refute it, or be unrelated to it. \
        You must summarize these in 350 words or less. Ensure that it is still clear which part is Claim 1 and which Claim 2. \
        Claim 1: {claim1} \n Claim 2: {claim2}'
        response: ChatResponse = chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': question,
        },
        ])
        summary = response.message.content
        texts.append(summary)
    
    return texts

label2id = {
    'LABEL_0': 0,
    'LABEL_1': 1,
    'LABEL_2': 2
}

def predict_arg_rels_bert(texts):
    model = DistilBertForSequenceClassification.from_pretrained('models/models/ArgRelClassifier')
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    tokenizer_kwargs = {"truncation": True, "padding": "max_length"}
    model_pipeline = pipeline('text-classification', model, tokenizer = tokenizer, device = 0)
    
    predictions = []
    outputs = [model_pipeline(example, **tokenizer_kwargs, top_k = None) for example in texts]
    print('Got all predictions.')
    for output in outputs:
        label_scores = [(label_score['label'], label_score['score']) for label_score in output]
        prediction = max(label_scores, key=lambda label_score: label_score[1])
        predictions.append(prediction)
    
    if len(predictions):
        labels, scores = zip(*predictions)
        labels = [label2id[label] for label in labels]
        return labels, scores
    else:
        return [], []

def predict_arg_rels_llama(texts, summary = False):
    predictions = []
    count = 0
    batch_count = 0
    for text in texts:
        if summary:
            question = question = 'In the summary given of an argument, tell me whether the first claim is supported by, attacked by, \
            or is neutral with respect to the second claim. If it is a supportive argument, simply state "SUPPORT." \
            If it is an attacking argument, simply state "ATTACK." If it is neutral, state "NEUTRAL." \n Summary: {}'.format(text)
        else:
            pair = text
            question = 'Tell me whether the second claim given supports or attacks the first claim, or if it neutral with respect to the \
            first claim. If it supports, simply state "SUPPORT." If it attacks, simply state "ATTACK." If it is neutral, state "NEUTRAL." \
            \n Claim 1: {} \n Claim 2: {}'.format(pair[0], pair[1])
        
        response: ChatResponse = chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': question,
        },
        ])
        if 'ATTACK' in response.message.content:
            predictions.append(0)
        elif 'SUPPORT' in response.message.content:
            predictions.append(1)
        else:
            predictions.append(2)
        count += 1
        if count % 50 == 0:
            batch_count += 1
            print('Batch {} complete.'.format(batch_count) )
        # print('-------')
    return predictions

# with open('datasets/processed/CMV/CMV_relations_annotated.json', 'r') as infile:
#     data = json.load(infile)

# relations = []
# claim_pairs = []

# for d in data:
#     # claim_pairs.append((d['root_text'], d['text']))
#     # relations.append(d['root_relation'])
#     if d['reply_relation']:
#         claim_pairs.append((d['reply_to_text'], d['text']))
#         relations.append(d['reply_relation'])


# #------------Llama Predictions------------------------------------------------------------------------

# print('Predicting using Llama...')
# predicted_relations_llama = predict_arg_rels_llama(claim_pairs)
# print('Llama F1: ', f1_score(relations, predicted_relations_llama, average = 'macro'))

# #------------DistilBERT Predictions (Truncated)--------------------------------------------------------

# claim_pairs_truncated = []
# for pair in claim_pairs:
#     claim1 = ' '.join(pair[0].split(' ')[:175])
#     claim2 = ' '.join(pair[1].split(' ')[:175])
#     claim_pairs_truncated.append((claim1, claim2))

# texts = [pair[0] + ' ' + pair[1] for pair in claim_pairs_truncated]
# print('Predicting using DistilBERT (truncated)...')
# predicted_relations_distilbert, scores_distilbert = predict_arg_rels_bert(texts)
# print('DistilBERT F1 (Truncated): ', f1_score(relations, predicted_relations_distilbert, average = 'macro'))

# #-------------- Obtaining Summaries ------------------------------------------------------

# claims = [pair[0] for pair in claim_pairs]
# premises = [pair[1] for pair in claim_pairs]
# texts = get_summaries(claims, premises)
# print('Got summaries.')

# #------------Llama Predictions (Summaries) ------------------------------------------------------------------------

# print('Predicting using Llama (summarized)...')
# predicted_relations_llama = predict_arg_rels_llama(texts, summary = True)
# print('Llama F1 (Summarized): ', f1_score(relations, predicted_relations_llama, average = 'macro'))

# #-------------- DistilBERT Predictions (Summaries) ------------------------------------------------------

# print('Predicting using DistilBERT (summarized)...')
# predicted_relations_distilbert, scores_distilbert = predict_arg_rels_bert(texts)
# print('DistilBERT F1 (Summarized): ', f1_score(relations, predicted_relations_distilbert, average = 'macro'))

