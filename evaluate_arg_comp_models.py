import json
from globals import PRETRAINED_TOKENIZER, TextDataset
from constants import ARG_COMP_TAGS, COMBINED_TAGS
from transformers import pipeline, DataCollatorForTokenClassification
import pickle
import evaluate
import numpy as np
from transformers.pipelines.pt_utils import KeyDataset

def predict(model_path, texts, num_labels):
    # eval_tokenized = PRETRAINED_TOKENIZER(texts, truncation=True, is_split_into_words=True)
    # eval_dataset = TextDataset(eval_tokenized)
    tokenizer_kwargs = {"is_split_into_words": True}
    model_pipeline = pipeline('token-classification', 
                                model_path, 
                                tokenizer = PRETRAINED_TOKENIZER, 
                                device = 0,
                                aggregation_strategy = "simple")
    # outputs = model_pipeline(tokenized_texts, **tokenizer_kwargs)
    outputs = [model_pipeline(example) for example in texts]
    # outputs = model_pipeline(KeyDataset(texts, key = 'text_words'))
    # data_collator = DataCollatorForTokenClassification(tokenizer=PRETRAINED_TOKENIZER)
    # outputs = model_pipeline(texts, batch_size = 8)

    predictions = []
    for output in outputs:
        p = []
        print(output)
        for o in output:
            word = ''.join([w['word'] for w in o])
            entities = [w['entity'] for w in o]
            unique = list(set(entities))
            counts = {
                entity: entities.count(entity) for entity in unique
            }
            max_count = 0
            final_entity = ''
            for entity in counts:
                if counts[entity] > max_count:
                    final_entity = entity
                    max_count = counts[entity]

            p.append((word, final_entity))
        predictions.append(p)
        # label_scores = [(label_score['label'], label_score['score']) for label_score in output]
        # predictions.append(max(label_scores, key=lambda label_score: label_score[1]))

    return predictions


model_path = 'models/ArgCompTagger'
dataset_path = 'datasets/processed/AnnotatedCMV/annotated_arg_comps.json'

with open(dataset_path, 'r') as infile:
    data = json.load(infile)

eval_texts = [d['text_words'] for d in data]
eval_labels = [d['arg_comp_tags'] for d in data]

outputs = predict(model_path, eval_texts, 5)
predictions = [[word_output[1] for word_output in output] for output in outputs]

with open('arg_comp_predictions.pickle', 'wb') as outfile:
    pickle.dump(predictions, outfile)

# with open('arg_comp_predictions.pickle','rb') as infile:
#     predictions = pickle.load(infile)

label2id =  {k: v for v, k in enumerate(ARG_COMP_TAGS)}
label2id[''] = 0
predictions = [[label2id[p] for p in prediction] for prediction in predictions]

#Compute metrics for each list in the list of predictions and average. 
metric = evaluate.load('f1')
f1_scores = [metric.compute(predictions = predictions[i], references = eval_labels[i], average = "macro")['f1'] for i in range(0, len(predictions))]
print(np.mean(f1_scores), np.median(f1_scores), np.max(f1_scores), np.min(f1_scores))



# model_path = 'models/CombinedTagger'
# dataset_path = 'datasets/processed/AnnotatedCMV/annotated_arg_comps.json'

# with open(dataset_path, 'r') as infile:
#     data = json.load(infile)

# eval_texts = [d['text_words'] for d in data]
# eval_labels = [d['combined_tags'] for d in data]

# outputs = predict(model_path, eval_texts, 5)
# predictions = [[word_output[1] for word_output in output] for output in outputs]

# with open('combined_predictions.pickle', 'wb') as outfile:
#     pickle.dump(predictions, outfile)

# # with open('arg_comp_predictions.pickle','rb') as infile:
# #     predictions = pickle.load(infile)

# label2id =  {k: v for v, k in enumerate(COMBINED_TAGS)}
# label2id[''] = 0
# predictions = [[label2id[p] for p in prediction] for prediction in predictions]

# #Compute metrics for each list in the list of predictions and average. 
# metric = evaluate.load('f1')
# f1_scores = [metric.compute(predictions = predictions[i], references = eval_labels[i], average = "macro")['f1'] for i in range(0, len(predictions))]
# print(np.mean(f1_scores), np.median(f1_scores), np.max(f1_scores), np.min(f1_scores))