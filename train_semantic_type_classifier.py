import json
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
import random
import numpy as np
import torch
import evaluate
import numpy as np
from globals import TextDataset, PRETRAINED_TOKENIZER, TOKENIZER_KWARGS
from common_functions import train_test_split

#Reading the data

with open('datasets/processed/AnnotatedCMV/annotated_arg_comps.json', 'r') as infile:
    data = json.load(infile)

texts = [segment for d in data for segment in d['text_segments']]
labels = [tag for d in data for tag in d['text_seg_sem_type_tags']]
arg_comps = [tag for d in data for tag in d['text_seg_arg_comp_tags']]

# train_texts, train_labels, test_texts, test_labels = train_test_split(0.85, texts, labels)
claim_texts = []
claim_labels = []
premise_texts = []
premise_labels = []
for i in range(0, len(texts)):
    if arg_comps[i] == 1:
        claim_texts.append(texts[i])
        claim_labels.append(labels[i])
    elif arg_comps[i] == 2:
        premise_texts.append(texts[i])
        premise_labels.append(labels[i])

claim_labels = [label - 8 for label in claim_labels]
premise_labels = [label - 1 for label in premise_labels]

print('Data read and split.')

'''
BERT - Classifier
'''

def compute_metrics(eval_pred):
    metric = evaluate.load('f1')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average = "macro")

def train_model(texts, labels, train_args_file, pretrained_model, model_name):
    train_texts, train_labels, test_texts, test_labels = train_test_split(0.85, texts, labels)

    train_tokenized = PRETRAINED_TOKENIZER(train_texts, **TOKENIZER_KWARGS)
    test_tokenized = PRETRAINED_TOKENIZER(test_texts, **TOKENIZER_KWARGS)

    train_dataset = TextDataset(train_tokenized, train_labels)
    test_dataset = TextDataset(test_tokenized, test_labels)

    training_args = TrainingArguments(output_dir="trainers/{}".format(train_args_file),   
                                        eval_strategy="epoch", 
                                        report_to ="none",
                                        num_train_epochs = 4,
                                        per_device_train_batch_size = 16)

    #Setting up the trainer

    trainer = Trainer(
        model=pretrained_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Training

    print('Starting training.')
    trainer.train()
    print('Finished training.')

    trainer.save_model('models/{}'.format(model_name))
    print('Saved model {}.'.format(model_name))


pretrained_model_claims = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased", num_labels=5).to('cuda')
train_model(claim_texts, claim_labels, 'claim_classifier_training', pretrained_model_claims, 'ClaimClassifier')

pretrained_model_premises = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased", num_labels=7).to('cuda')
train_model(premise_texts, premise_labels, 'premise_classifier_training', pretrained_model_premises, 'PremiseClassifier')