import json
from transformers import TrainingArguments, Trainer
import random
import numpy as np
import torch
import evaluate
import numpy as np
from globals import TextDataset, PRETRAINED_TOKENIZER, TOKENIZER_KWARGS, PRETRAINED_MODEL, train_test_split

#Reading the data

with open('datasets/processed/AnnotatedCMV/final.json', 'r') as infile:
    data = json.load(infile)

texts = [d['text'] for d in data if isinstance(d['text'], str)]
labels = [d['is_successful'] for d in data if isinstance(d['text'], str)]

train_texts, train_labels, test_texts, test_labels = train_test_split(0.85, texts, labels)

print('Data read and split.')

'''
BERT - Classifier
'''

def compute_metrics(eval_pred):
    metric = evaluate.load('f1')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average = "micro")
    
# Tokenizing data
train_tokenized = PRETRAINED_TOKENIZER(train_texts, **TOKENIZER_KWARGS)
test_tokenized = PRETRAINED_TOKENIZER(test_texts, **TOKENIZER_KWARGS)

print('Data tokenized.')

# Creating datasets in the form required by Trainer
train_dataset = TextDataset(train_tokenized, train_labels)
test_dataset = TextDataset(test_tokenized, test_labels)

print('Datasets created.')

#Setting up the trainer

training_args = TrainingArguments(output_dir="trainers/test_trainer", 
                                  eval_strategy="epoch", 
                                  report_to ="none",
                                  num_train_epochs = 4,
                                  per_device_train_batch_size = 16)
trainer = Trainer(
    model=PRETRAINED_MODEL,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Training

print('Starting training.')
trainer.train()
print('Finished training.')

trainer.save_model('models/AnnotatedCMV_Classifier')
print('Saved model AnnotatedCMV_Classifier.')