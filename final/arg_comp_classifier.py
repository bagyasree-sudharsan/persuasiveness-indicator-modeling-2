import json
from transformers import TrainingArguments, Trainer, DistilBertTokenizer, DistilBertForSequenceClassification
import random
import numpy as np
import torch
import evaluate
try:
    from final.common import train_test_split, TextDataset
except ImportError:
    from common import train_test_split, TextDataset
from sklearn.metrics import confusion_matrix

#Reading the data

with open('datasets/processed/AnnotatedCMV/annotated_arg_comps.json', 'r') as infile:
    data = json.load(infile)

texts = [segment for d in data for segment in d['text_segments']]
labels = [tag for d in data for tag in d['text_seg_arg_comp_tags']]
text_segment_tuples = list(zip(texts, labels))

train_tuples, test_tuples = train_test_split(0.85, text_segment_tuples)
train_texts, train_labels = zip(*train_tuples)
test_texts, test_labels = zip(*test_tuples)

print('Data read and split.')

'''
BERT - Classifier
'''

def compute_metrics(eval_pred):
    metric = evaluate.load('f1')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(confusion_matrix(predictions, labels))
    return metric.compute(predictions=predictions, references=labels, average = "macro")
    
# Tokenizing data
tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-cased")
train_tokenized = tokenizer(train_texts, truncation = True, padding = "max_length")
test_tokenized = tokenizer(test_texts, truncation = True, padding = "max_length")

print('Data tokenized.')

# Creating datasets in the form required by Trainer
train_dataset = TextDataset(train_tokenized, train_labels)
test_dataset = TextDataset(test_tokenized, test_labels)

print('Datasets created.')

model = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased", num_labels=3).to('cuda')

#Setting up the trainer

training_args = TrainingArguments(output_dir="trainers/arg_comp_trainer", 
                                  report_to ="none",
                                  num_train_epochs = 3,
                                  per_device_train_batch_size = 8,
                                  logging_strategy = 'epoch',
                                  eval_strategy = 'epoch',
                                  learning_rate = 3e-5)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset = test_dataset,
    compute_metrics=compute_metrics,
)

# Training

print('Starting training.')
trainer.train()
print('Finished training.')

trainer.save_model('models/ArgCompClassifier')
print('Saved model ArgCompClassifier.')