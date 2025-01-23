import json
from transformers import TrainingArguments, Trainer, AutoModel, DistilBertConfig, DistilBertModel, AutoConfig
import random
import numpy as np
import torch
import evaluate
import numpy as np
from globals import TextDataset, PRETRAINED_TOKENIZER, TOKENIZER_KWARGS, PRETRAINED_MODEL_REGRESSION, train_test_split, BertRegression, RegressionTrainer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

'''
Here, 'label' refers to the score of the text.
'''

#Reading the data

with open('datasets/processed/CMV/final.json', 'r') as infile:
    data = json.load(infile)

texts = [d['text'] for d in data if isinstance(d['text'], str)]
labels = [d['score'] for d in data if isinstance(d['text'], str)]

train_texts, train_labels, test_texts, test_labels = train_test_split(0.85, texts, labels)

print('Data read and split.')

'''
BERT - Regression
'''

# Tokenizing data
train_tokenized = PRETRAINED_TOKENIZER(train_texts, **TOKENIZER_KWARGS)
test_tokenized = PRETRAINED_TOKENIZER(test_texts, **TOKENIZER_KWARGS)

print('Data tokenized.')

# Creating datasets in the form required by Trainer
train_dataset = TextDataset(train_tokenized, train_labels, labels_are_float = True)
test_dataset = TextDataset(test_tokenized, test_labels, labels_are_float = True)

print('Datasets created.')

#Setting up the trainer

training_args = TrainingArguments(output_dir="trainers/test_trainer_regressor", 
                                  overwrite_output_dir = True,
                                  eval_strategy="epoch", 
                                  report_to ="none",
                                  num_train_epochs = 5,
                                  per_device_train_batch_size = 32,
                                  per_device_eval_batch_size = 16,
                                  learning_rate = 1e-5)

regression_model = BertRegression('distilbert/distilbert-base-cased')

trainer = RegressionTrainer(
    model=regression_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Training

print('Starting training.')
trainer.train()
print('Finished training.')

trainer.save_model('models/CMV_Regressor')
print('Saved model CMV_Regressor.')

