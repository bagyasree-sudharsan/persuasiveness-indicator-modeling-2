import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import random
import numpy as np
from torch.utils.data import Dataset
import torch
import evaluate
import numpy as np

def train_test_split(percentage, texts, labels):
  if len(texts) != len(labels):
    raise Exception('Number of texts and number of labels do not match.')
  else:
    num_train_examples = int(np.floor(len(texts) * percentage))
    train_indices = random.sample(range(0, len(texts)), num_train_examples)
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []

    for i in range(0, len(texts)):
      if i in train_indices:
        train_texts.append(texts[i])
        train_labels.append(labels[i])
      else:
        test_texts.append(texts[i])
        test_labels.append(labels[i])
    
    return train_texts, train_labels, test_texts, test_labels

#Reading the data

with open('datasets/processed/CMV/final.json', 'r') as infile:
    data = json.load(infile)

texts = []
labels = []

for d in data:
  if isinstance(d['text'], str):
    texts.append(d['text'])
    labels.append(d['is_successful'])

train_texts, train_labels, test_texts, test_labels = train_test_split(0.85, texts, labels)

print('Data read and split.')


'''
BERT - Classifier
'''

class TextDataset(Dataset):
    def __init__(self, tokenized_texts, labels):
        self.encodings = tokenized_texts
        self.labels = labels

    def __len__(self):
       return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(eval_pred):
    metric = evaluate.load('f1')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    
# Tokenizing data

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
train_tokenized = tokenizer(train_texts, padding="max_length", truncation = True)
test_tokenized = tokenizer(test_texts, padding = "max_length", truncation = True)

print('Data tokenized.')

# Creating datasets in the form required by Trainer
train_dataset = TextDataset(train_tokenized, train_labels)
test_dataset = TextDataset(test_tokenized, test_labels)

print('Datasets created.')

#Loading model and setting up the trainer

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased", num_labels=3).to('cuda')
print('Loaded model.')

training_args = TrainingArguments(output_dir="trainers/test_trainer", 
                                  eval_strategy="epoch", 
                                  report_to ="none",
                                  num_train_epochs = 5)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Training

print('Starting training.')
trainer.train()
print('Finished training.')

trainer.save_model('models/CMV_Classifier')
print('Saved model CMV_Classifier.')