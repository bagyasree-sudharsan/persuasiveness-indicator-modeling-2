import json
from transformers import TrainingArguments, Trainer, AutoModel, DistilBertConfig, DistilBertModel, AutoConfig
import random
import numpy as np
import torch
import evaluate
import numpy as np
from globals import TextDataset2, PRETRAINED_TOKENIZER, TOKENIZER_KWARGS, PRETRAINED_MODEL_REGRESSION, train_test_split, BertRegression, RegressionTrainer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

'''
Here, 'label' refers to the score of the text.
'''

#Reading the data

with open('datasets/processed/CMV/final.json', 'r') as infile:
    data = json.load(infile)

with open('datasets/processed/AnnotatedCMV/annotated_arg_comps.json', 'r') as infile:
    arg_comp_annotations = json.load(infile)

# data_ids = [d['id'] for d in data]
# aca_ids = [a['id'] for a in arg_comp_annotations]
# aca_ids_modified = []

usable_data = []

for d in data:
    for a in arg_comp_annotations:
        if a['id'] in d['id']:
            d['text_words'] = a['text_words']
            d['arg_comp_tags'] = a['arg_comp_tags']
            d['semantic_type_tags'] = a['semantic_type_tags']
            d['combined_tags'] = a['combined_tags']
            usable_data.append(d)


texts = [d['text_words'] for d in usable_data if isinstance(d['text'], str)]
arg_comp_tags = [d['arg_comp_tags'] for d in usable_data if isinstance(d['text'], str)]
semantic_type_tags = [d['semantic_type_tags'] for d in usable_data if isinstance(d['text'], str)]
labels = [d['score'] for d in usable_data if isinstance(d['text'], str)]

train_texts = texts
train_arg_comp_tags = arg_comp_tags
train_semantic_type_tags = semantic_type_tags
train_labels = labels

# train_texts, train_labels, test_texts, test_labels = train_test_split(0.85, texts, labels)

print('Data read and split.')

# '''
# BERT - Regression
# '''
def tokenize_and_align_labels(token_list, arg_comp_tag_list, semantic_tag_list, tokenizer):
    tokenized_inputs = tokenizer(token_list, truncation=True, is_split_into_words=True, padding = "max_length")
    
    labels = []
    for i, label_list in enumerate(arg_comp_tag_list):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label_list[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        label_ids = label_ids[:512]
        if len(label_ids) < 512:
            num = 512-len(label_ids)
            labels_ids.extend([-100] * num)
        labels.append(label_ids)
        
    tokenized_inputs["arg_comp_tags"] = labels

    labels = []
    for i, label_list in enumerate(semantic_tag_list):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label_list[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        label_ids = label_ids[:512]
        if len(label_ids) < 512:
            num = 512-len(label_ids)
            labels_ids.extend([-100] * num)
        labels.append(label_ids)
    tokenized_inputs["semantic_type_tags"] = labels

    return tokenized_inputs


# Tokenizing data
train_tokenized = tokenize_and_align_labels(train_texts, train_arg_comp_tags, train_semantic_type_tags, PRETRAINED_TOKENIZER)
# train_texts = [' '.join(text) for text in train_texts]
# train_tokenized = PRETRAINED_TOKENIZER(train_texts, **TOKENIZER_KWARGS)
print(len(train_tokenized.semantic_type_tags))

# test_tokenized = PRETRAINED_TOKENIZER(test_texts, **TOKENIZER_KWARGS)

print('Data tokenized.')

# # Creating datasets in the form required by Trainer
train_dataset = TextDataset2(train_tokenized, train_labels, arg_comp_tags = True, semantic_type_tags = True, labels_are_float = True)
# test_dataset = TextDataset(test_tokenized, test_labels, labels_are_float = True)

print('Datasets created.')

class BertRegression(torch.nn.Module):
  def __init__(self, model_name):
      super(BertRegression, self).__init__()
      self.config = AutoConfig.from_pretrained(model_name, output_attention = True, output_hidden_state = True)
      self.bert = AutoModel.from_pretrained(model_name, config = self.config)
      self.regression_dropout = torch.nn.Dropout(0.1)
      self.regressor = torch.nn.Linear(self.config.hidden_size * 3, 1)
      self.num_labels = 1
      self.arg_comp_tags = torch.nn.Embedding(5, self.config.hidden_size)
      self.semantic_type_tags = torch.nn.Embedding(13, self.config.hidden_size)

  def forward(self, input_ids, attention_mask, labels=None, arg_comp_tags = None, semantic_type_tags = None):
      pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
      embedded_arg_comp_tags = self.arg_comp_tags(arg_comp_tags)
      embedded_semantic_type_tags = self.semantic_type_tags(semantic_type_tags)
      cls_token_embedding = pooled_output[1]
      print(cls_token_embedding)
      concatenated_embeddings = torch.cat((cls_token_embedding, embedded_arg_comp_tags, embedded_semantic_type_tags), dim = 1)
      output = self.regressor(concatenated_embeddings)
      output = self.regression_dropout(output)

      loss_func = torch.nn.MSELoss()
      loss = loss_func(output, labels) if self.training else torch.empty(1)
      return {
              'loss': loss,
              'output': output
      }

regression_model = BertRegression('distilbert/distilbert-base-cased')

# #Setting up the trainer

training_args = TrainingArguments(output_dir="trainers/test_trainer_bert_arg_comps", 
                                  overwrite_output_dir = True,
                                  eval_strategy="no", 
                                  report_to ="none",
                                  num_train_epochs = 4,
                                  per_device_train_batch_size = 16)



trainer = RegressionTrainer(
    model=regression_model,
    args=training_args,
    train_dataset=train_dataset,
)

# # Training

print('Starting training.')
trainer.train()
print('Finished training.')

trainer.save_model('models/BERT_with_ArgComps')
print('Saved model BERT_with_ArgComps.')


# Try passing (1) just the features (2) the probabilities of the features (3) the features without the words
# Try passing the info (1) to the tokenizer (2) as a separate embedding to the model (3) arg_comps as tokens and semantic types as embeddings