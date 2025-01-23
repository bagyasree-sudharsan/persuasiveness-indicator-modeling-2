from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForPreTraining, Trainer, AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
import numpy as np
import random
from constants import TEXT_SEG_ARG_COMP_TAGS, SEMANTIC_TYPE_TAGS

class TextDataset(Dataset):
    def __init__(self, tokenized_texts, labels = None, labels_are_float = False):
        self.encodings = tokenized_texts
        if not labels_are_float:
          self.labels = labels
        else:
          self.labels = [float(label) for label in labels]

    def __len__(self):
       return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        # review = str(self.review[item])
        # review = " ".join(review.split())
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

class TextDataset2(Dataset):
    def __init__(self, tokenized_texts, labels = None, arg_comp_tags = False, semantic_type_tags = False, labels_are_float = False):
        self.encodings = tokenized_texts
        if not labels_are_float:
          self.labels = labels
        else:
          self.labels = [float(label) for label in labels]
        
        self.arg_comp_tags = tokenized_texts.arg_comp_tags if arg_comp_tags else None
        self.semantic_type_tags = tokenized_texts.semantic_type_tags if semantic_type_tags else None

    def __len__(self):
       return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        # if self.arg_comp_tags:
        #     item['arg_comp_tags'] = torch.tensor(self.arg_comp_tags[idx])
        # if self.semantic_type_tags:
        #     item['semantic_type_tags'] = torch.tensor(self.arg_comp_tags[idx])
        return item

class BertRegression(PreTrainedModel):
  def __init__(self, model_name, tokenizer = None):
      super(BertRegression, self).__init__(AutoConfig.from_pretrained(model_name, output_attention = True, output_hidden_state = True))
      # super(BertRegression, self).__init__()
      # self.config = AutoConfig.from_pretrained(model_name, output_attention = True, output_hidden_state = True)
      # self.bert = AutoModel.from_pretrained(model_name, config = self.config)

      self.config = AutoConfig.from_pretrained(model_name, output_attention = True, output_hidden_state = True)
      self.bert = AutoModel.from_pretrained(model_name, config = self.config)
      if tokenizer:
        self.bert.resize_token_embeddings(len(tokenizer))
      self.regression_dropout = torch.nn.Dropout(0.1)
      self.regressor = torch.nn.Linear(self.config.hidden_size, 1)
      self.num_labels = 1

  def forward(self, input_ids, attention_mask, labels=None):
      pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
      output = self.regressor(pooled_output)
      output = self.regression_dropout(output)

      loss_func = torch.nn.MSELoss()
      # loss = loss_func(output, labels) if self.training else torch.empty(1)
      loss = loss_func(output, labels)
      return {
              'loss': loss,
              'output': output
      }

class RegressionTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
      # print('IN COMPUTE LOSS------------------')
      labels = inputs.pop("labels")
      output = model(labels = labels, **inputs)
      # loss = output['loss']
      # outputs = output['output']
      loss = output.loss
      logits = output.logits
      loss_fct = torch.nn.MSELoss()
      loss = loss_fct(logits.view(-1), labels.view(-1))
      # loss, outputs = output
      # print('loss: ', loss)
      # print('outputs: ', outputs)
      return (loss, logits) if return_outputs else loss
  
def compute_metrics_for_regression(eval_pred):
  print('IN METRICS FOR REGRESSION--------------------')
  logits, labels = eval_pred
  mse = mean_squared_error(labels, logits)
  mae = mean_absolute_error(labels, logits)
  r2 = r2_score(labels, logits)
  single_squared_errors = ((logits - labels).flatten()**2).tolist()
  
  # Compute accuracy 
  # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
  accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
  
  return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}


PRETRAINED_TOKENIZER = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
TOKENIZER_KWARGS = {"padding": "max_length", "truncation": True}
# PRETRAINED_MODEL = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased", num_labels=3).to('cuda')
# PRETRAINED_MODEL_REGRESSION = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased", num_labels=1).to('cuda')


def create_custom_tokenizer():
  new_tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
  additional_special_tokens = ['[{}]'.format(tag.upper()) for tag in TEXT_SEG_ARG_COMP_TAGS]
  additional_special_tokens.extend(['[{}]'.format(tag.upper()) for tag in SEMANTIC_TYPE_TAGS])
  additional_special_tokens = list(set(additional_special_tokens))
  special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
  new_tokenizer.add_special_tokens(special_tokens_dict)
  return new_tokenizer

CUSTOM_TOKENIZER = create_custom_tokenizer()

