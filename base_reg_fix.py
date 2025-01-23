import json
from transformers import TrainingArguments, Trainer, AutoModel, DistilBertConfig, DistilBertModel, DistilBertForSequenceClassification, AutoConfig, PreTrainedModel, AutoModelForSequenceClassification, DistilBertPreTrainedModel
import random
import numpy as np
import torch
import evaluate
import numpy as np
from globals import TextDataset, PRETRAINED_TOKENIZER, TOKENIZER_KWARGS, RegressionTrainer
from common_functions import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

'''
Here, 'label' refers to the score of the text.
'''

#Reading the data

with open('datasets/processed/CMV/final.json', 'r') as infile:
    data = json.load(infile)

successful = [(d['text'], d['score']) for d in data if (isinstance(d['text'], str) and d['is_successful'] == 1)]
unsuccessful = [(d['text'], d['score']) for d in data if (isinstance(d['text'], str) and d['is_successful'] == 0)]
neutral = [(d['text'], d['score']) for d in data if (isinstance(d['text'], str) and d['is_successful'] == 2)]

random.shuffle(successful)
random.shuffle(unsuccessful)
random.shuffle(neutral)

data_tuples = successful[:700] + unsuccessful[:700] + neutral[:700]
random.shuffle(data_tuples)

texts = [d[0] for d in data_tuples]
labels = [d[1] for d in data_tuples]

# texts = [d['text'] for d in data if isinstance(d['text'], str)]
# labels = [d['score'] for d in data if isinstance(d['text'], str)]

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

training_args = TrainingArguments(output_dir="trainers/baseline_regressor_fixed", 
                                  overwrite_output_dir = True,
                                  report_to ="none",
                                  num_train_epochs = 3,
                                  per_device_train_batch_size = 16,
                                  learning_rate = 3e-5,
                                  logging_strategy = "epoch")

# class BertRegression(torch.nn.Module):
#     def __init__(self, model_name='distilbert-base-cased', num_features=1, dropout_rate=0.1):
#         super(BertRegression, self).__init__()
#         self.config = DistilBertConfig.from_pretrained(model_name)
#         self.bert = DistilBertModel.from_pretrained(model_name)
#         self.dropout = torch.nn.Dropout(dropout_rate)
#         self.regressor = torch.nn.Linear(self.config.hidden_size, num_features)
        
#         if torch.backends.mps.is_available():
#             self.device = torch.device("mps")
#         else:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         self.to(self.device)
    
#     def forward(self, input_ids, attention_mask, labels = None):
#         print('IN FORWARD----------------')
#         outputs = self.bert(input_ids, attention_mask=attention_mask)
#         print('AFTER GETTING BERT OUTPUT-------------------------------')
#         # print('OUTPUTS: ', outputs)
#         pooled_output = outputs.last_hidden_state  # Use the pooled output (the first token's hidden state)
#         pooled_output = pooled_output[:,0,:].view(-1,768)
#         print('pooled output: ', pooled_output.size())
#         x = self.dropout(pooled_output)
#         x = self.regressor(x)
#         print('labels: ', labels)
#         print('x: ', x)
#         loss_fct = torch.nn.MSELoss()
#         loss = loss_fct(x, labels)
#         return {
#             'loss': loss, 
#             'output': x
#         }

# class BertRegression(DistilBertPreTrainedModel):
#     # def __init__(self, model_name = 'distilbert/distilbert-base-cased', tokenizer = None):
#     def __init__(self, config):
#         # super(BertRegression, self).__init__(AutoConfig.from_pretrained(model_name, output_attention = True, output_hidden_state = True))
#         super(BertRegression, self).__init__(config = config)

#         # self.config = AutoConfig.from_pretrained(model_name, output_attention = True, output_hidden_state = True)
#         self.config = DistilBertConfig()
#     #   self.bert = AutoModel.from_pretrained(model_name, config = self.config)
#         self.bert = DistilBertModel(config = self.config)
#         # if tokenizer:
#         #     self.bert.resize_token_embeddings(len(tokenizer))
#         self.regression_dropout = torch.nn.Dropout(0.1)
#     #   self.regressor1 = torch.nn.Linear(, self.config.hidden_size)
#         self.regressor = torch.nn.Linear(self.config.hidden_size, 1)
#         self.num_labels = 1

#     def forward(self, input_ids, attention_mask, labels=None):
#         pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         #   pooled_output = pooled_output.last_hidden_state[:,0,:].view(-1,768)
#         pooled_output = pooled_output.last_hidden_state[:, 0]
#         print(pooled_output.size(), pooled_output)
#         output = self.regression_dropout(pooled_output)
#         #   output = self.regressor1(output)
#         #   output = self.regression_dropout(output)
#         output = self.regressor(output)
#         output = self.regression_dropout(output).squeeze()

#         loss_func = torch.nn.MSELoss()
#         # loss = loss_func(output, labels) if self.training else torch.empty(1)
#         loss = loss_func(output, labels)

#         print('labels: ', labels)
#         print('output: ', output)
#         print('loss: ', loss)
#         print
#         return {
#                 'loss': loss,
#                 'output': output
#         }


class BertRegression(DistilBertPreTrainedModel):
    def __init__(self, model_name = 'distilbert/distilbert-base-cased', tokenizer = None):
        super(BertRegression, self).__init__(AutoConfig.from_pretrained(model_name, output_attention = True, output_hidden_state = True))

        self.config = AutoConfig.from_pretrained(model_name, output_attention = True, output_hidden_state = True, num_labels = 1)
        # self.config = DistilBertConfig()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, config = self.config)
        # self.bert = DistilBertModel(config = self.config)
        if tokenizer:
            self.bert.resize_token_embeddings(len(tokenizer))
        self.regression_dropout = torch.nn.Dropout(0.1)
    #   self.regressor1 = torch.nn.Linear(, self.config.hidden_size)
        self.regressor = torch.nn.Linear(self.config.hidden_size, 1)
        self.num_labels = 1
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print(labels)
        # print(pooled_output.logits)
        # logits = self.regression_dropout(pooled_output.logits)
        # output_scores = self.regressor(logits)
        output_scores = pooled_output.logits
        logits = output_scores

        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(output_scores.view(-1), labels.view(-1))
        print(output_scores.view(-1), labels.view(-1))
        print('loss: ', loss)
        outputs = (loss, logits)
        return outputs  # (loss), logits, (hidden_states), (attentions)
        #   pooled_output = pooled_output.last_hidden_state[:,0,:].view(-1,768)
        # print(pooled_output)
        # print(pooled_output.logits.size())
        # pooled_output = pooled_output.logits[:, 0]
        # print(pooled_output.size(), pooled_output)
        # output = self.regression_dropout(pooled_output)
        # #   output = self.regressor1(output)
        # #   output = self.regression_dropout(output)
        # output = self.regressor(output)
        # output = self.regression_dropout(output).squeeze()

        # loss_func = torch.nn.MSELoss()
        # # loss = loss_func(output, labels) if self.training else torch.empty(1)
        # loss = loss_func(output, labels)

        # print('labels: ', labels)
        # print('output: ', output)
        # print('loss: ', loss)
        # print
        return {
                'loss': pooled_output.loss,
                'output': pooled_output.logits
        }

# regression_model = BertRegression()
regression_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert/distilbert-base-cased',
    num_labels = 1
).to('cuda')

trainer = Trainer(
    model=regression_model,
    args=training_args,
    train_dataset=train_dataset
)

# Training

print('Starting training.')
trainer.train()
print('Finished training.')

trainer.save_model('models/BaselineRegressorFixed')
print('Saved model BaselineRegressorFixed.')

# model = trainer.load('models/Regressor_Test')
# model.save_pretrained('models/Regressor_Test')
# model = regression_model.load_state_dict()
# model.save_pretrained('models/Regressor_Test')
