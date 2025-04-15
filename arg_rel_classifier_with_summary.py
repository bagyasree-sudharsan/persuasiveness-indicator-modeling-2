import json
# from common import add_tags_to_text, TextDataset, get_tokenizer
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from ollama import chat
from ollama import ChatResponse

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
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    
def compute_metrics(eval_pred):
    metric = evaluate.load('f1')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(confusion_matrix(predictions, labels))
    return metric.compute(predictions=predictions, references=labels, average = "macro")

def get_summaries(claims, premises):
    texts = []
    for i in range(0, len(claims)):
        claim = claims[i]
        premise = premises[i]
        question = 'I will show you two pieces of text, one marked "Claim" and one marked "Premise." \
        You must summarize these in 350 words or less. Ensure that it is still clear which part is the claim and which the premise. \
        Claim: {claim} \n Premise: {premise}'
        response: ChatResponse = chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': question,
        },
        ])
        summary = response.message.content
        texts.append(summary)
    
    return texts

def arg_rel_classifier(train_data_path, training_output_file, model_file, split_ratio= 0.85):
    with open(train_data_path, 'r') as infile:
        data = json.load(infile)

    #Training by just providing the claim as additional prior context.
    text_tuples = [
        (
        d['claim'], 
        d['premise'],
        d['relation']
       ) for d in data
    ]
    train_tuples, test_tuples = train_test_split(text_tuples, train_size = split_ratio, shuffle = True)
    train_claims, train_premises, train_labels = zip(*train_tuples)
    test_claims, test_premises, test_labels = zip(*test_tuples)
    # train_texts, train_labels, train_arg_comps, train_sem_types, train_text_segments = zip(*train_tuples)
    # test_texts, test_labels, test_arg_comps, test_sem_types, test_text_segments = zip(*test_tuples)
    print('Data read and split.')
    
    train_texts = get_summaries(train_claims, train_premises)
    test_texts = get_summaries(test_claims, test_premises)
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    train_tokenized = tokenizer(train_texts, truncation = True, padding = "max_length")
    test_tokenized = tokenizer(test_texts, truncation = True, padding = "max_length")

    print('Data tokenized.')

    # Creating datasets in the form required by Trainer
    train_dataset = TextDataset(train_tokenized, train_labels, labels_are_float = False)
    test_dataset = TextDataset(test_tokenized, test_labels, labels_are_float = False)

    print('Datasets created.')

    # # Setting up the trainer

    training_args = TrainingArguments(output_dir="trainers/{}".format(training_output_file), 
                                    overwrite_output_dir = True,
                                    report_to ="none",
                                    num_train_epochs = 5,
                                    per_device_train_batch_size = 8,
                                    learning_rate = 3e-5,
                                    logging_strategy = "epoch",
                                    eval_strategy = "epoch")

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert/distilbert-base-cased',
        num_labels = 3
    )
    # model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = test_dataset,
        compute_metrics=compute_metrics
    )


    # # Training

    print('Starting training.')
    trainer.train()
    print('Finished training.')

    trainer.save_model('models/{}'.format(model_file))
    print('Saved model {}.'.format(model_file))

arg_rel_classifier('datasets/processed/ArgRelData/arg_rel_data.json', 
                   training_output_file='trainers/ArgRelClassifierSummary', 
                   model_file = 'models/ArgRelClassifierSummary',
                   split_ratio=0.85)
