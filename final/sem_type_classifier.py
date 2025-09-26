import json
from transformers import TrainingArguments, Trainer, DistilBertTokenizer, DistilBertForSequenceClassification
import random
import numpy as np
import torch
import evaluate
from common import train_test_split, TextDataset
from sklearn.metrics import confusion_matrix

def compute_metrics(eval_pred):
    metric = evaluate.load('f1')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(confusion_matrix(predictions, labels))
    return metric.compute(predictions=predictions, references=labels, average = "micro")

def train_model(text_segment_tuples, train_args_file, pretrained_model, model_name, num_epochs, batch_size):
    # Tokenizing data
    train_tuples, test_tuples = train_test_split(0.85, text_segment_tuples)
    train_texts, train_labels = zip(*train_tuples)
    test_texts, test_labels = zip(*test_tuples)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    train_tokenized = tokenizer(train_texts, truncation = True, padding = "max_length")
    test_tokenized = tokenizer(test_texts, truncation = True, padding = "max_length")

    print('Data tokenized.')

    # Creating datasets in the form required by Trainer
    train_dataset = TextDataset(train_tokenized, train_labels)
    test_dataset = TextDataset(test_tokenized, test_labels)

    print('Datasets created.')

    #Setting up the trainer

    training_args = TrainingArguments(output_dir="trainers/{}".format(train_args_file), 
                                    report_to ="none",
                                    num_train_epochs = num_epochs,
                                    per_device_train_batch_size = batch_size,
                                    logging_strategy = 'epoch',
                                    eval_strategy = 'epoch', 
                                    learning_rate = 3e-5)


    trainer = Trainer(
        model=pretrained_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = test_dataset,
        compute_metrics=compute_metrics,
    )

    # Training

    print('Starting training.')
    trainer.train()
    print('Finished training.')

    trainer.save_model('models/{}'.format(model_name))
    print('Saved model {}.'.format(model_name))


with open('datasets/processed/AnnotatedCMV/annotated_arg_comps.json', 'r') as infile:
    data = json.load(infile)

texts = [segment for d in data for segment in d['text_segments']]
labels = [tag for d in data for tag in d['text_seg_sem_type_tags']]
arg_comps = [tag for d in data for tag in d['text_seg_arg_comp_tags']]
text_segment_tuples = list(zip(texts, labels, arg_comps))
claim_tuples = [(t[0], t[1] - 8) for t in text_segment_tuples if t[2] == 1]
premise_tuples = [(t[0], t[1] - 1) for t in text_segment_tuples if t[2] == 2]

pretrained_model_claims = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased", num_labels=5).to('cuda')
train_model(claim_tuples, 'claim_classifier_training', pretrained_model_claims, 'ClaimClassifier', num_epochs = 5, batch_size = 8)

pretrained_model_premises = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-cased", num_labels=7).to('cuda')
train_model(premise_tuples, 'premise_classifier_training', pretrained_model_premises, 'PremiseClassifier', num_epochs = 8, batch_size = 8)