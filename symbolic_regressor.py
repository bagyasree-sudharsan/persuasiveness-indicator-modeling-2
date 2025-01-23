import json
from transformers import TrainingArguments, Trainer, AutoModel, DistilBertConfig, DistilBertModel, AutoConfig, AutoTokenizer
import random
import numpy as np
import torch
import evaluate
from globals import TextDataset, BertRegression, RegressionTrainer, CUSTOM_TOKENIZER
from common_functions import train_test_split, add_tags_to_text
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from split_arguments import split_text
from constants import TEXT_SEG_ARG_COMP_TAGS, SEMANTIC_TYPE_TAGS


def symbolic_regressor(texts, labels, split_ratio, training_output_file, model_file):
    train_texts, train_labels, test_texts, test_labels = train_test_split(split_ratio, texts, labels)
    print('Data read and split.')

        
    # Tokenizing data
    id2label_argcomps = {i: v for i, v in enumerate(TEXT_SEG_ARG_COMP_TAGS)}
    id2label_semtypes = {i: v for i, v in enumerate(SEMANTIC_TYPE_TAGS)}
    train_texts_with_tags = add_tags_to_text(train_texts, id2label_argcomps, id2label_semtypes)
    test_texts_with_tags = add_tags_to_text(test_texts, id2label_argcomps, id2label_semtypes)

    train_tokenized = CUSTOM_TOKENIZER(train_texts_with_tags, truncation = True, padding = "max_length")
    test_tokenized = CUSTOM_TOKENIZER(test_texts_with_tags, truncation = True, padding = "max_length")

    print('Data tokenized.')

    # Creating datasets in the form required by Trainer
    train_dataset = TextDataset(train_tokenized, train_labels, labels_are_float = True)
    test_dataset = TextDataset(test_tokenized, test_labels, labels_are_float = True)

    print('Datasets created.')

    # Setting up the trainer

    training_args = TrainingArguments(output_dir="trainers/{}".format(training_output_file), 
                                    overwrite_output_dir = True,
                                    eval_strategy="epoch", 
                                    report_to ="none",
                                    num_train_epochs = 5,
                                    per_device_train_batch_size = 32,
                                    per_device_eval_batch_size = 16,
                                    learning_rate = 1e-5)

    regression_model = BertRegression('distilbert/distilbert-base-cased', tokenizer = CUSTOM_TOKENIZER)

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

    trainer.save_model('models/{}'.format(model_file))
    print('Saved model {}.'.format(model_file))

