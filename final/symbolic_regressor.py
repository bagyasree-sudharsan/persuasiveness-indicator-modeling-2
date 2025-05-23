import json
from common import train_test_split, add_tags_to_text, TextDataset, get_tokenizer
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments


def symbolic_regressor(train_data_path, training_output_file, model_file, split_ratio= 0.95, use_sem_types = True):
    with open(train_data_path, 'r') as infile:
        data = json.load(infile)

    text_tuples = [
        (
        d['text'],
        d['score'],
        d['arg_comps'],
        d['sem_types'],
        d['text_segments']) for d in data if (isinstance(d['text'], str))
    ]
    train_tuples, test_tuples = train_test_split(split_ratio, text_tuples)
    train_texts, train_labels, train_arg_comps, train_sem_types, train_text_segments = zip(*train_tuples)
    test_texts, test_labels, test_arg_comps, test_sem_types, test_text_segments = zip(*test_tuples)
    print('Data read and split.')

    train_texts_with_tags = add_tags_to_text(train_text_segments, train_arg_comps, train_sem_types, use_sem_types)
    test_texts_with_tags = add_tags_to_text(test_text_segments, test_arg_comps, test_sem_types, use_sem_types)

    symbolic = 'sem_types' if use_sem_types else 'arg_comps'
    tokenizer = get_tokenizer(symbolic)
    train_tokenized = tokenizer(train_texts_with_tags, truncation = True, padding = "max_length")
    test_tokenized = tokenizer(test_texts_with_tags, truncation = True, padding = "max_length")

    print('Data tokenized.')

    # Creating datasets in the form required by Trainer
    train_dataset = TextDataset(train_tokenized, train_labels, labels_are_float = True)
    test_dataset = TextDataset(test_tokenized, test_labels, labels_are_float = True)

    print('Datasets created.')

    # Setting up the trainer

    training_args = TrainingArguments(output_dir="trainers/{}".format(training_output_file), 
                                    overwrite_output_dir = True,
                                    report_to ="none",
                                    num_train_epochs = 12,
                                    per_device_train_batch_size = 16,
                                    learning_rate = 3e-5,
                                    logging_strategy = "epoch")

    regression_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert/distilbert-base-cased',
        num_labels = 1
    )
    regression_model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(
        model=regression_model,
        args=training_args,
        train_dataset=train_dataset
    )


    # Training

    print('Starting training.')
    trainer.train()
    print('Finished training.')

    trainer.save_model('models/{}'.format(model_file))
    print('Saved model {}.'.format(model_file))

