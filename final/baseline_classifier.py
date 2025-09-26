import json
try:
    from final.common import TextDataset, train_test_split
except ImportError:
    from common import TextDataset, train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, TrainingArguments, Trainer
import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_metrics(eval_pred):
    metric = evaluate.load('f1')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(confusion_matrix(predictions, labels))
    return metric.compute(predictions=predictions, references=labels, average = "macro")

def baseline_classifier(file_path, split_ratio, training_output_file, model_file):
    with open(file_path, 'r') as infile:
        data = json.load(infile)
    
    text_tuples = [(
        d['text'],
        d['is_successful'] 
    ) for d in data if (isinstance(d['text'], str))]
    
    train_tuples, test_tuples = train_test_split(split_ratio, text_tuples)
    train_texts, train_labels = zip(*train_tuples)
    test_texts, test_labels = zip(*test_tuples)


    print('Data read and split.')

    # Tokenizing data
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    train_tokenized = tokenizer(train_texts, truncation = True, padding = "max_length")
    test_tokenized = tokenizer(test_texts, truncation = True, padding = "max_length")

    print('Data tokenized.')

    # Creating datasets in the form required by Trainer
    train_dataset = TextDataset(train_tokenized, train_labels, labels_are_float = False)
    test_dataset = TextDataset(test_tokenized, test_labels, labels_are_float = False)

    print('Datasets created.')

    #Setting up the trainer

    training_args = TrainingArguments(output_dir="trainers/{}".format(training_output_file), 
                                    overwrite_output_dir = True,
                                    report_to ="none",
                                    num_train_epochs = 3,
                                    per_device_train_batch_size = 16,
                                    learning_rate = 3e-5,
                                    logging_strategy = "epoch",
                                    eval_strategy = "epoch")

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert/distilbert-base-cased',
        num_labels = 3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = test_dataset,
        compute_metrics=compute_metrics
    )

    # Training

    print('Starting training.')
    trainer.train()
    print('Finished training.')

    trainer.save_model('models/{}'.format(model_file))
    print('Saved model {}.'.format(model_file))
