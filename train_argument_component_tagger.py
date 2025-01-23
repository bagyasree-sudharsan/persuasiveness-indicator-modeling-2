from globals import PRETRAINED_TOKENIZER, TextDataset, train_test_split
from constants import ARG_COMP_TAGS
import json
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

with open('datasets/processed/AnnotatedCMV/annotated_arg_comps.json', 'r') as infile:
    dataset = json.load(infile)

texts = [d['text_words'] for d in dataset]
labels = [d['arg_comp_tags'] for d in dataset]
train_texts, train_labels, test_texts, test_labels = train_test_split(0.85, texts, labels)

print('Data read and split.')

def tokenize_and_align_labels(token_list, token_tag_list, tokenizer):
    tokenized_inputs = tokenizer(token_list, truncation=True, is_split_into_words=True)

    labels = []
    for i, label_list in enumerate(token_tag_list):
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
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


train_tokenized = tokenize_and_align_labels(train_texts, train_labels, PRETRAINED_TOKENIZER)
test_tokenized = tokenize_and_align_labels(test_texts, test_labels, PRETRAINED_TOKENIZER)
print('Tokenized and labels aligned.')

train_dataset = TextDataset(train_tokenized, train_tokenized['labels'])
test_dataset = TextDataset(test_tokenized, test_tokenized['labels'])

print('Created datasets.')

data_collator = DataCollatorForTokenClassification(tokenizer=PRETRAINED_TOKENIZER)
seqeval = evaluate.load("seqeval")

labels = [d['arg_comp_tags'] for d in dataset]
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [ARG_COMP_TAGS[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ARG_COMP_TAGS[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

label2id =  {k: v for v, k in enumerate(ARG_COMP_TAGS)}
id2label =  {v: k for v, k in enumerate(ARG_COMP_TAGS)}

model = AutoModelForTokenClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=5, id2label=id2label, label2id=label2id).to('cuda')

print('Loaded model.')

training_args = TrainingArguments(
    output_dir="trainers/arg_comp_trainer",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to ="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=PRETRAINED_TOKENIZER,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


print('Starting training.')
trainer.train()
print('Finished training.')

trainer.save_model('models/ArgCompTagger')
print('Saved model ArgCompTagger.')
