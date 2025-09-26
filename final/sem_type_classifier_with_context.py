import json
from transformers import TrainingArguments, Trainer, DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertModel
import random
import numpy as np
import torch
import evaluate
try:
    from final.common import train_test_split, TextDataset
except ImportError:
    from common import train_test_split, TextDataset
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
import torch.nn as nn

PAD_TOKEN = "[PAD]"

class HierarchicalTransformerDistilBERT(nn.Module):
    def __init__(self, num_labels=3, n_heads=4, n_layers=2, hidden_dim=768):
        super().__init__()
        # Base encoder for segments
        self.bert = DistilBertModel.from_pretrained("distilbert/distilbert-base-cased")
        self.hidden_dim = hidden_dim
        
        # Transformer encoder for [prev, focus, next] embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            batch_first=True,   # (batch, seq, dim)
            dim_feedforward=hidden_dim*4,
            dropout=0.1
        )
        self.context_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final classifier: we’ll just take the *focus embedding after context attention*
        self.fc = nn.Linear(hidden_dim, num_labels)

    def encode(self, input_ids, attn_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token (batch, hidden)

    def forward(self, prev_input_ids, prev_attn_mask,
                      focus_input_ids, focus_attn_mask,
                      next_input_ids, next_attn_mask,
                      labels=None):
        # Step 1: encode each segment separately
        prev_emb = self.encode(prev_input_ids, prev_attn_mask)   # (batch, hidden)
        focus_emb = self.encode(focus_input_ids, focus_attn_mask)
        next_emb = self.encode(next_input_ids, next_attn_mask)

        # Step 2: stack embeddings into sequence [prev, focus, next]
        seq = torch.stack([prev_emb, focus_emb, next_emb], dim=1)  # (batch, 3, hidden)

        # Step 3: let Transformer learn relationships between them
        seq_out = self.context_transformer(seq)  # (batch, 3, hidden)

        # Step 4: extract the transformed focus embedding (position 1)
        focus_ctx = seq_out[:, 1, :]  # (batch, hidden)

        # Step 5: classify
        logits = self.fc(focus_ctx)

        # Optional loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def build_triplets_from_labels(data, target_tag_value):
    triplets = []
    
    for d in data:
        segments = d['text_segments']
        sem_labels = d['text_seg_sem_type_tags']  # semantic type
        arg_labels = d['text_seg_arg_comp_tags']  # argument component type
        n = len(segments)
        
        for i in range(n):
            if arg_labels[i] != target_tag_value:
                continue  # skip segments not of this type
            
            # prev / focus / next
            prev_seg = segments[i-1] if i > 0 else PAD_TOKEN
            focus_seg = segments[i]
            next_seg = segments[i+1] if i < n-1 else PAD_TOKEN
            
            if target_tag_value == 1:  # claim
                label = sem_labels[i] - 8
            else:                     # premise
                label = sem_labels[i] - 1
            
            triplets.append((prev_seg, focus_seg, next_seg, label))
    
    return triplets

def collate_fn(batch):
    return {
        "prev_input_ids": torch.stack([x["prev_input_ids"] for x in batch]),
        "prev_attn_mask": torch.stack([x["prev_attn_mask"] for x in batch]),
        "focus_input_ids": torch.stack([x["focus_input_ids"] for x in batch]),
        "focus_attn_mask": torch.stack([x["focus_attn_mask"] for x in batch]),
        "next_input_ids": torch.stack([x["next_input_ids"] for x in batch]),
        "next_attn_mask": torch.stack([x["next_attn_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch])
    }

class TripleDataset(Dataset):
    def __init__(self, triples, tokenizer, max_len=128):
        self.triples = triples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        prev, focus, nxt, label = self.triples[idx]
        prev_enc = self.tokenizer(prev, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        focus_enc = self.tokenizer(focus, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        next_enc = self.tokenizer(nxt, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        
        return {
            "prev_input_ids": prev_enc["input_ids"].squeeze(),
            "prev_attn_mask": prev_enc["attention_mask"].squeeze(),
            "focus_input_ids": focus_enc["input_ids"].squeeze(),
            "focus_attn_mask": focus_enc["attention_mask"].squeeze(),
            "next_input_ids": next_enc["input_ids"].squeeze(),
            "next_attn_mask": next_enc["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    metric = evaluate.load('f1')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(confusion_matrix(predictions, labels))
    return metric.compute(predictions=predictions, references=labels, average = "micro")

def train_model(triples, train_args_file, pretrained_model, model_name, num_epochs, batch_size):
    # Tokenizing data
    train_triples, test_triples = train_test_split(0.85, triples)
    # train_tuples, test_tuples = train_test_split(0.85, text_segment_tuples)
    # train_texts, train_labels = zip(*train_tuples)
    # test_texts, test_labels = zip(*test_tuples)
    print('Data split into triples.')

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    # train_tokenized = tokenizer(train_texts, truncation = True, padding = "max_length")
    # test_tokenized = tokenizer(test_texts, truncation = True, padding = "max_length")


    # Creating datasets in the form required by Trainer
    train_dataset = TripleDataset(train_triples, tokenizer)
    test_dataset = TripleDataset(test_triples, tokenizer)

    print('Datasets created.')

    #Setting up the trainer

    training_args = TrainingArguments(output_dir="trainers/{}".format(train_args_file), 
                                    report_to="none",
                                    num_train_epochs=3,
                                    per_device_train_batch_size=8,
                                    logging_strategy="epoch",
                                    evaluation_strategy="epoch",
                                    learning_rate=3e-5
    )


    trainer = Trainer(
        model=pretrained_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = test_dataset,
        data_collator=collate_fn,
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

claim_triples = build_triplets_from_labels(data, target_tag_value=1)
model_claims = HierarchicalTransformerDistilBERT(num_labels=5).to("cuda")
train_model(claim_triples, 'claim_classifier_training_with_context', model_claims, 'ClaimClassifierWithContext', num_epochs = 5, batch_size = 8)

premise_triples = build_triplets_from_labels(data, target_tag_value=2)
model_premises = HierarchicalTransformerDistilBERT(num_labels=7).to("cuda")
train_model(premise_triples, 'premise_classifier_training_with_context', model_premises, 'PremiseClassifierWithContext', num_epochs = 8, batch_size = 8)