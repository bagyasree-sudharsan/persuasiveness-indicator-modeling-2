from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer
import json
# from globals import TextDataset

with open('datasets/processed/AnnotatedCMV/final.json', 'r') as infile:
    data = json.load(infile)
texts = [d['text'] for d in data if isinstance(d['text'], str)]

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased")
model = DistilBertModel.from_pretrained('models/BaselineRegressorFixed')

tokenized_texts = tokenizer(texts[:5], truncation = True, padding = "max_length", return_tensors = "pt")
outputs = model(**tokenized_texts)
print(outputs)