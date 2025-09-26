import random
import numpy as np
try:
    from final.constants import CONJUNCTION_LIST, ARG_COMPS, SEM_TYPES
except ImportError:
    from constants import CONJUNCTION_LIST, ARG_COMPS, SEM_TYPES
from torch.utils.data import Dataset
import torch
from transformers import DistilBertTokenizer

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

def split_text(texts):
    sentence_list = [text.split('.') for text in texts]
    sentence_list = [[sentence.strip() for sentence in sentences if len(sentence.strip())] for sentences in sentence_list ]
    text_segment_list = []
    for sentences in sentence_list:
        text_segments = []
        for sentence in sentences:
            words = sentence.split(' ')
            segments = []
            segment = ''
            for word in words:
                if word.lower() not in CONJUNCTION_LIST:
                    segment = segment + ' ' + word if len(segment) else segment + word
                else:
                    if len(segment):
                        segments.append(segment)
                    segment = word
            if len(segment):
                segments.append(segment)
            text_segments.extend(segments)
        text_segment_list.append(text_segments)
    return text_segment_list

def train_test_split(percentage, text_tuples):
    num_train_examples = int(np.floor(len(text_tuples) * percentage))
    train_indices = random.sample(range(0, len(text_tuples)), num_train_examples)
    train_tuples = [text_tuples[i] for i in range(len(text_tuples)) if i in train_indices]
    test_tuples = [text_tuples[i] for i in range(len(text_tuples)) if i not in train_indices]
    return train_tuples, test_tuples

def add_tags_to_text(segmented_texts, arg_comps, sem_types = [], use_sem_types = True):
    segmented_texts_with_tags = []

    id2label_argcomps = {i: v for i, v in enumerate(ARG_COMPS)}
    id2label_semtypes = {i: v for i, v in enumerate(SEM_TYPES)}

    for i in range(len(segmented_texts)):
        segmented_text_i = segmented_texts[i]
        arg_comps_i = ['[{}]'.format(id2label_argcomps[arg_comp].upper()) for arg_comp in arg_comps[i]]
        if use_sem_types:
            sem_types_i = ['[{}]'.format(id2label_semtypes[sem_type].upper()) for sem_type in sem_types[i]]
            text_with_tags = ' '.join([sem_types_i[j] + ' ' + arg_comps_i[j] + ' ' + segmented_text_i[j] for j in range(0, len(segmented_text_i))])
        else:
            text_with_tags = ' '.join([arg_comps_i[j] + ' ' + segmented_text_i[j] for j in range(0, len(segmented_text_i))])

        segmented_texts_with_tags.append(text_with_tags)
    
    return segmented_texts_with_tags

def get_tokenizer(symbolic = None):
    new_tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-cased")
    if symbolic is not None:
        additional_special_tokens = ['[{}]'.format(tag.upper()) for tag in ARG_COMPS]
        additional_special_tokens.extend(['[{}]'.format(tag.upper()) for tag in SEM_TYPES])
        additional_special_tokens = list(set(additional_special_tokens))
        special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
        new_tokenizer.add_special_tokens(special_tokens_dict)
    
    return new_tokenizer

