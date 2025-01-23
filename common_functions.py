from evaluation_classes.predictions import Prediction
import random
import numpy as np
from split_arguments import split_text

def train_test_split(percentage, texts, labels):
  if len(texts) != len(labels):
    raise Exception('Number of texts and number of labels do not match.')
  else:
    num_train_examples = int(np.floor(len(texts) * percentage))
    train_indices = random.sample(range(0, len(texts)), num_train_examples)
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []

    for i in range(0, len(texts)):
      if i in train_indices:
        train_texts.append(texts[i])
        train_labels.append(labels[i])
      else:
        test_texts.append(texts[i])
        test_labels.append(labels[i])
    
    return train_texts, train_labels, test_texts, test_labels

def add_tags_to_text(texts, id2label_argcomps = None, id2label_semtypes = None, arg_comps_from_data = None, sem_types_from_data = None):
    prediction_obj = Prediction()

    segmented_texts = split_text(texts)
    segmented_texts_with_tags = []

    for text_segments in segmented_texts:
        if id2label_argcomps:
            if arg_comps_from_data:
              arg_comp_labels = arg_comps_from_data
            else:
              predictions = prediction_obj.predict_arg_comps(text_segments, arg_comps = None)
              arg_comp_labels = prediction_obj.get_predicted_labels(predictions)
            arg_comps = ['[{}]'.format(id2label_argcomps[arg_comp].upper()) for arg_comp in arg_comp_labels]
            if id2label_semtypes: 
                if sem_types_from_data:
                  sem_types = sem_types_from_data
                else:
                  sem_type_predictions = prediction_obj.predict_arg_comps(text_segments, arg_comp_labels)
                  sem_types = prediction_obj.get_predicted_labels(sem_type_predictions)
                sem_types = ['[{}]'.format(id2label_semtypes[sem_type].upper()) for sem_type in sem_types]

                segments_with_tags = ' '.join([sem_types[i] + ' ' + arg_comps[i] + ' ' + text_segments[i] for i in range(0, len(text_segments))])
            else:
                segments_with_tags = ' '.join([arg_comps[i] + ' ' + text_segments[i] for i in range(0, len(text_segments))])
        else:
            segments_with_tags = ' '.join(text_segments)
        segmented_texts_with_tags.append(segments_with_tags)
    
    return segmented_texts_with_tags