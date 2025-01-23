from constants import CONJUNCTION_LIST
import re 

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
