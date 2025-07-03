import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

df = pd.read_csv('SCOA_subset_for_sequence_labeling_scores.csv')

predicted_arg_comps = df['arg_comp'].to_list()[:60]
actual_arg_comps = df['arg_comp_correct'].to_list()[:60]
predicted_arg_comps=['NA' if x is np.nan else x for x in predicted_arg_comps]
actual_arg_comps=['NA' if x is np.nan else x for x in actual_arg_comps]

predicted_sem_types = df['sem_type'].to_list()[:60]
actual_sem_types = df['sem_type_correct'].to_list()[:60]

predicted_claims = [predicted_sem_types[i] for i in range(0, 60) if predicted_arg_comps[i] == 'claim']
predicted_premises = [predicted_sem_types[i] for i in range(0, 60) if predicted_arg_comps[i] == 'premise']
actual_claims =[actual_sem_types[i] for i in range(0, 60) if predicted_arg_comps[i] == 'claim']
actual_premises = [actual_sem_types[i] for i in range(0, 60) if predicted_arg_comps[i] == 'premise']

print('ArgComp Classifier: ', f1_score(predicted_arg_comps, actual_arg_comps, average = 'macro'))
print('Claim Classifier: ', f1_score(predicted_claims, actual_claims, average = 'macro'))
print('Premise Classifier: ', f1_score(predicted_premises, actual_premises, average = 'macro'))
